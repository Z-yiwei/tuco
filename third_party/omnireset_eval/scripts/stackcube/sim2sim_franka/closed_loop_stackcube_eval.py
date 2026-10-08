"""Closed-loop MuJoCo eval for the StackCube OmniReset policy.

This is intentionally stricter than the replay scripts:

* reset states are injected only once at episode start;
* actions come only from the loaded policy and live MuJoCo observations;
* success is counted only after opening the gripper and settling;
* the final success window requires stack alignment, low object velocity, and
  no robot/contact support on the insertive cube.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_reconstruct as R
import quat_utils as Q
from compare_stackcube_onestep_mujoco import (
    DECIM,
    KD,
    KP,
    SCALE,
    TAU_MAX,
    VEL_MAX,
    Controller,
    build_model,
    quat_from_aa,
)
from franka_policy import FrankaPolicy
from mlp_util import load_mlp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CO_CURATION_ROOT = Path(
    os.environ.get("CO_CURATION_ROOT", ROOT.parent / "co-curation")
).expanduser()
DEFAULT_CKPT = str(
    CO_CURATION_ROOT
    / "logs/rsl_rl/franka_fr3_gripper_omnireset_agent"
    / "2026-07-11_19-39-00_stackcube_stage2_b3_2500_4gpu_from7800/model_31000.pt"
)
STAGE1_CKPT = str(
    CO_CURATION_ROOT
    / "logs/rsl_rl/franka_fr3_gripper_omnireset_agent"
    / "2026-07-08_17-45-26_stackcube_stage1_2500_4gpu_resume800/model_7800.pt"
)
DEFAULT_RESET = str(
    CO_CURATION_ROOT
    / "Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube"
    / "resets_ObjectAnywhereEEAnywhere.pt"
)
DEFAULT_ROLLOUT = str(
    ROOT
    / "log/active/current_stackcube_sim2sim/source_rollout/"
    "isaac_stackcube_task0_model31000_len120_with_controller.npz"
)
DEFAULT_OUT = "log/active/current_stackcube_sim2sim/mujoco_closed_loop_stackcube_model31000_20260714"

HIST = 5
BLOCKS = {
    "poseA": (0, 6),
    "prev_actions": (30, 7),
    "joint_pos": (65, 9),
    "poseB": (110, 6),
    "poseC": (140, 6),
    "poseD": (170, 6),
}
MAPPING = {
    "poseA": "peg_in_hole",   # insertive cube in receptive cube frame
    "poseB": "ee_pose",
    "poseC": "peg_in_hand",   # insertive cube in hand frame
    "poseD": "hole_in_hand",  # receptive cube in hand frame
}

INSERTIVE_ASSEMBLED_OFFSET_POS = np.array([0.0, 0.0, -0.02], dtype=np.float64)
INSERTIVE_ASSEMBLED_OFFSET_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
RECEPTIVE_ASSEMBLED_OFFSET_POS = np.array([0.0, 0.0, 0.02], dtype=np.float64)
RECEPTIVE_ASSEMBLED_OFFSET_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
SUCCESS_POS_THRESH = 0.005
SUCCESS_ORI_XY_THRESH = 0.025
CENTER_Z_TARGET = 0.04

CONTROL_PROFILES = {
    "stage1_legacy": {
        "scale": np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.2], dtype=np.float64),
        "kp": np.array([200.0, 200.0, 200.0, 3.0, 3.0, 3.0], dtype=np.float64),
        "zeta": np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0], dtype=np.float64),
        "source": (
            "StackCube model_7800 params/env.yaml; same Stage1 OSC definition "
            "as the frozen Peg Stage1 baseline"
        ),
    },
    "stage2_deploy": {
        "scale": np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2], dtype=np.float64),
        "kp": np.array([1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0], dtype=np.float64),
        "zeta": np.ones(6, dtype=np.float64),
        "source": "Stage2 terminal curriculum/deploy OSC definition",
    },
}


class MLPBCPolicyRunner:
    """Stateful 2-step MLP-BC inference matching the IsaacSim evaluator."""

    def __init__(self, checkpoint, use_ema=True, device="cpu"):
        self.model, self.norm = load_mlp(checkpoint, device=device, use_ema=use_ema)
        self.device = torch.device(device)
        self.s_mean = torch.as_tensor(self.norm["s_mean"], dtype=torch.float32, device=self.device)
        self.s_std = torch.as_tensor(self.norm["s_std"], dtype=torch.float32, device=self.device)
        self.a_center = torch.as_tensor(
            self.norm["a_center"], dtype=torch.float32, device=self.device
        )
        self.a_scale = torch.as_tensor(
            self.norm["a_scale"], dtype=torch.float32, device=self.device
        )
        self.n_obs = int(self.norm["n_obs"])
        self.use_ema = bool(use_ema)
        self.obs_history = None

    def reset(self):
        self.obs_history = None

    def __call__(self, obs_tensor):
        obs = obs_tensor.to(device=self.device, dtype=torch.float32)
        if obs.ndim != 2 or obs.shape[0] != 1:
            raise ValueError(f"MLPBCPolicyRunner expects [1, obs_dim], got {tuple(obs.shape)}")
        current = obs[0]
        if self.obs_history is None:
            self.obs_history = current.unsqueeze(0).repeat(self.n_obs, 1)
        else:
            self.obs_history = torch.cat(
                [self.obs_history[1:], current.unsqueeze(0)], dim=0
            )
        obs_normalized = (self.obs_history - self.s_mean) / self.s_std
        action_normalized = self.model(obs_normalized.unsqueeze(0))
        return action_normalized * self.a_scale + self.a_center

    def metadata(self):
        return {
            "type": "mlp_bc",
            "weights": "ema" if self.use_ema else "raw",
            "n_obs_steps": self.n_obs,
            "head_key": self.norm["head_key"],
        }


def load_policy(args):
    if args.policy_type == "rsl_rl":
        policy = FrankaPolicy.load_from_checkpoint(args.checkpoint)
        return policy, {
            "type": "rsl_rl",
            "checkpoint_iteration": int(policy.ckpt_iter),
            "weights": "actor_mean",
        }
    if args.checkpoint == DEFAULT_CKPT:
        raise ValueError("--policy_type mlp_bc requires an explicit MLP --checkpoint")
    policy = MLPBCPolicyRunner(
        args.checkpoint, use_ema=not args.mlp_no_ema, device=args.policy_device
    )
    metadata = policy.metadata()
    metadata["checkpoint_iteration"] = -1
    return policy, metadata


def configure_control_profile(name):
    """Select one complete action/controller interface; never mix stages."""
    global SCALE, KP, KD
    cfg = CONTROL_PROFILES[name]
    SCALE = cfg["scale"].copy()
    KP = cfg["kp"].copy()
    KD = 2.0 * np.sqrt(KP) * cfg["zeta"]
    return {
        "name": name,
        "scale": SCALE.copy(),
        "kp": KP.copy(),
        "zeta": cfg["zeta"].copy(),
        "kd": KD.copy(),
        "source": cfg["source"],
    }


def file_md5(path, block_size=1 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def quat_angle(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)))


def euler_xyz_from_quat(q):
    """Return XYZ fixed-angle equivalent used only for the official XY tilt metric."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    pitch_y = np.arcsin(np.clip(t2, -1.0, 1.0))
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z], dtype=np.float64)


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def as_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def first_sample(seq, index):
    return np.asarray(as_np(seq[index]), dtype=np.float64).copy()


def root_frame_pose(root_pose, pose):
    pos, quat = Q.subtract_frame_transforms(root_pose[:3], root_pose[3:7], pose[:3], pose[3:7])
    return np.concatenate([pos, quat])


def root_frame_velocity(root_pose, vel):
    """Rotate a [linear, angular] world velocity into the robot-root frame."""
    q_inv = Q.quat_inv(root_pose[3:7])
    out = np.asarray(vel, dtype=np.float64).copy()
    out[:3] = Q.quat_apply(q_inv, out[:3])
    out[3:6] = Q.quat_apply(q_inv, out[3:6])
    return out


def load_reset_state(reset_path, index):
    d = torch.load(reset_path, map_location="cpu", weights_only=False)["initial_state"]
    robot = d["articulation"]["robot"]
    ins = d["rigid_object"]["insertive_object"]
    rec = d["rigid_object"]["receptive_object"]
    root = first_sample(robot["root_pose"], index)
    q9 = first_sample(robot["joint_position"], index)
    qvel9 = first_sample(robot["joint_velocity"], index)
    insertive_pose = root_frame_pose(root, first_sample(ins["root_pose"], index))
    receptive_pose = root_frame_pose(root, first_sample(rec["root_pose"], index))
    insertive_vel = root_frame_velocity(root, first_sample(ins["root_velocity"], index))
    return {
        "index": int(index),
        "robot_root_pose_source": root,
        "q9": q9,
        "qvel9": qvel9,
        "insertive_pose": insertive_pose,
        "insertive_vel": insertive_vel,
        "receptive_pose": receptive_pose,
    }


def reset_count(reset_path):
    d = torch.load(reset_path, map_location="cpu", weights_only=False)["initial_state"]
    return len(d["articulation"]["robot"]["joint_position"])


class StackCubeObsBuilder:
    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.fk = R.FrankaFK()
        self.hist = {name: deque(maxlen=HIST) for name in BLOCKS}

    def _terms(self, data):
        raw = np.zeros(57, dtype=np.float64)
        raw[0:9] = data.qpos[:9]
        raw[18:21] = data.xpos[self.ctrl.root]
        raw[21:25] = data.xquat[self.ctrl.root]
        raw[31:34] = data.xpos[self.ctrl.obj]
        raw[34:38] = data.xquat[self.ctrl.obj]
        raw[44:47] = data.xpos[self.ctrl.receptive]
        raw[47:51] = data.xquat[self.ctrl.receptive]
        pose, _ = R.pose_terms_from_raw(raw, self.fk)
        return pose

    def reset(self, data):
        pose = self._terms(data)
        vals = {
            "poseA": pose[MAPPING["poseA"]],
            "poseB": pose[MAPPING["poseB"]],
            "poseC": pose[MAPPING["poseC"]],
            "poseD": pose[MAPPING["poseD"]],
            "joint_pos": data.qpos[:9].copy(),
            "prev_actions": np.zeros(7, dtype=np.float64),
        }
        for name in BLOCKS:
            self.hist[name].clear()
            for _ in range(HIST):
                self.hist[name].append(vals[name].copy())

    def step(self, data, prev_action):
        pose = self._terms(data)
        vals = {
            "poseA": pose[MAPPING["poseA"]],
            "poseB": pose[MAPPING["poseB"]],
            "poseC": pose[MAPPING["poseC"]],
            "poseD": pose[MAPPING["poseD"]],
            "joint_pos": data.qpos[:9].copy(),
            "prev_actions": np.asarray(prev_action, dtype=np.float64).copy(),
        }
        for name in BLOCKS:
            self.hist[name].append(vals[name])
        obs = np.zeros(200, dtype=np.float32)
        for name, (start, width) in BLOCKS.items():
            for hi, val in enumerate(self.hist[name]):
                obs[start + hi * width : start + (hi + 1) * width] = val
        return obs


def configure_controller(model):
    ctrl = Controller(model)
    ctrl.receptive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "receptive_cube")
    ctrl.receptive_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "receptive_cube_geom")
    ctrl.obj_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "insertive_cube_geom")
    return ctrl


def set_initial_state(model, data, ctrl, init):
    data.qpos[:9] = init["q9"]
    data.qvel[:9] = init["qvel9"]
    data.qpos[ctrl.obj_qadr : ctrl.obj_qadr + 3] = init["insertive_pose"][:3]
    data.qpos[ctrl.obj_qadr + 3 : ctrl.obj_qadr + 7] = init["insertive_pose"][3:7]
    data.qvel[ctrl.obj_dadr : ctrl.obj_dadr + 6] = init["insertive_vel"]
    mujoco.mj_forward(model, data)


def step_osc(model, data, ctrl, desired_pos, desired_quat, gripper_ctrl):
    data.ctrl[ctrl.grip_act] = gripper_ctrl
    for _ in range(DECIM):
        ee_pos, ee_quat = ctrl.ee_root(data)
        jac = ctrl.jac_arm(data)
        ee_vel = jac @ data.qvel[:7]
        pos_err = desired_pos - ee_pos
        quat_err = Q.quat_mul(desired_quat, Q.quat_inv(ee_quat))
        aa_err = Q.axis_angle_from_quat(quat_err)
        task_force = KP * np.concatenate([pos_err, aa_err]) + KD * (-ee_vel)
        data.qfrc_applied[:7] = np.clip(jac.T @ task_force, -TAU_MAX, TAU_MAX)
        mujoco.mj_step(model, data)
        np.clip(data.qvel[:7], -VEL_MAX, VEL_MAX, out=data.qvel[:7])


def apply_policy_action(model, data, ctrl, action):
    scaled = np.asarray(action[:6], dtype=np.float64) * SCALE
    ee_pos, ee_quat = ctrl.ee_root(data)
    desired_pos = ee_pos + scaled[:3]
    desired_quat = Q.quat_mul(quat_from_aa(scaled[3:6]), ee_quat)
    gripper_ctrl = 0.0 if ctrl.grasp_close(data) else 255.0
    step_osc(model, data, ctrl, desired_pos, desired_quat, gripper_ctrl)
    return desired_pos, desired_quat, gripper_ctrl


def robot_object_contact(model, data, ctrl):
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == ctrl.obj_geom:
            other = c.geom2
        elif c.geom2 == ctrl.obj_geom:
            other = c.geom1
        else:
            continue
        body = model.geom_bodyid[other]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.startswith("link") or name in {"hand", "left_finger", "right_finger"}:
            return True
    return False


def receptive_contact(data, ctrl):
    for i in range(data.ncon):
        c = data.contact[i]
        if {c.geom1, c.geom2} == {ctrl.obj_geom, ctrl.receptive_geom}:
            return True
    return False


def stack_metrics(model, data, ctrl):
    obj_pose = np.concatenate(
        [data.qpos[ctrl.obj_qadr : ctrl.obj_qadr + 3], data.qpos[ctrl.obj_qadr + 3 : ctrl.obj_qadr + 7]]
    )
    rec_pos = data.xpos[ctrl.receptive].copy()
    rec_quat = data.xquat[ctrl.receptive].copy()
    center_rel = Q.pose_in_root(rec_pos, rec_quat, obj_pose[:3], obj_pose[3:7])

    ins_align_pos, ins_align_quat = Q.combine_frame_transforms(
        obj_pose[:3], obj_pose[3:7], INSERTIVE_ASSEMBLED_OFFSET_POS, INSERTIVE_ASSEMBLED_OFFSET_QUAT
    )
    rec_align_pos, rec_align_quat = Q.combine_frame_transforms(
        rec_pos, rec_quat, RECEPTIVE_ASSEMBLED_OFFSET_POS, RECEPTIVE_ASSEMBLED_OFFSET_QUAT
    )
    align_rel_pos, align_rel_quat = Q.subtract_frame_transforms(
        rec_align_pos, rec_align_quat, ins_align_pos, ins_align_quat
    )
    euler = euler_xyz_from_quat(align_rel_quat)
    euler_xy = abs(wrap_pi(euler[0])) + abs(wrap_pi(euler[1]))
    pos_dist = float(np.linalg.norm(align_rel_pos))

    lin_vel = data.qvel[ctrl.obj_dadr : ctrl.obj_dadr + 3].copy()
    ang_vel = data.qvel[ctrl.obj_dadr + 3 : ctrl.obj_dadr + 6].copy()
    no_robot_contact = not robot_object_contact(model, data, ctrl)
    has_receptive_contact = receptive_contact(data, ctrl)
    center_lateral = float(np.linalg.norm(center_rel[:2]))
    center_z_err = float(abs(center_rel[2] - CENTER_Z_TARGET))
    proxy_pose = pos_dist < SUCCESS_POS_THRESH and euler_xy < SUCCESS_ORI_XY_THRESH
    strict_pose = center_lateral < SUCCESS_POS_THRESH and center_z_err < SUCCESS_POS_THRESH and euler_xy < SUCCESS_ORI_XY_THRESH
    return {
        "proxy_pose": bool(proxy_pose),
        "strict_pose": bool(strict_pose),
        "align_pos_dist_m": pos_dist,
        "align_euler_xy_rad": float(euler_xy),
        "align_full_rot_deg": float(np.degrees(quat_angle(align_rel_quat, np.array([1.0, 0.0, 0.0, 0.0])))),
        "center_lateral_m": center_lateral,
        "center_z_m": float(center_rel[2]),
        "center_z_err_m": center_z_err,
        "lin_vel_mps": float(np.linalg.norm(lin_vel)),
        "ang_vel_radps": float(np.linalg.norm(ang_vel)),
        "no_robot_contact": bool(no_robot_contact),
        "receptive_contact": bool(has_receptive_contact),
    }


def label_frame(img, text):
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, pil.width, 24], fill=(0, 0, 0))
        draw.text((8, 5), text, fill=(255, 255, 255))
        return np.asarray(pil)
    except Exception:
        return img


def parse_render_episodes(s):
    if not s:
        return set()
    return {int(x) for x in s.split(",") if x.strip()}


def validate_rollout_obs(rollout_path):
    ref = np.load(rollout_path, allow_pickle=True)
    root = ref["robot_root_state_w"][0, :7]
    obj = ref["insertive_root_state_w"][0].copy()
    rec = np.concatenate([ref["receptive_root_pos_w"][0], ref["receptive_root_quat_w"][0]])
    obj[:7] = root_frame_pose(root, obj[:7])
    rec = root_frame_pose(root, rec)
    model = build_model(rec)
    ctrl = configure_controller(model)
    data = mujoco.MjData(model)
    init = {
        "q9": ref["robot_joint_pos"][0],
        "qvel9": ref["robot_joint_vel"][0],
        "insertive_pose": obj[:7],
        "insertive_vel": root_frame_velocity(root, obj[7:13]),
    }
    set_initial_state(model, data, ctrl, init)
    obb = StackCubeObsBuilder(ctrl)
    obb.reset(data)
    obs = obb.step(data, np.zeros(7, dtype=np.float32))
    diff = obs - ref["observation"][0]
    return {"l2": float(np.linalg.norm(diff)), "max_abs": float(np.max(np.abs(diff)))}


def run_episode(args, policy, ep, reset_index, render=False, capture_trajectory=False):
    if hasattr(policy, "reset"):
        policy.reset()
    init = load_reset_state(args.reset_state, reset_index)
    model = build_model(init["receptive_pose"])
    ctrl = configure_controller(model)
    data = mujoco.MjData(model)
    set_initial_state(model, data, ctrl, init)
    obb = StackCubeObsBuilder(ctrl)
    obb.reset(data)
    prev_action = np.zeros(7, dtype=np.float32)

    renderer = None
    writer = None
    video_path = None
    cam_id = -1
    if render:
        renderer = mujoco.Renderer(model, args.height, args.width)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        video_path = Path(args.out) / f"closed_loop_ep{ep:03d}_reset{reset_index:04d}_{args.camera}.mp4"
        writer = imageio.get_writer(video_path, fps=args.fps, macro_block_size=1)

    best_proxy_dist = float("inf")
    best_strict_dist = float("inf")
    first_proxy_step = -1
    first_strict_step = -1
    actions = []
    states = []
    insertive_positions = []
    robot_contacts = []
    policy_samples = []

    hold_pos = hold_quat = None
    for t in range(args.horizon):
        obs = obb.step(data, prev_action)
        if capture_trajectory:
            states.append(obs.copy())
            insertive_positions.append(
                data.qpos[ctrl.obj_qadr : ctrl.obj_qadr + 3].astype(np.float32).copy()
            )
            robot_contacts.append(np.float32(robot_object_contact(model, data, ctrl)))
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).float().unsqueeze(0)).cpu().numpy()[0]
        prev_action = action.astype(np.float32)
        hold_pos, hold_quat, gripper_ctrl = apply_policy_action(model, data, ctrl, action)
        m = stack_metrics(model, data, ctrl)
        best_proxy_dist = min(best_proxy_dist, m["align_pos_dist_m"])
        best_strict_dist = min(best_strict_dist, m["center_z_err_m"] + m["center_lateral_m"])
        if m["proxy_pose"] and first_proxy_step < 0:
            first_proxy_step = t + 1
        if m["strict_pose"] and first_strict_step < 0:
            first_strict_step = t + 1
        actions.append(action.astype(np.float32))
        if t % max(args.sample_every, 1) == 0:
            policy_samples.append({"phase": "policy", "step": t + 1, **m})
        if writer is not None and (t % args.render_every == 0 or t == args.horizon - 1):
            renderer.update_scene(data, camera=cam_id)
            img = renderer.render()
            text = (
                f"ep={ep} reset={reset_index} policy {t+1}/{args.horizon} "
                f"strict={int(m['strict_pose'])} lat={m['center_lateral_m']*1000:.1f}mm "
                f"z={m['center_z_m']*1000:.1f}mm contact_robot={int(not m['no_robot_contact'])}"
            )
            writer.append_data(label_frame(img, text))
        if args.release_on_first_proxy and m["proxy_pose"]:
            break

    release_samples = []
    stable_flags = []
    if hold_pos is None:
        hold_pos, hold_quat = ctrl.ee_root(data)
    for t in range(args.release_steps):
        step_osc(model, data, ctrl, hold_pos, hold_quat, 255.0)
        m = stack_metrics(model, data, ctrl)
        stable = (
            m["strict_pose"]
            and m["no_robot_contact"]
            and m["receptive_contact"]
            and m["lin_vel_mps"] < args.stable_lin_vel
            and m["ang_vel_radps"] < args.stable_ang_vel
        )
        stable_flags.append(bool(stable))
        if t % max(args.sample_every, 1) == 0 or t >= args.release_steps - args.stable_window:
            release_samples.append({"phase": "release", "step": t + 1, "stable": bool(stable), **m})
        if writer is not None and (t % args.render_every == 0 or t == args.release_steps - 1):
            renderer.update_scene(data, camera=cam_id)
            img = renderer.render()
            text = (
                f"ep={ep} reset={reset_index} release {t+1}/{args.release_steps} "
                f"stable={int(stable)} lat={m['center_lateral_m']*1000:.1f}mm "
                f"z={m['center_z_m']*1000:.1f}mm v={m['lin_vel_mps']*1000:.1f}mm/s "
                f"robot_contact={int(not m['no_robot_contact'])}"
            )
            writer.append_data(label_frame(img, text))

    if writer is not None:
        writer.close()

    final = stack_metrics(model, data, ctrl)
    stable_after_release = (
        len(stable_flags) >= args.stable_window and all(stable_flags[-args.stable_window :])
    )
    result = {
        "episode": int(ep),
        "reset_index": int(reset_index),
        "horizon": int(args.horizon),
        "policy_steps_executed": int(len(actions)),
        "release_steps": int(args.release_steps),
        "stable_window": int(args.stable_window),
        "success_stable_after_release": bool(stable_after_release),
        "success_proxy_final": bool(final["proxy_pose"]),
        "success_strict_final": bool(final["strict_pose"]),
        "first_proxy_step": int(first_proxy_step),
        "first_strict_step": int(first_strict_step),
        "best_proxy_align_pos_dist_m": float(best_proxy_dist),
        "best_strict_center_lateral_plus_zerr_m": float(best_strict_dist),
        "final": final,
        "video": str(video_path) if video_path is not None else "",
        "actions": np.asarray(actions, dtype=np.float32),
        "samples": policy_samples + release_samples,
    }
    if capture_trajectory:
        result.update(
            {
                "states": np.asarray(states, dtype=np.float32),
                "insertive_positions": np.asarray(insertive_positions, dtype=np.float32),
                "robot_contacts": np.asarray(robot_contacts, dtype=np.float32),
            }
        )
    return result


def write_outputs(args, results, obs_validation, reset_total, control_profile, policy_metadata):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    compact = []
    for r in results:
        row = {
            k: v
            for k, v in r.items()
            if k
            not in {
                "actions",
                "samples",
                "final",
                "states",
                "insertive_positions",
                "robot_contacts",
            }
        }
        row.update({f"final_{k}": v for k, v in r["final"].items()})
        compact.append(row)

    jsonl_path = out / "closed_loop_stackcube_episodes.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in compact:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    csv_path = out / "closed_loop_stackcube_episodes.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(compact[0].keys()) if compact else [])
        if compact:
            writer.writeheader()
            writer.writerows(compact)

    npz_path = out / "closed_loop_stackcube_eval.npz"
    np.savez_compressed(
        npz_path,
        reset_indices=np.asarray([r["reset_index"] for r in results], dtype=np.int32),
        stable_success=np.asarray([r["success_stable_after_release"] for r in results], dtype=np.bool_),
        strict_final=np.asarray([r["success_strict_final"] for r in results], dtype=np.bool_),
        proxy_final=np.asarray([r["success_proxy_final"] for r in results], dtype=np.bool_),
        final_center_lateral_m=np.asarray([r["final"]["center_lateral_m"] for r in results], dtype=np.float32),
        final_center_z_m=np.asarray([r["final"]["center_z_m"] for r in results], dtype=np.float32),
        final_align_pos_dist_m=np.asarray([r["final"]["align_pos_dist_m"] for r in results], dtype=np.float32),
        final_align_euler_xy_rad=np.asarray([r["final"]["align_euler_xy_rad"] for r in results], dtype=np.float32),
    )

    n = len(results)
    stable = sum(r["success_stable_after_release"] for r in results)
    strict = sum(r["success_strict_final"] for r in results)
    proxy = sum(r["success_proxy_final"] for r in results)
    summary = {
        "definition": (
            "closed-loop MuJoCo eval: official reset state at t=0 only; live MuJoCo obs; "
            "policy actions from checkpoint; success requires release+settle stable window, "
            "strict stack pose, receptive contact, low object velocity, and no robot-object contact"
        ),
        "checkpoint": args.checkpoint,
        "checkpoint_md5": file_md5(args.checkpoint),
        "checkpoint_iteration": int(policy_metadata["checkpoint_iteration"]),
        "policy_type": policy_metadata["type"],
        "policy_metadata": policy_metadata,
        "runtime": {"torch_threads": int(args.torch_threads)},
        "profile": args.profile,
        "control_profile": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in control_profile.items()
        },
        "mujoco_profile": {
            "definition": (
                "legacy StackCube MuJoCo model shared by the prior model_31000 eval; "
                "empirical tendon gripper, 30g insertive cube, no B3 arm sysid"
            ),
            "state_restore": "episode initialization only",
            "arm_velocity_limit": "post-step clip matching the frozen Peg Stage1 evaluator",
        },
        "reset_state": args.reset_state,
        "reset_total": int(reset_total),
        "episodes": int(n),
        "start_index": int(args.start_index),
        "stride": int(args.stride),
        "horizon": int(args.horizon),
        "release_on_first_proxy": bool(args.release_on_first_proxy),
        "release_steps": int(args.release_steps),
        "stable_window": int(args.stable_window),
        "thresholds": {
            "success_position_m": SUCCESS_POS_THRESH,
            "success_orientation_xy_rad": SUCCESS_ORI_XY_THRESH,
            "center_z_target_m": CENTER_Z_TARGET,
            "stable_lin_vel_mps": args.stable_lin_vel,
            "stable_ang_vel_radps": args.stable_ang_vel,
        },
        "obs_validation": obs_validation,
        "stable_after_release_sr": float(stable / max(n, 1)),
        "stable_after_release_successes": int(stable),
        "strict_final_pose_sr_diagnostic": float(strict / max(n, 1)),
        "proxy_final_pose_sr_diagnostic": float(proxy / max(n, 1)),
        "final_center_lateral_mm_mean": float(np.mean([r["final"]["center_lateral_m"] for r in results]) * 1000.0)
        if results
        else None,
        "final_center_lateral_mm_median": float(np.median([r["final"]["center_lateral_m"] for r in results]) * 1000.0)
        if results
        else None,
        "final_center_z_mm_mean": float(np.mean([r["final"]["center_z_m"] for r in results]) * 1000.0)
        if results
        else None,
        "outputs": {
            "summary": str(out / "closed_loop_stackcube_summary.json"),
            "episodes_jsonl": str(jsonl_path),
            "episodes_csv": str(csv_path),
            "npz": str(npz_path),
            "videos": [r["video"] for r in results if r.get("video")],
        },
    }
    summary_path = out / "closed_loop_stackcube_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def aggregate_parts(args):
    """Merge disjoint startN eval chunks without rerunning simulation."""
    parts_dir = Path(args.aggregate_parts).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    part_summaries = []
    compact = []
    for summary_path in sorted(parts_dir.glob("start*/closed_loop_stackcube_summary.json")):
        with summary_path.open(encoding="utf-8") as file:
            part_summaries.append(json.load(file))
        jsonl_path = summary_path.parent / "closed_loop_stackcube_episodes.jsonl"
        with jsonl_path.open(encoding="utf-8") as file:
            compact.extend(json.loads(line) for line in file if line.strip())
    if not part_summaries:
        raise FileNotFoundError(f"no start*/closed_loop_stackcube_summary.json under {parts_dir}")
    compact.sort(key=lambda row: int(row["reset_index"]))
    reset_indices = [int(row["reset_index"]) for row in compact]
    expected = [
        (args.start_index + i * args.stride) % int(part_summaries[0]["reset_total"])
        for i in range(args.episodes)
    ]
    if reset_indices != expected:
        raise ValueError(f"part reset indices mismatch: got {reset_indices}, expected {expected}")
    for episode, row in enumerate(compact):
        row["episode"] = episode

    jsonl_path = out / "closed_loop_stackcube_episodes.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in compact:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    csv_path = out / "closed_loop_stackcube_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)
    npz_path = out / "closed_loop_stackcube_eval.npz"
    np.savez_compressed(
        npz_path,
        reset_indices=np.asarray(reset_indices, dtype=np.int32),
        stable_success=np.asarray(
            [row["success_stable_after_release"] for row in compact], dtype=np.bool_
        ),
        strict_final=np.asarray([row["success_strict_final"] for row in compact], dtype=np.bool_),
        proxy_final=np.asarray([row["success_proxy_final"] for row in compact], dtype=np.bool_),
        final_center_lateral_m=np.asarray(
            [row["final_center_lateral_m"] for row in compact], dtype=np.float32
        ),
        final_center_z_m=np.asarray(
            [row["final_center_z_m"] for row in compact], dtype=np.float32
        ),
        final_align_pos_dist_m=np.asarray(
            [row["final_align_pos_dist_m"] for row in compact], dtype=np.float32
        ),
        final_align_euler_xy_rad=np.asarray(
            [row["final_align_euler_xy_rad"] for row in compact], dtype=np.float32
        ),
    )

    first = part_summaries[0]
    stable = sum(bool(row["success_stable_after_release"]) for row in compact)
    strict = sum(bool(row["success_strict_final"]) for row in compact)
    proxy = sum(bool(row["success_proxy_final"]) for row in compact)
    summary = {
        key: first[key]
        for key in (
            "definition",
            "checkpoint",
            "checkpoint_md5",
            "checkpoint_iteration",
            "profile",
            "control_profile",
            "mujoco_profile",
            "reset_state",
            "reset_total",
            "horizon",
            "release_steps",
            "stable_window",
            "thresholds",
            "obs_validation",
        )
    }
    for key in ("policy_type", "policy_metadata", "runtime"):
        if key in first:
            summary[key] = first[key]
    summary.update({
        "parts_dir": str(parts_dir),
        "episodes": len(compact),
        "start_index": args.start_index,
        "stride": args.stride,
        "stable_after_release_sr": stable / len(compact),
        "stable_after_release_successes": stable,
        "strict_final_pose_sr_diagnostic": strict / len(compact),
        "proxy_final_pose_sr_diagnostic": proxy / len(compact),
        "final_center_lateral_mm_mean": float(
            np.mean([row["final_center_lateral_m"] for row in compact]) * 1000.0
        ),
        "final_center_lateral_mm_median": float(
            np.median([row["final_center_lateral_m"] for row in compact]) * 1000.0
        ),
        "final_center_z_mm_mean": float(
            np.mean([row["final_center_z_m"] for row in compact]) * 1000.0
        ),
        "stable_success_reset_indices": [
            row["reset_index"] for row in compact if row["success_stable_after_release"]
        ],
        "outputs": {
            "summary": str(out / "closed_loop_stackcube_summary.json"),
            "episodes_jsonl": str(jsonl_path),
            "episodes_csv": str(csv_path),
            "npz": str(npz_path),
            "videos": [],
        },
    })
    summary_path = out / "closed_loop_stackcube_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    print(
        f"[stackcube-closed-loop] aggregate stable_after_release SR="
        f"{summary['stable_after_release_sr']:.3f} "
        f"({stable}/{len(compact)}) -> {summary_path}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument(
        "--policy_type",
        choices=("rsl_rl", "mlp_bc"),
        default="rsl_rl",
        help="checkpoint interface; mlp_bc uses checkpoint normalization and rolling obs history",
    )
    ap.add_argument(
        "--policy_device",
        default="cpu",
        help="torch device for policy inference; MuJoCo remains on CPU",
    )
    ap.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="0 keeps the ambient PyTorch runtime; positive values freeze CPU inference threads",
    )
    ap.add_argument(
        "--mlp_no_ema",
        action="store_true",
        help="use raw MLP weights instead of the canonical EMA weights",
    )
    ap.add_argument(
        "--profile",
        choices=tuple(CONTROL_PROFILES),
        default="stage2_deploy",
        help="complete action-scale and OSC-gain profile; do not mix checkpoint stages",
    )
    ap.add_argument("--reset_state", default=DEFAULT_RESET)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--release_steps", type=int, default=60)
    ap.add_argument(
        "--release_on_first_proxy",
        action="store_true",
        help=(
            "stop policy execution at the first official proxy-pose success and "
            "immediately enter the forced-open release/settle phase; use this for "
            "datasets truncated at their first success-causing action"
        ),
    )
    ap.add_argument("--stable_window", type=int, default=10)
    ap.add_argument("--stable_lin_vel", type=float, default=0.01)
    ap.add_argument("--stable_ang_vel", type=float, default=0.2)
    ap.add_argument("--validate_rollout_obs", default=DEFAULT_ROLLOUT)
    ap.add_argument("--obs_max_abs_fail", type=float, default=1e-4)
    ap.add_argument("--render_episodes", default="0,1,2")
    ap.add_argument("--render_every", type=int, default=2)
    ap.add_argument("--sample_every", type=int, default=5)
    ap.add_argument("--camera", default="cam_front")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument(
        "--aggregate_parts",
        default=None,
        help="merge disjoint startN output directories under this path",
    )
    args = ap.parse_args()

    if args.aggregate_parts:
        aggregate_parts(args)
        return

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    control_profile = configure_control_profile(args.profile)
    if (
        args.policy_type == "rsl_rl"
        and args.profile == "stage1_legacy"
        and args.checkpoint == DEFAULT_CKPT
    ):
        args.checkpoint = STAGE1_CKPT

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    obs_validation = None
    if args.validate_rollout_obs:
        obs_validation = validate_rollout_obs(args.validate_rollout_obs)
        print(
            f"[stackcube-closed-loop] obs validation: "
            f"l2={obs_validation['l2']:.6g} max_abs={obs_validation['max_abs']:.6g}"
        )
        if obs_validation["max_abs"] > args.obs_max_abs_fail:
            raise RuntimeError(f"obs reconstruction failed: {obs_validation}")

    reset_total = reset_count(args.reset_state)
    indices = [(args.start_index + i * args.stride) % reset_total for i in range(args.episodes)]
    print(f"[stackcube-closed-loop] reset_total={reset_total} indices={indices}")
    policy, policy_metadata = load_policy(args)
    print(
        f"[stackcube-closed-loop] profile={args.profile} "
        f"policy={policy_metadata['type']} "
        f"weights={policy_metadata['weights']} "
        f"iter={policy_metadata['checkpoint_iteration']} "
        f"scale={SCALE.tolist()} kp={KP.tolist()}"
    )

    render_eps = parse_render_episodes(args.render_episodes)
    results = []
    for ep, reset_index in enumerate(indices):
        res = run_episode(args, policy, ep, reset_index, render=ep in render_eps)
        results.append(res)
        final = res["final"]
        print(
            f"[stackcube-closed-loop] ep {ep:03d} reset={reset_index:04d} "
            f"stable={int(res['success_stable_after_release'])} "
            f"strict_final={int(res['success_strict_final'])} "
            f"proxy_final={int(res['success_proxy_final'])} "
            f"lat={final['center_lateral_m']*1000:.1f}mm "
            f"z={final['center_z_m']*1000:.1f}mm "
            f"align={final['align_pos_dist_m']*1000:.1f}mm "
            f"xy={final['align_euler_xy_rad']:.4f} "
            f"v={final['lin_vel_mps']*1000:.1f}mm/s "
            f"robot_contact={int(not final['no_robot_contact'])}"
        )

    summary = write_outputs(
        args, results, obs_validation, reset_total, control_profile, policy_metadata
    )
    print(
        f"[stackcube-closed-loop] stable_after_release SR="
        f"{summary['stable_after_release_sr']:.3f} "
        f"({summary['stable_after_release_successes']}/{summary['episodes']})"
    )
    print(f"[stackcube-closed-loop] summary -> {summary['outputs']['summary']}")


if __name__ == "__main__":
    main()
