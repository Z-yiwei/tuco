#!/usr/bin/env python3
"""Collect/evaluate the active Peg contract in MuJoCo (simulator A).

40 mm hole, Stage-1 OSC, 200-D state, 7-D action, 160 policy steps,
and success = position < 1 cm AND |roll| + |pitch| < 0.1 rad (yaw ignored).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim_eval"))

import closed_loop_eval as CL  # noqa: E402
import quat_utils as Q  # noqa: E402
from franka_policy import FrankaPolicy  # noqa: E402
from mlp_util import load_mlp  # noqa: E402

PROFILE = {
    "profile": "stage1_active_big_legacy_gripper",
    "controller": "stage1",
    "peg_mass": 0.02,
    "peg_collision": "box",
    "peg_contact_solref": [0.001, 1.0],
    "hole_collision": "box_ring_big",
    # model_4550's validated MuJoCo actuator/contact mapping.  The Isaac-side
    # 1000/14/60 values must not be converted through the Stage-2 B3 profile.
    "gripper": dict(CL.LEGACY_GRIPPER),
    "sample_gripper": False,
    "physics_substeps": 16,
    "finger_velocity_limit": 0.025,
    "real_finger_inertia": True,
}
SCALE = CL.STAGE1_SCALE
KP = CL.STAGE1_KP
KD = 2.0 * np.sqrt(KP) * CL.STAGE1_ZETA
TARGET_POS = np.array([0.0, 0.0, CL.ASSEMBLED_Z])
_WORKER_TEACHER = None
_WORKER_STUDENT = None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_files(pattern: str) -> list[str]:
    def seed(path: str) -> int:
        return int(path.replace("\\", "/").split("/seed", 1)[1].split("/", 1)[0])

    return sorted(glob.glob(pattern), key=seed)


def reset_entries(path: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    state = torch.load(path, map_location="cpu", weights_only=False)["initial_state"]
    robot = state["articulation"]["robot"]["joint_position"]
    peg = state["rigid_object"]["insertive_object"]["root_pose"]
    hole = state["rigid_object"]["receptive_object"]["root_pose"]
    assert len(robot) == len(peg) == len(hole)
    return [(np.asarray(robot[i]), np.asarray(peg[i]), np.asarray(hole[i]), i)
            for i in range(len(robot))]


def initial_state(source) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not isinstance(source, (str, os.PathLike)):
        return source
    path = str(source)
    blob = torch.load(path, map_location="cpu", weights_only=False)["initial_state"]
    q9 = np.asarray(blob["articulation"]["robot"]["joint_position"]).reshape(-1)
    peg = np.asarray(blob["rigid_object"]["insertive_object"]["root_pose"]).reshape(-1)
    hole = np.asarray(blob["rigid_object"]["receptive_object"]["root_pose"]).reshape(-1)
    seed = int(path.replace("\\", "/").split("/seed", 1)[1].split("/", 1)[0])
    return q9, peg, hole, seed


def euler_xyz_from_quat(q: np.ndarray) -> np.ndarray:
    """IsaacLab-compatible XYZ Euler angles for a wxyz quaternion."""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def active_success(d: mujoco.MjData, ctrl: CL.Controller) -> tuple[bool, float, float]:
    pos, quat = Q.subtract_frame_transforms(
        d.xpos[ctrl.hole], d.xquat[ctrl.hole], d.xpos[ctrl.peg], d.xquat[ctrl.peg]
    )
    pos_err = float(np.linalg.norm(pos - TARGET_POS))
    euler = euler_xyz_from_quat(quat)
    ori_err = float(abs(np.arctan2(np.sin(euler[0]), np.cos(euler[0]))) +
                    abs(np.arctan2(np.sin(euler[1]), np.cos(euler[1]))))
    return pos_err < 0.01 and ori_err < 0.1, pos_err, ori_err


def make_world(q9: np.ndarray, peg: np.ndarray, hole: np.ndarray):
    model = CL.build_model(hole[:3], hole[3:7], PROFILE)
    ctrl = CL.Controller(model)
    obs_builder = CL.ObsBuilder(ctrl)
    data = mujoco.MjData(model)
    peg_joint = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg")
    ]
    data.qpos[:9] = q9
    data.qpos[peg_joint:peg_joint + 3] = peg[:3]
    data.qpos[peg_joint + 3:peg_joint + 7] = peg[3:7]
    mujoco.mj_forward(model, data)
    obs_builder.reset(data)
    return model, ctrl, obs_builder, data


def step_action(model, ctrl, data, action: np.ndarray) -> None:
    scaled = action[:6] * SCALE
    ee_pos, ee_quat = ctrl.ee_root(data)
    desired_pos = ee_pos + scaled[:3]
    desired_quat = Q.quat_mul(CL.quat_from_aa(scaled[3:6]), ee_quat)
    data.ctrl[ctrl.grip_act] = 0.0 if ctrl.grasp_close(data) else 255.0
    substeps = int(PROFILE["physics_substeps"])
    finger_limit = PROFILE.get("finger_velocity_limit")
    if finger_limit is not None:
        finger_limit = float(finger_limit)
    for _ in range(CL.DECIM * substeps):
        finger_before = data.qpos[7:9].copy() if finger_limit is not None else None
        ee_pos, ee_quat = ctrl.ee_root(data)
        jac = ctrl.jac_arm(data)
        ee_vel = jac @ data.qvel[:7]
        quat_err = Q.quat_mul(desired_quat, Q.quat_inv(ee_quat))
        pose_err = np.concatenate([desired_pos - ee_pos, Q.axis_angle_from_quat(quat_err)])
        data.qfrc_applied[:7] = np.clip(
            jac.T @ (KP * pose_err + KD * (-ee_vel)), -CL.TAU_MAX, CL.TAU_MAX
        )
        mujoco.mj_step(model, data)
        np.clip(data.qvel[:7], -CL.VEL_MAX, CL.VEL_MAX, out=data.qvel[:7])
        if finger_limit is not None:
            np.clip(data.qvel[7:9], -finger_limit, finger_limit, out=data.qvel[7:9])
            max_delta = finger_limit * model.opt.timestep
            data.qpos[7:9] = np.clip(
                data.qpos[7:9], finger_before - max_delta, finger_before + max_delta
            )


def rollout_teacher(source, teacher: FrankaPolicy):
    q9, peg, hole, seed = initial_state(source)
    model, ctrl, obs_builder, data = make_world(q9, peg, hole)
    states, actions = [], []
    prev_action = np.zeros(7, dtype=np.float32)
    ever_success = False
    best_pos, best_ori = float("inf"), float("inf")
    for _ in range(CL.EP_LEN):
        obs = obs_builder.step(data, prev_action).astype(np.float32)
        with torch.inference_mode():
            action = teacher(torch.from_numpy(obs)[None]).numpy()[0].astype(np.float32)
        states.append(obs)
        actions.append(action)
        step_action(model, ctrl, data, action)
        ok, pe, oe = active_success(data, ctrl)
        ever_success |= ok
        best_pos, best_ori = min(best_pos, pe), min(best_ori, oe)
        prev_action = action
    return np.stack(states), np.stack(actions), ever_success, seed, best_pos, best_ori


def init_teacher_worker(checkpoint: str, profile: dict | None = None) -> None:
    global _WORKER_TEACHER, PROFILE
    torch.set_num_threads(1)
    if profile is not None:
        PROFILE = profile
    _WORKER_TEACHER = FrankaPolicy.load_from_checkpoint(checkpoint)


def rollout_teacher_worker(source):
    return rollout_teacher(source, _WORKER_TEACHER)


def rollout_teacher_metric_worker(source):
    _, _, ok, seed, pos, ori = rollout_teacher(source, _WORKER_TEACHER)
    return ok, seed, pos, ori


def collect(args) -> None:
    torch.set_num_threads(1)
    source_all = reset_entries(args.reset_pt) if args.reset_pt else snapshot_files(args.snap_glob)
    candidates = source_all[args.candidate_start:args.candidate_stop]
    states, actions, ends, seeds = [], [], [], []
    if args.workers == 1:
        teacher = FrankaPolicy.load_from_checkpoint(args.checkpoint)
        iterator = (rollout_teacher(source, teacher) for source in candidates)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(
            args.workers, initializer=init_teacher_worker, initargs=(args.checkpoint,)
        )
        iterator = pool.imap(rollout_teacher_worker, candidates, chunksize=1)
    for index, result in enumerate(iterator, 1):
        s, a, ok, seed, pos, ori = result
        if ok:
            states.append(s)
            actions.append(a)
            ends.append(len(states) * CL.EP_LEN)
            seeds.append(seed)
        print(f"[collect] {index}/{len(candidates)} seed={seed} success={ok} "
              f"best_pos={pos:.5f} best_rp={ori:.5f} kept={len(states)}/{args.num_demos}", flush=True)
        if len(states) == args.num_demos:
            break
    if pool is not None:
        pool.terminate()
        pool.join()
    if len(states) != args.num_demos:
        raise RuntimeError(f"only collected {len(states)}/{args.num_demos} successful demos")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out), mode="w")
    state = np.concatenate(states).astype(np.float32)
    action = np.concatenate(actions).astype(np.float32)
    root.create_dataset("data/state", data=state, chunks=(1024, 200))
    root.create_dataset("data/action", data=action, chunks=(1024, 7))
    root.create_dataset("meta/episode_ends", data=np.asarray(ends, dtype=np.int64))
    root.create_dataset("meta/source_snapshot_seed", data=np.asarray(seeds, dtype=np.int32))
    root.attrs.update({
        "simulator": "MuJoCo", "data_role": "A fixed training set",
        "num_demos": args.num_demos, "episode_steps": CL.EP_LEN,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint), "profile": PROFILE,
        "success": "position<0.01 and abs(roll)+abs(pitch)<0.1; yaw ignored",
        "candidate_slice": [args.candidate_start, args.candidate_stop],
    })
    print(f"[collect] saved {out}: state={state.shape} action={action.shape}")


def rollout_student(source, checkpoint: str | None = None, record: bool = False):
    if checkpoint is None:
        model_policy, norm = _WORKER_STUDENT
    else:
        model_policy, norm = load_mlp(checkpoint)
    q9, peg, hole, seed = initial_state(source)
    model, ctrl, obs_builder, data = make_world(q9, peg, hole)
    history = collections.deque(maxlen=norm["n_obs"])
    prev_action = np.zeros(7, dtype=np.float32)
    ever_success = False
    final_pos = final_ori = float("inf")
    states, actions = [], []
    for _ in range(CL.EP_LEN):
        obs = obs_builder.step(data, prev_action).astype(np.float32)
        if not history:
            for _ in range(norm["n_obs"]):
                history.append(obs)
        else:
            history.append(obs)
        x = (np.stack(history) - norm["s_mean"]) / norm["s_std"]
        with torch.inference_mode():
            pred = model_policy(torch.from_numpy(x).float()[None]).numpy()[0]
        action = (pred * norm["a_scale"] + norm["a_center"]).astype(np.float32)
        if record:
            states.append(obs)
            actions.append(action)
        step_action(model, ctrl, data, action)
        ok, final_pos, final_ori = active_success(data, ctrl)
        ever_success |= ok
        prev_action = action
    result = (ever_success, seed, final_pos, final_ori)
    if record:
        return result + (np.stack(states), np.stack(actions))
    return result


def init_student_worker(checkpoint: str) -> None:
    global _WORKER_STUDENT
    torch.set_num_threads(1)
    _WORKER_STUDENT = load_mlp(checkpoint)


def rollout_student_worker(source):
    return rollout_student(source, None)


def rollout_student_record_worker(source):
    return rollout_student(source, None, record=True)


def evaluation_sources(args):
    source_all = reset_entries(args.reset_pt) if args.reset_pt else snapshot_files(args.snap_glob)
    if args.eval_source_seeds_zarr:
        source_ids = np.asarray(
            zarr.open(args.eval_source_seeds_zarr, mode="r")["meta/source_snapshot_seed"]
        ).astype(np.int64)
        sources = [source_all[int(source_id)] for source_id in source_ids]
    else:
        sources = source_all[args.eval_start:args.eval_stop]
    if args.num_episodes is not None:
        sources = sources[:args.num_episodes]
    return sources


def evaluate(args) -> None:
    files = evaluation_sources(args)
    if args.save_rollouts and len(args.checkpoints) != 1:
        raise ValueError("--save_rollouts requires exactly one checkpoint")
    results = []
    for checkpoint in args.checkpoints:
        success = 0
        rows = []
        rollout_states, rollout_actions, rollout_success, rollout_seeds = [], [], [], []
        if args.workers == 1:
            iterator = (
                rollout_student(source, checkpoint, record=bool(args.save_rollouts))
                for source in files
            )
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(
                args.workers, initializer=init_student_worker, initargs=(checkpoint,)
            )
            worker = rollout_student_record_worker if args.save_rollouts else rollout_student_worker
            iterator = pool.imap(worker, files, chunksize=1)
        for index, result in enumerate(iterator, 1):
            ok, seed, pos, ori = result[:4]
            if args.save_rollouts:
                rollout_states.append(result[4])
                rollout_actions.append(result[5])
                rollout_success.append(bool(ok))
                rollout_seeds.append(int(seed))
            success += int(ok)
            rows.append({"snapshot_seed": seed, "success": bool(ok),
                         "final_pos_error": pos, "final_roll_pitch_error": ori})
            print(f"[eval] {Path(checkpoint).name} {index}/{len(files)} seed={seed} "
                  f"success={ok} running={success}/{index}", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
        payload = {
            "checkpoint": os.path.abspath(checkpoint), "successes": success,
            "episodes": len(files), "sr": success / len(files), "profile": PROFILE,
            "eval_slice": [args.eval_start, args.eval_stop], "episodes_detail": rows,
        }
        results.append(payload)
        print(f"[RESULT] {checkpoint}: {success}/{len(files)} = {success/len(files):.4f}")
        if args.save_rollouts:
            out = Path(args.save_rollouts)
            out.parent.mkdir(parents=True, exist_ok=True)
            root = zarr.open(str(out), mode="w")
            state = np.concatenate(rollout_states).astype(np.float32)
            action = np.concatenate(rollout_actions).astype(np.float32)
            ends = np.arange(1, len(files) + 1, dtype=np.int64) * CL.EP_LEN
            root.create_dataset("data/state", data=state, chunks=(1024, 200))
            root.create_dataset("data/action", data=action, chunks=(1024, 7))
            root.create_dataset(
                "data/success", data=np.asarray(rollout_success, dtype=bool)
            )
            root.create_dataset("meta/episode_ends", data=ends)
            root.create_dataset(
                "meta/source_snapshot_seed",
                data=np.asarray(rollout_seeds, dtype=np.int32),
            )
            root.attrs.update({
                "eval_contract_id": "mujoco-peg-stage1-active-big-legacygripper-eval-v1",
                "simulator": "MuJoCo",
                "task": "Peg-MuJoCo-Stage1-ActiveBig-v1",
                "reset_type": args.reset_type,
                "seed": int(args.eval_start),
                "num_envs": 1,
                "num_episodes": len(files),
                "success_pos_threshold": 0.01,
                "success_ori_threshold": 0.1,
                "checkpoint": os.path.abspath(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "profile": PROFILE,
                "eval_slice": [args.eval_start, args.eval_stop],
            })
            print(
                f"[rollouts] saved {out}: state={state.shape} action={action.shape} "
                f"success={success}/{len(files)}"
            )
    if args.result_json:
        out = Path(args.result_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")


def evaluate_teacher(args) -> None:
    """Evaluate the RL teacher with the exact collection dynamics and success rule."""
    sources = evaluation_sources(args)
    profile = dict(PROFILE)
    if args.teacher_profile == "active_big_b3_gripper":
        profile["gripper"] = dict(CL.B3_GRIPPER)
    elif args.teacher_profile == "active_big_legacy_gripper":
        profile["gripper"] = dict(CL.LEGACY_GRIPPER)
    elif args.teacher_profile == "legacy_stage1":
        profile.update({
            "profile": "legacy_stage1",
            "peg_contact_solref": [0.02, 1.0],
            "hole_collision": "box_ring",
            "gripper": dict(CL.LEGACY_GRIPPER),
            "physics_substeps": 1,
            "finger_velocity_limit": None,
            "real_finger_inertia": False,
        })
    pool = mp.get_context("spawn").Pool(
        args.workers, initializer=init_teacher_worker, initargs=(args.checkpoint, profile)
    )
    rows = []
    success = 0
    try:
        iterator = pool.imap(rollout_teacher_metric_worker, sources, chunksize=1)
        for index, (ok, seed, pos, ori) in enumerate(iterator, 1):
            success += int(ok)
            rows.append({"snapshot_seed": seed, "success": bool(ok),
                         "final_pos_error": pos, "final_roll_pitch_error": ori})
            print(f"[teacher-eval] {index}/{len(sources)} seed={seed} success={ok} "
                  f"running={success}/{index}", flush=True)
    finally:
        pool.close()
        pool.join()
    payload = {
        "checkpoint": os.path.abspath(args.checkpoint), "successes": success,
        "episodes": len(sources), "sr": success / len(sources), "profile": profile,
        "eval_slice": [args.eval_start, args.eval_stop], "episodes_detail": rows,
    }
    print(f"[TEACHER RESULT] {success}/{len(sources)} = {success/len(sources):.4f}")
    if args.result_json:
        out = Path(args.result_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "eval", "teacher_eval"), required=True)
    parser.add_argument("--snap_glob", default=str(ROOT / "datasets/snapshots_pegHole_upright_yaw_3cm/seed*/demo_0000.pt"))
    parser.add_argument("--reset_pt", default=str(ROOT / "Datasets/OmniReset/Resets/Peg__PegHole/resets_ObjectAnywhereEEAnywhere_upright_yaw_3cm.pt"))
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/teachers/peg_stage1_model4550/model_4550.pt"))
    parser.add_argument(
        "--teacher_profile",
        choices=("active_big", "active_big_b3_gripper", "active_big_legacy_gripper", "legacy_stage1"),
        default="active_big",
        help="Controlled teacher-eval ablation; collection always uses active_big.",
    )
    parser.add_argument("--output", default=str(ROOT / "datasets/peg_reversed_ab/mujoco_A200_active_bighole_legacygrip_seed42.zarr"))
    parser.add_argument("--num_demos", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidate_start", type=int, default=0)
    parser.add_argument("--candidate_stop", type=int, default=5000)
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--eval_start", type=int, default=400)
    parser.add_argument("--eval_stop", type=int, default=500)
    parser.add_argument("--reset_type", default="ObjectAnywhereEEAnywhere_upright_yaw_3cm")
    parser.add_argument(
        "--eval_source_seeds_zarr",
        help="Evaluate the reset row IDs stored in meta/source_snapshot_seed of this zarr.",
    )
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--result_json")
    parser.add_argument("--save_rollouts")
    args = parser.parse_args()
    if args.mode == "collect":
        collect(args)
    elif args.mode == "eval":
        if not args.checkpoints:
            parser.error("--checkpoints is required for --mode eval")
        evaluate(args)
    else:
        evaluate_teacher(args)


if __name__ == "__main__":
    main()
