"""StackCube IsaacSim-to-MuJoCo per-step reset replay.

Input is an IsaacSim rollout exported by co-curation's
``export_stackcube_rollout.py``. For every policy step t:

1. reset MuJoCo to IsaacSim state(t)
2. execute action(t) for one policy step
3. compare MuJoCo state(t+1) to IsaacSim state(t+1)

The video renders LEFT = IsaacSim recorded state visualized kinematically in
MuJoCo, RIGHT = MuJoCo one-step replay.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quat_utils as Q


ROOT = Path(__file__).resolve().parents[3]
PANDA = ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
MIMIC_LF = ROOT / "assets" / "assets_mjcf" / "franka_mimic_leftfinger_col.stl"
MIMIC_RF = ROOT / "assets" / "assets_mjcf" / "franka_mimic_rightfinger_col.stl"

SIM_DT = 1.0 / 120.0
DECIM = 12
CUBE_HALF = 0.02

# Finetune-Play uses FrankaFr3GripperRelativeOSCEvalAction.
SCALE = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2], dtype=np.float64)
KP = np.array([1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0], dtype=np.float64)
ZETA = np.ones(6, dtype=np.float64)
KD = 2.0 * np.sqrt(KP) * ZETA
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)
VEL_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float64)
TCP_OFF = np.array([0.0, 0.0, 0.1034], dtype=np.float64)
LAT_THRESH = 0.02
VERT_THRESH = 0.03


def quat_angle(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)))


def quat_from_aa(axis_angle):
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis_angle / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


def build_model(receptive_pose):
    spec = mujoco.MjSpec.from_file(str(PANDA))
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.impratio = 10.0
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 5
    spec.option.iterations = 150

    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0

    for actuator in spec.actuators:
        if actuator.name in {f"actuator{i}" for i in range(1, 8)}:
            actuator.gainprm = np.zeros_like(np.array(actuator.gainprm))
            actuator.biasprm = np.zeros_like(np.array(actuator.biasprm))
        if actuator.name == "actuator8":
            actuator.gainprm = np.array([0.0627, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            actuator.biasprm = np.array([0, -400.0, -20.0, 0, 0, 0, 0, 0, 0, 0])
            actuator.forcerange = [-60.0, 60.0]

    for bn in ("left_finger", "right_finger"):
        for geom in spec.body(bn).geoms:
            if str(geom.type).endswith("MESH") and geom.contype != 0:
                geom.contype, geom.conaffinity = 0, 0

    spec.add_mesh(name="mimic_lf", file=str(MIMIC_LF))
    spec.add_mesh(name="mimic_rf", file=str(MIMIC_RF))
    for bn, mesh_name in (("left_finger", "mimic_lf"), ("right_finger", "mimic_rf")):
        body = spec.body(bn)
        for geom in body.geoms:
            if str(geom.type).endswith("BOX"):
                geom.contype, geom.conaffinity = 0, 0
        body.add_geom(
            name=f"{bn}_mimic",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            friction=[2.0, 0.1, 0.05],
            condim=6,
            rgba=[0.70, 0.70, 0.72, 1.0],
        )

    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    # Large enough for the calibrated 960x540 fixed cameras.  This only sizes
    # the offscreen render target and has no effect on MuJoCo physics.
    spec.visual.global_.offwidth = 960
    spec.visual.global_.offheight = 540
    wb = spec.worldbody
    wb.add_light(
        pos=[0.4, 0.0, 1.6],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
    )
    wb.add_light(pos=[0.4, -0.8, 1.2], dir=[0, 0.6, -1], diffuse=[0.4, 0.4, 0.4])
    wb.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.1],
        pos=[0, 0, -0.5],
        rgba=[0.30, 0.30, 0.32, 1.0],
    )
    table = wb.add_body(name="table", pos=[0.4, 0.0, -0.04])
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.45, 0.45, 0.02],
        rgba=[0.45, 0.45, 0.50, 1.0],
        condim=3,
        friction=[1.0, 0.05, 0.01],
    )

    insertive = wb.add_body(name="insertive_cube", pos=[0.45, 0.0, 0.02])
    insertive.add_freejoint(name="insertive_cube_joint")
    insertive.add_geom(
        name="insertive_cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
        mass=0.03,
        rgba=[0.85, 0.12, 0.08, 1.0],
        friction=[1.0, 0.02, 0.01],
        condim=6,
        solref=[0.02, 1],
        solimp=[0.9, 0.95, 0.001, 0.5, 2],
        priority=1,
    )

    receptive_pos = receptive_pose[:3]
    receptive_quat = receptive_pose[3:7]
    receptive = wb.add_body(name="receptive_cube", pos=list(receptive_pos), quat=list(receptive_quat))
    receptive.add_geom(
        name="receptive_cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
        rgba=[0.16, 0.30, 0.55, 1.0],
        friction=[1.0, 0.05, 0.01],
        condim=4,
        solref=[0.01, 1],
        solimp=[0.95, 0.99, 0.001, 0.5, 2],
    )

    wb.add_camera(name="cam_front", pos=[1.05, -0.55, 0.62], xyaxes=[0.5, 0.86, 0.0, -0.28, 0.16, 0.95])
    wb.add_camera(name="cam_side", pos=[0.4, -0.9, 0.45], xyaxes=[1, 0, 0, 0, 0.35, 0.94])

    # Vision-DP cameras.  Keep the legacy debug cameras above unchanged: these
    # three use the same OpenGL camera poses and calibrated vertical fields of
    # view as IsaacSim's ``vision_dp_20260715`` profile.  The wrist camera is
    # rigidly attached to panda ``hand``, exactly like the Isaac camera parent.
    dp_front = wb.add_camera(
        name="dp_front",
        pos=[1.252999568, 0.046552280, 1.022142743],
        quat=[0.658530788, 0.240652938, 0.263012492, 0.662757718],
    )
    dp_front.fovy = 2.0 * np.degrees(np.arctan(540.0 / (2.0 * 670.89000)))
    dp_front.resolution = [960, 540]
    dp_side = wb.add_camera(
        name="dp_side",
        pos=[0.742531964, -0.607496677, 0.375367773],
        quat=[0.767350171, 0.636200032, 0.070531898, 0.038058987],
    )
    dp_side.fovy = 2.0 * np.degrees(np.arctan(540.0 / (2.0 * 678.28275)))
    dp_side.resolution = [960, 540]
    dp_wrist = spec.body("hand").add_camera(
        name="dp_wrist",
        pos=[-0.197695705, 0.013046155, 0.022151960],
        quat=[0.274094481, 0.657434782, -0.629528490, -0.310395882],
    )
    dp_wrist.fovy = 2.0 * np.degrees(np.arctan(480.0 / (2.0 * 394.07100)))
    dp_wrist.resolution = [640, 480]
    model = spec.compile()
    model.dof_damping[:7] = 0.0
    return model


class Controller:
    def __init__(self, model):
        self.model = model
        self.hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self.obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "insertive_cube")
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "insertive_cube_joint")
        self.obj_qadr = model.jnt_qposadr[self.obj_jid]
        self.obj_dadr = model.jnt_dofadr[self.obj_jid]
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def ee_root(self, data):
        rp, rq = data.xpos[self.root], data.xquat[self.root]
        return Q.subtract_frame_transforms(rp, rq, data.xpos[self.hand].copy(), data.xquat[self.hand].copy())

    def jac_arm(self, data):
        mujoco.mj_jac(self.model, data, self.jacp, self.jacr, data.xpos[self.hand], self.hand)
        return np.vstack([self.jacp[:, :7], self.jacr[:, :7]])

    def grasp_close(self, data):
        hand_pos, hand_quat = data.xpos[self.hand], data.xquat[self.hand]
        tcp = hand_pos + Q.quat_apply(hand_quat, TCP_OFF)
        local = Q.quat_apply(Q.quat_inv(hand_quat), data.xpos[self.obj] - tcp)
        return (np.hypot(local[0], local[1]) < LAT_THRESH) and (abs(local[2]) < VERT_THRESH)


def set_state(model, data, ctrl, q, qvel, obj_state):
    data.qpos[:9] = q
    data.qvel[:9] = qvel
    data.qpos[ctrl.obj_qadr:ctrl.obj_qadr + 3] = obj_state[:3]
    data.qpos[ctrl.obj_qadr + 3:ctrl.obj_qadr + 7] = obj_state[3:7]
    if len(obj_state) >= 13:
        data.qvel[ctrl.obj_dadr:ctrl.obj_dadr + 6] = obj_state[7:13]
    mujoco.mj_forward(model, data)


def step_action(model, data, ctrl, action):
    scaled = action[:6] * SCALE
    ee_pos, ee_quat = ctrl.ee_root(data)
    desired_pos = ee_pos + scaled[:3]
    desired_quat = Q.quat_mul(quat_from_aa(scaled[3:6]), ee_quat)
    data.ctrl[ctrl.grip_act] = 0.0 if ctrl.grasp_close(data) else 255.0

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


def render_pair(model, ref, replay, renderer, cam_id, label):
    renderer.update_scene(ref, camera=cam_id)
    left = renderer.render()
    renderer.update_scene(replay, camera=cam_id)
    right = renderer.render()
    img = np.concatenate([left, np.zeros((left.shape[0], 6, 3), dtype=left.dtype), right], axis=1)
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, pil.width, 24], fill=(0, 0, 0))
        draw.text((8, 5), label, fill=(255, 255, 255))
        img = np.asarray(pil)
    except Exception:
        pass
    return img


def moving_average(x, h):
    x = np.asarray(x, dtype=np.float64)
    if h <= 1:
        return x
    kernel = np.ones(h, dtype=np.float64) / h
    return np.convolve(x, kernel, mode="same")


def plot_outputs(out_dir, stem, obj_err, rot_err, joint_err, chunk_h):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(obj_err))
    obj_mm = obj_err * 1000.0
    rot_deg = np.degrees(rot_err)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True)
    axes[0].plot(t, obj_mm, color="#2f855a", alpha=0.45, label="raw")
    axes[0].plot(t, moving_average(obj_mm, chunk_h), color="#14532d", lw=2.2, label=f"H={chunk_h}")
    axes[0].set_ylabel("Object pos (mm)")
    axes[0].legend(loc="upper right")
    axes[1].plot(t, rot_deg, color="#805ad5", alpha=0.45, label="raw")
    axes[1].plot(t, moving_average(rot_deg, chunk_h), color="#44337a", lw=2.2, label=f"H={chunk_h}")
    axes[1].set_ylabel("Object rot (deg)")
    axes[1].legend(loc="upper right")
    axes[2].plot(t, joint_err, color="#2b6cb0", alpha=0.45, label="raw")
    axes[2].plot(t, moving_average(joint_err, chunk_h), color="#1a365d", lw=2.2, label=f"H={chunk_h}")
    axes[2].set_ylabel("Joint L2 (rad)")
    axes[2].set_xlabel("Frame")
    axes[2].legend(loc="upper right")
    fig.suptitle("StackCube IsaacSim-to-MuJoCo per-step reset error")
    fig.tight_layout()
    curve = out_dir / f"{stem}_error_curve.png"
    fig.savefig(curve, dpi=180)
    plt.close(fig)

    rows = [
        obj_mm / max(np.percentile(obj_mm, 95), 1e-6),
        rot_deg / max(np.percentile(rot_deg, 95), 1e-6),
        joint_err / max(np.percentile(joint_err, 95), 1e-6),
    ]
    heat = np.clip(np.vstack(rows), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    im = ax.imshow(heat, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_yticks([0, 1, 2], ["obj pos", "obj rot", "joint"])
    ax.set_xlabel("Frame")
    ax.set_title("Row-normalized error heatmap")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    fig.tight_layout()
    heat_path = out_dir / f"{stem}_heatmap.png"
    fig.savefig(heat_path, dpi=180)
    plt.close(fig)

    n_chunks = int(np.ceil(len(obj_err) / chunk_h))
    chunks = np.zeros((3, n_chunks), dtype=np.float64)
    for ci in range(n_chunks):
        s = ci * chunk_h
        e = min(len(obj_err), s + chunk_h)
        chunks[0, ci] = obj_mm[s:e].mean()
        chunks[1, ci] = rot_deg[s:e].mean()
        chunks[2, ci] = joint_err[s:e].mean()
    chunks_norm = chunks / np.maximum(np.percentile(chunks, 95, axis=1, keepdims=True), 1e-6)
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    im = ax.imshow(np.clip(chunks_norm, 0.0, 1.0), aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks([0, 1, 2], ["obj pos", "obj rot", "joint"])
    ax.set_xlabel(f"Chunk index (H={chunk_h})")
    ax.set_title("Chunk-mean row-normalized heatmap")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    fig.tight_layout()
    chunk_path = out_dir / f"{stem}_chunk_heatmap_h{chunk_h}.png"
    fig.savefig(chunk_path, dpi=180)
    plt.close(fig)
    return curve, heat_path, chunk_path


def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = np.load(args.rollout, allow_pickle=True)
    q = ref["robot_joint_pos"]
    qvel = ref["robot_joint_vel"]
    obj = ref["insertive_root_state_w"]
    rec_pose = np.concatenate([ref["receptive_root_pos_w"][0], ref["receptive_root_quat_w"][0]])
    actions = ref["action"]
    limit = min(args.limit_steps if args.limit_steps > 0 else len(actions), len(actions))

    model = build_model(rec_pose)
    ctrl = Controller(model)
    data = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)

    obj_pred = []
    q_pred = []
    obj_err = []
    rot_err = []
    joint_err = []

    renderer = mujoco.Renderer(model, args.height, args.width)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    video_path = out_dir / "stackcube_task0_isaac_ref_left_mujoco_onestep_right.mp4"
    writer = imageio.get_writer(video_path, fps=args.fps, macro_block_size=1)

    for t in range(limit):
        set_state(model, data, ctrl, q[t], qvel[t], obj[t])
        step_action(model, data, ctrl, actions[t])

        pred_pose = np.concatenate(
            [data.qpos[ctrl.obj_qadr:ctrl.obj_qadr + 3], data.qpos[ctrl.obj_qadr + 3:ctrl.obj_qadr + 7]]
        )
        ref_pose_next = obj[t + 1, :7]
        obj_pred.append(pred_pose)
        q_pred.append(data.qpos[:9].copy())
        obj_err.append(float(np.linalg.norm(pred_pose[:3] - ref_pose_next[:3])))
        rot_err.append(quat_angle(pred_pose[3:7], ref_pose_next[3:7]))
        joint_err.append(float(np.linalg.norm(data.qpos[:9] - q[t + 1])))

        set_state(model, data_ref, ctrl, q[t + 1], qvel[t + 1], obj[t + 1])
        label = (
            f"LEFT IsaacSim ref | RIGHT MuJoCo one-step reset | frame={t+1}/{limit} "
            f"obj={obj_err[-1]*1000:.1f}mm rot={np.degrees(rot_err[-1]):.1f}deg joint={joint_err[-1]:.4f}rad"
        )
        writer.append_data(render_pair(model, data_ref, data, renderer, cam_id, label))
        if (t + 1) % 10 == 0:
            print(
                f"[stackcube-mujoco] {t+1:03d}/{limit}: "
                f"obj={obj_err[-1]*1000:.2f}mm rot={np.degrees(rot_err[-1]):.2f}deg joint={joint_err[-1]:.4f}"
            )

    writer.close()

    obj_err = np.asarray(obj_err, dtype=np.float32)
    rot_err = np.asarray(rot_err, dtype=np.float32)
    joint_err = np.asarray(joint_err, dtype=np.float32)
    stem = "stackcube_task0_mujoco_onestep"
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        source_rollout=np.array(str(args.rollout)),
        q_ref=q[: limit + 1],
        qvel_ref=qvel[: limit + 1],
        insertive_ref=obj[: limit + 1],
        receptive_pose=rec_pose.astype(np.float32),
        action=actions[:limit],
        q_pred=np.asarray(q_pred, dtype=np.float32),
        insertive_pred=np.asarray(obj_pred, dtype=np.float32),
        object_pos_err=obj_err,
        object_rot_err=rot_err,
        joint_err=joint_err,
        video_path=np.array(str(video_path)),
    )
    curve, heat, chunk = plot_outputs(out_dir, stem, obj_err, rot_err, joint_err, args.chunk_h)
    summary = {
        "definition": "reset MuJoCo to IsaacSim state(t), execute action(t), compare to IsaacSim state(t+1)",
        "source_rollout": str(args.rollout),
        "limit_steps": int(limit),
        "object_pos_max_mm": float(obj_err.max() * 1000.0),
        "object_pos_argmax": int(obj_err.argmax()),
        "object_rot_max_deg": float(np.degrees(rot_err.max())),
        "object_rot_argmax": int(rot_err.argmax()),
        "joint_max_rad": float(joint_err.max()),
        "joint_argmax": int(joint_err.argmax()),
        "outputs": {
            "npz": str(npz_path),
            "video": str(video_path),
            "curve": str(curve),
            "heatmap": str(heat),
            "chunk_heatmap": str(chunk),
        },
    }
    summary_path = out_dir / f"{stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[stackcube-mujoco] npz -> {npz_path}")
    print(f"[stackcube-mujoco] video -> {video_path}")
    print(f"[stackcube-mujoco] curve -> {curve}")
    print(f"[stackcube-mujoco] heatmap -> {heat}")
    print(f"[stackcube-mujoco] chunk heatmap -> {chunk}")
    print(f"[stackcube-mujoco] summary -> {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default="log/current_stackcube_sim2sim/isaac_stackcube_task0_model31000_len120.npz")
    ap.add_argument("--out", default="log/current_stackcube_sim2sim/mujoco_onestep")
    ap.add_argument("--limit_steps", type=int, default=120)
    ap.add_argument("--chunk_h", type=int, default=5)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--camera", default="cam_front")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
