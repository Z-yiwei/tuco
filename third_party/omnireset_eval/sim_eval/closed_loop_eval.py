"""Stage 4/5: CLOSED-LOOP policy eval of the Franka OmniReset RL expert in MuJoCo.

No replay — the policy drives the sim. Per 10 Hz policy step:
  1. reconstruct the 200-D obs from live MuJoCo state  (validated, gate3)
  2. action = FrankaPolicy(obs)                          (validated, gate1)
  3. process action -> desired EE pose (RelCartesianOSC), then 12 physics steps
     of OSC torque  tau = J^T (Kp*pose_err - Kd*ee_vel)  via qfrc_applied
  4. grasp-guard rule sets the fingers (policy gripper action ignored)
Success = peg seated in hole (peg_in_hole lateral/vert/rot within calibrated tol).

Run (rv_dp_mujoco):
  MUJOCO_GL=egl python scripts/sim2sim/franka/closed_loop_eval.py \
      --zarr datasets/franka_valset.zarr --episodes 20 --render 0
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import torch
import zarr
import imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quat_utils as Q
import obs_reconstruct as R
from obs_reconstruct import BLOCKS, HIST
from franka_policy import FrankaPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)  # omnireset_sim2sim
ROOT = PROJ  # kept for git_state() below
EVAL_SNAPSHOT = os.path.join(PROJ, "third_party", "co_curation_eval")
ASSETS = os.path.join(PROJ, "assets")
PANDA = os.path.join(ASSETS, "mujoco_menagerie", "franka_emika_panda", "panda.xml")
HOLE_VIS = os.path.join(ASSETS, "assets_mjcf", "hole_visual.stl")
HOLE_COL = os.path.join(ASSETS, "assets_mjcf", "hole_collision.stl")
HOLE_VIS_BIG = os.path.join(ASSETS, "assets_mjcf", "hole_visual_big.stl")
HOLE_COL_BIG = os.path.join(ASSETS, "assets_mjcf", "hole_collision_big.stl")
PEG_COL = os.path.join(ASSETS, "assets_mjcf", "peg_collision.stl")
# REAL franka_mimic fingertip collision meshes, extracted from the task's own USD
# (lab_assets .../Forge/franka_mimic.usd) via the isaacsim USD libs and exported to STL.
# These are the breakthrough: the Menagerie panda fingertip is ~10mm too short AND a
# different shape -> a hacked flat pad "clocks" the flat peg ~27deg in the grip, so the
# policy's reorientation can't verticalize it (SR ~1/15). The real finger mesh grips the
# peg at the correct orientation -> closed-loop SR jumps to 12/15 (80%).
MIMIC_LF = os.path.join(ASSETS, "assets_mjcf", "franka_mimic_leftfinger_col.stl")
MIMIC_RF = os.path.join(ASSETS, "assets_mjcf", "franka_mimic_rightfinger_col.stl")
CKPT = "logs/rsl_rl/franka_fr3_gripper_omnireset_agent/2026-06-01_13-12-28/model_4550.pt"
B3_SYSID_METADATA = os.path.join(EVAL_SNAPSHOT, "data", "sysid", "metadata_franka_b3.yaml")

# --- control constants (exact, from actions.py FRANKA_FR3_RELATIVE_OSC) ---
SCALE = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.2])
KP = np.array([200., 200., 200., 3., 3., 3.])
ZETA = np.array([3., 3., 3., 1., 1., 1.])
KD = 2.0 * np.sqrt(KP) * ZETA
STAGE1_SCALE = SCALE.copy()
STAGE1_KP = KP.copy()
STAGE1_ZETA = ZETA.copy()
STAGE2_SCALE = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2])
STAGE2_KP = np.array([1000., 1000., 1000., 50., 50., 50.])
STAGE2_ZETA = np.ones(6)
TAU_MAX = np.array([87., 87., 87., 87., 12., 12., 12.])
VEL_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # FR3 velocity_limit_sim
MIMIC_FINGER = True  # use the REAL franka_mimic fingertip collision mesh (default). THE fix:
#                      grips the flat peg at the correct orientation -> closed-loop SR 80%.
BIG_PAD = False  # legacy hacked flat pad (clocks the peg ~27deg -> SR ~1/15). Kept for ablation.
BOX_HOLE = True  # primitive box-ring hole collision (default; cleaner than SDF, no wedge/metastable artifacts)
PHYSX_CONTACT = True  # mimic PhysX TGS contact: 5mm margin (== contact_offset) + high solver iters
CMARGIN = 0.005       # PhysX contact_offset; gradual pre-contact engage (soft-guides peg into hole)
DECIM = 12
SIM_DT = 1.0 / 120.0
EP_LEN = 160
# pose-block -> term, resolved by gate3
MAPPING = {"poseA": "peg_in_hole", "poseB": "ee_pose", "poseC": "peg_in_hand", "poseD": "hole_in_hand"}
# grasp guard (actions.py GRASP_GUARDED_GRIPPER)
TCP_OFF = np.array([0.0, 0.0, 0.1034])
LAT_THRESH, VERT_THRESH = 0.02, 0.03
# success = the REAL task threshold (PegHole/metadata.yaml success_thresholds):
# peg within 2.5mm (3D) of the assembled pose + within 0.025rad (1.4deg) of aligned.
ASSEMBLED_Z = 0.014837   # PegHole metadata assembled_offset.pos z (seated peg z in hole frame)
SUC_POS, SUC_ROT = 0.0025, 0.025
STABLE_RELEASE_STEPS = 5  # 5 policy steps at 10 Hz = 0.5 s
PROFILE_CHOICES = (
    "legacy_stage1",
    "stage1_active_big",
    "stage2_b3_center",
    "stage2_b3_adr_sampled",
    "stage2_b3_box_hole_ablation",
)
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

LEGACY_GRIPPER = {
    "profile": "legacy_tuned",
    "tendon_stiffness": 400.0,
    "tendon_damping": 20.0,
    "tendon_force_limit": 60.0,
    "control_gain": 0.0627,
    "joint_stiffness_equivalent": 200.0,
    "joint_damping_equivalent": 10.0,
    "joint_effort_limit_equivalent": 30.0,
    "joint_friction": None,
    "joint_armature": None,
    "finger_contact_solref": None,
    "source": "legacy model_4550 MuJoCo tuning; retained byte-for-byte for baseline parity",
}

# The USD uses one driven finger joint plus a PhysX mimic joint. MuJoCo's
# Menagerie model represents the pair with a 0.5/0.5 fixed tendon and equality,
# so tendon K/D/force are doubled to preserve each finger's generalized force.
B3_GRIPPER = {
    "profile": "b3_training_center",
    "tendon_stiffness": 2000.0,
    "tendon_damping": 28.0,
    "tendon_force_limit": 120.0,
    "control_gain": None,
    "joint_stiffness_equivalent": 1000.0,
    "joint_damping_equivalent": 14.0,
    "joint_effort_limit_equivalent": 60.0,
    "joint_friction": 0.5,
    "joint_armature": 0.0,
    "finger_contact_solref": [0.001, 1.0],
    "finger_contact_solimp": [0.95, 0.99, 0.001, 0.5, 2.0],
    "finger_contact_friction": [2.0, 2.0, 0.3, 0.5, 0.5],
    "solver_calibration": (
        "16 MuJoCo substeps and 1ms contact solref reproduce Isaac B3's "
        "~15mm closed finger equilibrium for the 30mm peg"
    ),
    "source": "B3 params/env.yaml panda_hand: stiffness=1000 damping=14 effort_limit_sim=60 friction=0.5 armature=0",
}

USD_FINGER_INERTIA = {
    "mass": 0.02969999983906746,
    "com": np.array([0.0, 0.01140000019222498, 0.023099999874830246]),
    "diagonal": np.array([7.552000170107931e-06, 7.375549557764316e-06, 2.135450131390826e-06]),
    "principal_axes": np.array([0.994766116142273, 0.1021781712770462, 0.0, 0.0]),
}


def set_controller_gains(scale, kp, zeta):
    global SCALE, KP, ZETA, KD
    SCALE = np.asarray(scale, dtype=np.float64)
    KP = np.asarray(kp, dtype=np.float64)
    ZETA = np.asarray(zeta, dtype=np.float64)
    KD = 2.0 * np.sqrt(KP) * ZETA


def file_md5(path, block_size=1 << 20):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_state(path):
    def _run(args):
        try:
            p = subprocess.run(args, cwd=path, text=True, capture_output=True, check=False)
        except Exception as exc:
            return f"ERROR: {exc}"
        if p.returncode != 0:
            return p.stderr.strip() or p.stdout.strip()
        return p.stdout.strip()

    return {
        "root": _run(["git", "rev-parse", "--show-toplevel"]),
        "head": _run(["git", "rev-parse", "HEAD"]),
        "status_short": _run(["git", "status", "--short"]),
    }


def _simple_sysid_yaml(path):
    sysid = {}
    current = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("_note"):
                continue
            if stripped == "sysid:":
                continue
            if stripped.endswith(":") and not stripped.startswith("-"):
                current = stripped[:-1]
                sysid[current] = []
                continue
            if stripped.startswith("-") and current:
                sysid[current].append(float(stripped[1:].strip()))
                continue
            if ":" in stripped:
                key, val = stripped.split(":", 1)
                val = val.split("#", 1)[0].strip()
                if val:
                    try:
                        sysid[key.strip()] = int(val)
                    except ValueError:
                        sysid[key.strip()] = float(val)
    return {"sysid": sysid}


def load_sysid_metadata(path):
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception:
        data = _simple_sysid_yaml(path)
    sysid = data["sysid"]
    required = ["armature", "static_friction", "dynamic_ratio", "viscous_friction", "motor_delay"]
    missing = [k for k in required if k not in sysid]
    if missing:
        raise ValueError(f"sysid metadata missing keys: {missing}")
    for k in required[:-1]:
        if len(sysid[k]) != 7:
            raise ValueError(f"sysid metadata key {k} must have 7 values, got {len(sysid[k])}")
    return sysid


def sample_sysid(sysid, rng, sampled):
    out = {}
    scales = {}
    for key in ("armature", "static_friction", "dynamic_ratio", "viscous_friction"):
        nominal = np.asarray(sysid[key], dtype=np.float64)
        scale = rng.uniform(0.8, 1.2, size=7) if sampled else np.ones(7)
        out[key] = nominal * scale
        scales[key] = scale
    out["dynamic_friction"] = np.minimum(out["dynamic_ratio"] * out["static_friction"], out["static_friction"])
    out["motor_delay"] = int(sysid["motor_delay"])
    out["scale_factors"] = scales
    return out


def sample_stage2_gains(rng, sampled):
    kp = STAGE2_KP.copy()
    zeta = STAGE2_ZETA.copy()
    factors = {"kp_xyz": 1.0, "kp_rpy": 1.0, "zeta_xyz": 1.0, "zeta_rpy": 1.0}
    if sampled:
        factors = {
            "kp_xyz": float(rng.uniform(0.8, 1.2)),
            "kp_rpy": float(rng.uniform(0.8, 1.2)),
            "zeta_xyz": float(rng.uniform(0.8, 1.2)),
            "zeta_rpy": float(rng.uniform(0.8, 1.2)),
        }
        kp[:3] *= factors["kp_xyz"]
        kp[3:] *= factors["kp_rpy"]
        zeta[:3] *= factors["zeta_xyz"]
        zeta[3:] *= factors["zeta_rpy"]
    return STAGE2_SCALE.copy(), kp, zeta, factors


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def runtime_versions():
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "zarr": zarr.__version__,
        "imageio": imageio.__version__,
    }


def gripper_config(name):
    if name == "legacy":
        return dict(LEGACY_GRIPPER)
    if name == "b3":
        return dict(B3_GRIPPER)
    raise ValueError(f"unknown gripper profile: {name}")


def zarr_signature(z):
    raw = np.asarray(z["data/raw_state"])
    ends = np.asarray(z["meta/episode_ends"])
    sig = {
        "raw_state_shape": list(raw.shape),
        "episode_ends_shape": list(ends.shape),
        "episode_ends_md5": hashlib.md5(ends.tobytes()).hexdigest(),
    }
    if raw.size:
        probe = np.concatenate([raw[0].ravel(), raw[min(len(raw) - 1, 31)].ravel(), raw[-1].ravel()])
        sig["raw_probe_md5"] = hashlib.md5(probe.astype(np.float64).tobytes()).hexdigest()
    return sig


def make_base_profile(args):
    cfg = {
        "profile": args.profile,
        "controller": "stage1",
        "sysid_metadata_path": None,
        "apply_sysid": False,
        "sample_sysid": False,
        "sample_osc_gains": False,
        "peg_mass": 0.02,
        "peg_collision": "box",
        "peg_contact_solref": [0.02, 1.0],
        "hole_collision": "box_ring",
        "gripper": gripper_config("legacy"),
        "sample_gripper": False,
        "motor_delay_mode": "not_applied",
        "motor_delay_note": "legacy profile has no B3 delay",
        "physics_substeps": 1,
        "finger_velocity_limit": None,
        "real_finger_inertia": False,
        "robot_root_frame_init": False,
        "unexplained_mismatches": [
            "PhysX TGS contact is approximated by MuJoCo implicitfast/elliptic contact.",
            "Legacy gripper values are empirical model_4550 tuning, not an Isaac actuator match.",
        ],
    }

    if args.profile.startswith("stage2_b3"):
        sysid_path = os.path.abspath(args.sysid_metadata)
        cfg.update({
            "controller": "stage2_deploy",
            "sysid_metadata_path": sysid_path,
            "sysid_nominal": load_sysid_metadata(sysid_path),
            "apply_sysid": True,
            "sample_sysid": args.profile == "stage2_b3_adr_sampled",
            "sample_osc_gains": args.profile == "stage2_b3_adr_sampled",
            "peg_mass": 0.001,
            "peg_collision": "mesh",
            "peg_contact_solref": [0.001, 1.0],
            "hole_collision": "mesh_sdf",
            "gripper": gripper_config("b3"),
            "sample_gripper": args.profile == "stage2_b3_adr_sampled",
            "motor_delay_mode": "not_applied",
            "motor_delay_note": "metadata has motor_delay=8, but current IsaacSim Stage-2 uses ImplicitActuatorCfg so delay buffer is not active.",
            "physics_substeps": 16,
            "finger_velocity_limit": 0.025,
            "real_finger_inertia": True,
            "robot_root_frame_init": True,
            "unexplained_mismatches": [
                "MuJoCo has no direct dynamic joint-friction coefficient matching PhysX; static frictionloss and viscous damping are applied, dynamic_ratio is logged.",
                "PhysX TGS contact/material behavior is approximated by MuJoCo implicitfast/elliptic contact.",
                "MuJoCo uses 16 numerical substeps per Isaac physics step to reproduce the rigid-contact finger equilibrium.",
                "Isaac independent finger PD is converted to the equivalent Menagerie tendon PD using the 0.5 tendon coefficients.",
                "USD collision meshes are represented by exported STL meshes/SDF; exact USD collision cooking is not reproduced.",
                "The effective 0.025m/s mimic-finger limit is measured from the Isaac validation trajectory; the USD authors 0.05m/s on the master and 0.04m/s on the mimic joint.",
            ],
        })
    if args.profile == "stage1_active_big":
        cfg.update({
            "controller": "stage1",
            "peg_mass": 0.02,
            "peg_collision": "box",
            "peg_contact_solref": [0.001, 1.0],
            "hole_collision": "box_ring_big",
            "gripper": gripper_config("b3"),
            "sample_gripper": False,
            "physics_substeps": 16,
            "finger_velocity_limit": 0.025,
            "real_finger_inertia": True,
            "unexplained_mismatches": [
                "PhysX TGS contact is approximated by MuJoCo implicitfast/elliptic contact.",
                "MuJoCo uses 16 numerical substeps per Isaac physics step for the stiff 1000/14 finger actuator.",
            ],
        })
    if args.profile == "stage2_b3_box_hole_ablation":
        cfg["hole_collision"] = "box_ring"
        cfg["unexplained_mismatches"].append("Box-ring hole is an explicit ablation and is not the primary B3-aligned geometry.")
    if args.eval_gains and cfg["controller"] == "stage1":
        cfg["controller"] = "stage2_deploy_old_harness"
    if args.controller_profile != "profile":
        cfg["controller"] = args.controller_profile
    if args.sysid_mode != "profile":
        cfg["apply_sysid"] = args.sysid_mode != "off"
        cfg["sample_sysid"] = args.sysid_mode == "sampled"
        if cfg["apply_sysid"]:
            sysid_path = os.path.abspath(args.sysid_metadata)
            cfg["sysid_metadata_path"] = sysid_path
            cfg["sysid_nominal"] = load_sysid_metadata(sysid_path)
    if args.gripper_profile != "profile":
        cfg["gripper"] = gripper_config(args.gripper_profile)
        cfg["sample_gripper"] = False
    for arg_name in (
        "tendon_stiffness",
        "tendon_damping",
        "tendon_force_limit",
        "joint_friction",
        "joint_armature",
    ):
        value = getattr(args, f"gripper_{arg_name}")
        if value is not None:
            cfg["gripper"][arg_name] = value
    if args.gripper_tendon_stiffness is not None:
        cfg["gripper"]["control_gain"] = None
    if args.gripper_contact_timeconst is not None:
        solref = cfg["gripper"].get("finger_contact_solref") or [0.02, 1.0]
        cfg["gripper"]["finger_contact_solref"] = [args.gripper_contact_timeconst, solref[1]]
    if args.gripper_contact_dampratio is not None:
        solref = cfg["gripper"].get("finger_contact_solref") or [0.02, 1.0]
        cfg["gripper"]["finger_contact_solref"] = [solref[0], args.gripper_contact_dampratio]
    if args.gripper_contact_stiffness is not None:
        damping = args.gripper_contact_damping
        if damping is None:
            damping = 2.0 * np.sqrt(args.gripper_contact_stiffness)
        cfg["gripper"]["finger_contact_solref"] = [
            -args.gripper_contact_stiffness,
            -damping,
        ]
    if args.gripper_contact_impedance is not None:
        solimp = cfg["gripper"].get("finger_contact_solimp") or [0.9, 0.95, 0.001, 0.5, 2.0]
        cfg["gripper"]["finger_contact_solimp"] = [
            args.gripper_contact_impedance,
            args.gripper_contact_impedance,
            *solimp[2:],
        ]
    if args.peg_mass is not None:
        cfg["peg_mass"] = args.peg_mass
    if args.peg_collision != "profile":
        cfg["peg_collision"] = args.peg_collision
    if args.hole_collision != "profile":
        cfg["hole_collision"] = args.hole_collision
    if args.box_hole:
        cfg["hole_collision"] = "box_ring"
    if args.physics_substeps is not None:
        cfg["physics_substeps"] = args.physics_substeps
    return cfg


def make_episode_config(base_cfg, episode_idx, seed):
    rng = np.random.default_rng(seed + episode_idx)
    cfg = dict(base_cfg)
    if base_cfg["controller"] == "stage1":
        scale, kp, zeta = STAGE1_SCALE.copy(), STAGE1_KP.copy(), STAGE1_ZETA.copy()
        gain_factors = {"kp_xyz": 1.0, "kp_rpy": 1.0, "zeta_xyz": 1.0, "zeta_rpy": 1.0}
    else:
        scale, kp, zeta, gain_factors = sample_stage2_gains(rng, base_cfg["sample_osc_gains"])
    cfg["controller_values"] = {
        "scale": scale,
        "kp": kp,
        "zeta": zeta,
        "kd": 2.0 * np.sqrt(kp) * zeta,
        "gain_scale_factors": gain_factors,
    }
    gripper = dict(base_cfg["gripper"])
    if base_cfg["sample_gripper"]:
        stiffness_scale = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        damping_scale = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        gripper["tendon_stiffness"] *= stiffness_scale
        gripper["tendon_damping"] *= damping_scale
        gripper["joint_stiffness_equivalent"] *= stiffness_scale
        gripper["joint_damping_equivalent"] *= damping_scale
        gripper["scale_factors"] = {
            "stiffness": stiffness_scale,
            "damping": damping_scale,
        }
    cfg["gripper"] = gripper
    if base_cfg["apply_sysid"]:
        sysid_nominal = base_cfg["sysid_nominal"]
        cfg["sysid"] = sample_sysid(sysid_nominal, rng, base_cfg["sample_sysid"])
        cfg["sysid_joint_names"] = ARM_JOINT_NAMES
    else:
        cfg["sysid"] = None
    return cfg


def quat_from_aa(aa):
    angle = np.linalg.norm(aa)
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))])


def square_symmetry_rot_error(hole_quat, peg_quat):
    """Rotation error for a square peg, where 90-degree yaw turns are equivalent."""
    best = float("inf")
    for yaw in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        yaw_quat = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
        desired = Q.quat_mul(hole_quat, yaw_quat)
        error = Q.quat_mul(Q.quat_inv(desired), peg_quat)
        best = min(best, float(np.linalg.norm(Q.axis_angle_from_quat(error))))
    return best


def build_model(hole_pos, hole_quat, profile_cfg=None):
    profile_cfg = profile_cfg or {}
    gripper_cfg = profile_cfg.get("gripper", LEGACY_GRIPPER)
    hole_collision = profile_cfg.get("hole_collision", "box_ring" if BOX_HOLE else "mesh_sdf")
    big_hole = hole_collision in ("mesh_sdf_big", "box_ring_big")
    spec = mujoco.MjSpec.from_file(PANDA)
    spec.option.timestep = SIM_DT / int(profile_cfg.get("physics_substeps", 1))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # solver context for stable grasping (matches working UR5e omnireset scene)
    spec.option.impratio = 10.0
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 5
    if PHYSX_CONTACT:
        # mimic PhysX TGS rigid contact: high solver iterations (PhysX uses 192 pos-iter)
        # + 5mm contact margin == PhysX contact_offset=0.005 (gradual pre-contact engage,
        # "soft-guides" the peg into the hole instead of hard-catching the rim).
        spec.option.iterations = 150

    # IsaacLab spawns the Franka with disable_gravity=True (factory_assets_cfg.py:36),
    # and the OSC law has no gravity comp -> match it by gravity-compensating the arm
    # (peg keeps gravity). Without this the arm sags ~0.12m and never tracks.
    for b in spec.bodies:
        if b.name != "world":           # official bodies: link0-7, hand, left/right_finger
            b.gravcomp = 1.0

    # neutralize the arm position-servos (we drive the arm by qfrc_applied OSC torque).
    # KEEP actuator8 = official gripper (tendon 'split' + equality joint1=joint2) -> the
    # two fingers move STRICTLY SYMMETRICALLY (fixes the asymmetric-close peg-spin from
    # the old 2-independent-actuator hack).
    for a in spec.actuators:
        if a.name in ("actuator1", "actuator2", "actuator3", "actuator4",
                      "actuator5", "actuator6", "actuator7"):
            a.gainprm = np.zeros_like(np.array(a.gainprm))
            a.biasprm = np.zeros_like(np.array(a.biasprm))
        if a.name == "actuator8":
            tendon_kp = float(gripper_cfg["tendon_stiffness"])
            tendon_kd = float(gripper_cfg["tendon_damping"])
            tendon_force = float(gripper_cfg["tendon_force_limit"])
            # ctrl 0/255 maps to tendon target 0/0.04 m. The 0.5 coefficients
            # in Menagerie's split tendon are already accounted for in the B3
            # equivalent tendon K/D/force values above.
            ctrl_gain = gripper_cfg.get("control_gain")
            if ctrl_gain is None:
                ctrl_gain = tendon_kp * 0.04 / 255.0
            a.gainprm = np.array([ctrl_gain, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            a.biasprm = np.array([0, -tendon_kp, -tendon_kd, 0, 0, 0, 0, 0, 0, 0])
            a.forcerange = [-tendon_force, tendon_force]

    # isolate the grip to the high-friction pads: drop peg contact on the broad
    # finger collision MESH (it leaks a low-friction condim=3 contact in parallel)
    for bn in ("left_finger", "right_finger"):
        for g in spec.body(bn).geoms:
            if str(g.type).endswith("MESH") and g.contype != 0:
                g.contype, g.conaffinity = 0, 0

    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    wb = spec.worldbody
    wb.add_light(pos=[0.4, 0, 1.6], dir=[0, 0, -1],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    wb.add_light(pos=[0.4, -0.8, 1.2], dir=[0, 0.6, -1], diffuse=[0.4, 0.4, 0.4])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                pos=[0, 0, -0.5], rgba=[0.3, 0.3, 0.32, 1])
    table = wb.add_body(name="table", pos=[0.4, 0.0, -0.0335])
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.4, 0.4, 0.02],
                   rgba=[0.45, 0.45, 0.5, 1], condim=3, friction=[1.0, 0.05, 0.01])

    peg = wb.add_body(name="peg", pos=[0.45, 0.1, 0.05])
    peg.add_freejoint()
    # LOW peg-geom friction so peg<->hole/table SLIDES (no wedge/metastable tilt).
    # The firm GRASP comes from the explicit peg<->pad contact pairs below (which
    # override friction), so lowering this does NOT weaken the grip.
    pm = 0.0 if MIMIC_FINGER else (CMARGIN if PHYSX_CONTACT else 0.0)  # mesh grip needs no margin
    pf = 1.0 if MIMIC_FINGER else 0.6  # source peg friction is 1.0-2.0; mesh grip uses peg(priority) friction
    peg_mass = float(profile_cfg.get("peg_mass", 0.02))
    peg_collision = profile_cfg.get("peg_collision", "box")
    peg_contact_solref = profile_cfg.get("peg_contact_solref", [0.02, 1.0])
    if peg_collision == "mesh":
        spec.add_mesh(name="peg_col", file=PEG_COL)
        peg.add_geom(name="peg_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="peg_col",
                     mass=peg_mass, rgba=[0.85, 0.12, 0.08, 1], friction=[pf, 0.02, 0.01],
                     condim=6, solref=peg_contact_solref, solimp=[0.9, 0.95, 0.001, 0.5, 2], priority=1,
                     margin=pm, gap=pm)
    else:
        peg.add_geom(name="peg_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.015, 0.03],
                     mass=peg_mass, rgba=[0.85, 0.12, 0.08, 1], friction=[pf, 0.02, 0.01],
                     condim=6, solref=peg_contact_solref, solimp=[0.9, 0.95, 0.001, 0.5, 2], priority=1,
                     margin=pm, gap=pm)

    # stable grasp: explicit peg<->pad contact pairs (UR5e recipe), condim=6 +
    # torsional/rolling friction, critically-damped solref.
    if MIMIC_FINGER:
        # THE FIX: real franka_mimic fingertip collision mesh (extracted from the task's
        # own USD). Menagerie's fingertip is short + wrong-shaped -> the hacked flat pad
        # clocks the flat peg ~27deg in the grip, so the policy can't verticalize it.
        # The real finger mesh grips at the correct orientation -> closed-loop SR 80%.
        spec.add_mesh(name="mimic_lf", file=MIMIC_LF)
        spec.add_mesh(name="mimic_rf", file=MIMIC_RF)
        for bn, mn in (("left_finger", "mimic_lf"), ("right_finger", "mimic_rf")):
            body = spec.body(bn)
            for g in body.geoms:               # disable Menagerie small box pads
                if str(g.type).endswith("BOX"):
                    g.contype, g.conaffinity = 0, 0
            body.add_geom(name=f"{bn}_mimic", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn,
                          friction=[2.0, 0.1, 0.05], condim=6, rgba=[0.7, 0.7, 0.72, 1])
        if gripper_cfg.get("finger_contact_solref") is not None:
            for bn in ("left_finger", "right_finger"):
                spec.add_pair(
                    geomname1="peg_geom",
                    geomname2=f"{bn}_mimic",
                    condim=6,
                    friction=gripper_cfg["finger_contact_friction"],
                    solref=gripper_cfg["finger_contact_solref"],
                    solimp=gripper_cfg["finger_contact_solimp"],
                )
    elif BIG_PAD:
        # ABLATION: replace the 5 small fingertip pads with ONE large flat pad per
        # finger (covers the whole peg contact face) -> tests pad-geometry vs solver.
        for bn, pref in (("left_finger", "lbig"), ("right_finger", "rbig")):
            body = spec.body(bn)
            for g in body.geoms:
                if str(g.type).endswith("BOX"):
                    g.contype, g.conaffinity = 0, 0  # disable small pads
            body.add_geom(name=pref, type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[0.012, 0.004, 0.024], pos=[0.0, 0.0055, 0.040],
                          friction=[2.0, 0.1, 0.05], condim=6, rgba=[0.1, 0.8, 0.1, 1])
        pad_names = ["lbig", "rbig"]
        for nm in pad_names:
            spec.add_pair(geomname1="peg_geom", geomname2=nm, condim=6,
                          friction=[2, 2, 0.3, 0.5, 0.5], solref=[0.04, 1.0],
                          solimp=[0.85, 0.95, 0.004, 0.5, 2])
    else:
        for bn, pref in (("left_finger", "lpad"), ("right_finger", "rpad")):
            boxes = [g for g in spec.body(bn).geoms if str(g.type).endswith("BOX")]
            for i, g in enumerate(boxes):
                g.name = f"{pref}{i}"
                g.friction = [2.0, 0.1, 0.05]
                g.condim = 6
        for pref in ("lpad", "rpad"):
            for i in range(5):
                spec.add_pair(geomname1="peg_geom", geomname2=f"{pref}{i}", condim=6,
                              friction=[2, 2, 0.3, 0.5, 0.5], solref=[0.04, 1.0],
                              solimp=[0.85, 0.95, 0.004, 0.5, 2])

    spec.add_mesh(name="hole_vis", file=HOLE_VIS_BIG if big_hole else HOLE_VIS, scale=[1, 1, 1])
    hole = wb.add_body(name="peghole", pos=list(hole_pos), quat=list(hole_quat))
    hole.add_geom(name="hole_vis_g", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hole_vis",
                  contype=0, conaffinity=0, rgba=[0.16, 0.30, 0.55, 1])
    if hole_collision in ("box_ring", "box_ring_big"):
        # Primitive square-ring collision. The active big-hole contract uses a
        # 40 mm opening; the legacy profile remains at 33 mm.
        # Clean box contact normals -> no SDF wedge/metastable-tilt artifacts.
        zc, zh = 0.00465, 0.01825   # walls span floor(-0.0136) .. top(0.0229)
        wf = dict(condim=4, friction=[0.4, 0.02, 0.01], solref=[0.01, 1],
                  solimp=[0.95, 0.99, 0.001, 0.5, 2], rgba=[0.16, 0.30, 0.55, 1])
        inner = 0.020 if hole_collision == "box_ring_big" else 0.0165
        outer = 0.034
        wall_half = 0.5 * (outer - inner)
        wall_center = 0.5 * (outer + inner)
        hole.add_geom(name="hwall_px", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[wall_center, 0, zc], size=[wall_half, outer, zh], **wf)
        hole.add_geom(name="hwall_nx", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-wall_center, 0, zc], size=[wall_half, outer, zh], **wf)
        hole.add_geom(name="hwall_py", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, wall_center, zc], size=[inner, wall_half, zh], **wf)
        hole.add_geom(name="hwall_ny", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -wall_center, zc], size=[inner, wall_half, zh], **wf)
        if hole_collision == "box_ring_big":
            floor_top = ASSEMBLED_Z - 0.03
            floor_bottom = -0.01770333
            floor_center = 0.5 * (floor_top + floor_bottom)
            floor_half = 0.5 * (floor_top - floor_bottom)
        else:
            floor_center, floor_half = -0.0161, 0.0025
        hole.add_geom(name="hfloor", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[0, 0, floor_center], size=[inner, inner, floor_half], **wf)
    else:
        spec.add_mesh(name="hole_col", file=HOLE_COL_BIG if big_hole else HOLE_COL, scale=[1, 1, 1])
        hole.add_geom(name="hole_col_g", type=mujoco.mjtGeom.mjGEOM_SDF, meshname="hole_col",
                      friction=[0.4, 0.005, 0.001], solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])

    wb.add_camera(name="cam_front", pos=[1.05, -0.55, 0.62],
                  xyaxes=[0.5, 0.86, 0.0, -0.28, 0.16, 0.95])
    wb.add_camera(name="cam_side", pos=[0.4, -0.9, 0.45], xyaxes=[1, 0, 0, 0, 0.35, 0.94])
    m = spec.compile()
    sysid = profile_cfg.get("sysid")
    if sysid is not None:
        # B3 Stage-2 writes these via randomize_arm_from_sysid in IsaacSim.
        # MuJoCo has frictionloss + viscous damping, but no direct equivalent for
        # PhysX dynamic_ratio, so dynamic_friction is logged in effective_config.
        m.dof_armature[:7] = np.asarray(sysid["armature"], dtype=np.float64)
        m.dof_frictionloss[:7] = np.asarray(sysid["static_friction"], dtype=np.float64)
        m.dof_damping[:7] = np.asarray(sysid["viscous_friction"], dtype=np.float64)
    else:
        m.dof_damping[:7] = 0.0  # strict align: IsaacLab arm damping=0.0 (panda.xml class default is 1.0)
    if gripper_cfg.get("joint_armature") is not None:
        m.dof_armature[7:9] = float(gripper_cfg["joint_armature"])
    if gripper_cfg.get("joint_friction") is not None:
        m.dof_frictionloss[7:9] = float(gripper_cfg["joint_friction"])
    if gripper_cfg["profile"] == "b3_training_center":
        # Damping is represented by actuator8's tendon velocity bias. Remove
        # Menagerie's inherited passive joint damping to avoid double counting.
        m.dof_damping[7:9] = 0.0
    if profile_cfg.get("real_finger_inertia", False):
        for body_name in ("left_finger", "right_finger"):
            body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            m.body_mass[body_id] = USD_FINGER_INERTIA["mass"]
            m.body_ipos[body_id] = USD_FINGER_INERTIA["com"]
            m.body_inertia[body_id] = USD_FINGER_INERTIA["diagonal"]
            m.body_iquat[body_id] = USD_FINGER_INERTIA["principal_axes"]
        mujoco.mj_setConst(m, mujoco.MjData(m))
    return m


class Controller:
    def __init__(self, m):
        self.m = m
        self.hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.l0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self.peg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.hole = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peghole")
        self.peg_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
        self.robot_body_ids = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in [*(f"link{i}" for i in range(8)), "hand", "left_finger", "right_finger"]
        }
        # official single gripper actuator (tendon 'split' + equality -> symmetric fingers)
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.peg_dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "peg")]
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    def ee_root(self, d):
        rp, rq = d.xpos[self.l0], d.xquat[self.l0]
        return Q.subtract_frame_transforms(rp, rq, d.xpos[self.hand].copy(), d.xquat[self.hand].copy())

    def jac_arm(self, d):
        mujoco.mj_jac(self.m, d, self.jacp, self.jacr, d.xpos[self.hand], self.hand)
        return np.vstack([self.jacp[:, :7], self.jacr[:, :7]])  # 6x7 (root=world)

    def grasp_close(self, d):
        """True if peg root is within the TCP capture zone (-> close fingers)."""
        hp, hq = d.xpos[self.hand], d.xquat[self.hand]
        tcp = hp + Q.quat_apply(hq, TCP_OFF)
        local = Q.quat_apply(Q.quat_inv(hq), d.xpos[self.peg] - tcp)
        lateral = np.hypot(local[0], local[1])
        return (lateral < LAT_THRESH) and (abs(local[2]) < VERT_THRESH)

    def peg_robot_contact(self, d):
        return bool(self.peg_robot_contact_bodies(d))

    def peg_robot_contact_bodies(self, d):
        bodies = set()
        for contact in d.contact[:d.ncon]:
            if contact.geom1 == self.peg_geom:
                other = contact.geom2
            elif contact.geom2 == self.peg_geom:
                other = contact.geom1
            else:
                continue
            body_id = self.m.geom_bodyid[other]
            if body_id in self.robot_body_ids:
                bodies.add(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, body_id))
        return bodies


class ObsBuilder:
    """Assembles the 200-D policy obs from live MuJoCo state, term-major w/ 5-step history."""

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.fk = R.FrankaFK()
        self.hist = {b: collections.deque(maxlen=HIST) for b in BLOCKS}

    def _terms(self, d):
        raw = np.zeros(57)
        raw[0:9] = d.qpos[:9]
        raw[18:21] = d.xpos[self.ctrl.l0]
        raw[21:25] = d.xquat[self.ctrl.l0]
        raw[31:34] = d.xpos[self.ctrl.peg]
        raw[34:38] = d.xquat[self.ctrl.peg]
        raw[44:47] = d.xpos[self.ctrl.hole]
        raw[47:51] = d.xquat[self.ctrl.hole]
        pose, _ = R.pose_terms_from_raw(raw, self.fk)
        return pose, raw

    def reset(self, d, fill_repeat=True):
        pose, _ = self._terms(d)
        vals = {
            "poseA": pose[MAPPING["poseA"]], "poseB": pose[MAPPING["poseB"]],
            "poseC": pose[MAPPING["poseC"]], "poseD": pose[MAPPING["poseD"]],
            "joint_pos": d.qpos[:9].copy(), "prev_actions": np.zeros(7),
        }
        for b in BLOCKS:
            self.hist[b].clear()
            for _ in range(HIST):
                self.hist[b].append(vals[b] if fill_repeat else np.zeros_like(vals[b]))

    def step(self, d, prev_action):
        pose, _ = self._terms(d)
        push = {
            "poseA": pose[MAPPING["poseA"]], "poseB": pose[MAPPING["poseB"]],
            "poseC": pose[MAPPING["poseC"]], "poseD": pose[MAPPING["poseD"]],
            "joint_pos": d.qpos[:9].copy(), "prev_actions": prev_action,
        }
        for b in BLOCKS:
            self.hist[b].append(push[b])
        obs = np.zeros(200, dtype=np.float32)
        for b, (start, w) in BLOCKS.items():
            for hi in range(HIST):  # deque is oldest->newest, matches obs oldest-first
                obs[start + hi * w: start + (hi + 1) * w] = self.hist[b][hi]
        return obs


def peg_in_hole_pose(d, ctrl):
    return Q.pose_in_root(d.xpos[ctrl.hole], d.xquat[ctrl.hole], d.xpos[ctrl.peg], d.xquat[ctrl.peg])


def run_episode(
    m,
    ctrl,
    obb,
    policy,
    init,
    render=False,
    w=480,
    h=360,
    grasp_assist=False,
    finger_velocity_limit=None,
):
    d = mujoco.MjData(m)
    d.qpos[:9] = init["q9"]
    pj = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "peg")]
    d.qpos[pj:pj + 3] = init["peg_pos"]
    d.qpos[pj + 3:pj + 7] = init["peg_quat"]
    d.qvel[:9] = init.get("qvel9", 0.0)
    d.qvel[ctrl.peg_dof:ctrl.peg_dof + 6] = init.get("peg_vel", 0.0)
    mujoco.mj_forward(m, d)
    obb.reset(d)

    rends = {}
    frames = []
    if render:
        rends = {c: mujoco.Renderer(m, h, w) for c in ("cam_front", "cam_side")}

    pj = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "peg")]
    prev_action = np.zeros(7, dtype=np.float32)
    success = False
    best = 1e9
    best_rot = 1e9
    best_physical_rot = 1e9
    physical_success = False
    glued = False
    rel_pos = rel_quat = None
    closed_steps = 0
    ever_closed = False
    grasp_lost_after_close = False
    release_step = None
    stable_release_streak = 0
    max_stable_release_streak = 0
    stable_success = False
    arm_min_limit_margin = float("inf")
    for t in range(EP_LEN):
        obs = obb.step(d, prev_action)
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).float().unsqueeze(0)).numpy()[0]  # (7,)
        prev_action = action.astype(np.float32)
        a6 = action[:6]
        scaled = a6 * SCALE

        # process: desired EE pose in root frame = current + delta
        ee_pos, ee_quat = ctrl.ee_root(d)
        ee_pos_des = ee_pos + scaled[:3]
        ee_quat_des = Q.quat_mul(quat_from_aa(scaled[3:6]), ee_quat)

        close = ctrl.grasp_close(d)
        if close:
            ever_closed = True
            closed_steps += 1
        elif ever_closed:
            grasp_lost_after_close = True
            if release_step is None:
                release_step = t
        # official gripper actuator8 (tendon+equality, symmetric): ctrl 0=closed, 255=open(0.04).
        # close -> 0 (symmetric squeeze, no asymmetric spin); open -> 255.
        d.ctrl[ctrl.grip_act] = 0.0 if close else 255.0

        if grasp_assist:  # idealized grasp: weld peg to hand while in capture zone
            if close and not glued:
                hp, hq = d.xpos[ctrl.hand].copy(), d.xquat[ctrl.hand].copy()
                rel_pos = Q.quat_apply(Q.quat_inv(hq), d.qpos[pj:pj + 3] - hp)
                rel_quat = Q.quat_mul(Q.quat_inv(hq), d.qpos[pj + 3:pj + 7].copy())
                glued = True
            elif not close:
                glued = False

        physics_substeps = max(1, int(round(SIM_DT / m.opt.timestep)))
        for _ in range(DECIM * physics_substeps):
            finger_pos_before = d.qpos[7:9].copy() if finger_velocity_limit is not None else None
            ee_pos, ee_quat = ctrl.ee_root(d)
            J = ctrl.jac_arm(d)
            dq = d.qvel[:7]
            ee_vel = J @ dq
            pos_err = ee_pos_des - ee_pos
            quat_err = Q.quat_mul(ee_quat_des, Q.quat_inv(ee_quat))
            aa_err = Q.axis_angle_from_quat(quat_err)
            pose_err = np.concatenate([pos_err, aa_err])
            task_force = KP * pose_err + KD * (-ee_vel)
            tau = np.clip(J.T @ task_force, -TAU_MAX, TAU_MAX)
            d.qfrc_applied[:7] = tau
            mujoco.mj_step(m, d)
            np.clip(d.qvel[:7], -VEL_MAX, VEL_MAX, out=d.qvel[:7])  # strict align: PhysX velocity_limit_sim
            if finger_velocity_limit is not None:
                np.clip(d.qvel[7:9], -finger_velocity_limit, finger_velocity_limit, out=d.qvel[7:9])
                max_delta = finger_velocity_limit * m.opt.timestep
                d.qpos[7:9] = np.clip(
                    d.qpos[7:9], finger_pos_before - max_delta, finger_pos_before + max_delta
                )
            q = d.qpos[:7]
            jr = m.jnt_range[:7]
            valid = jr[:, 1] > jr[:, 0]
            if np.any(valid):
                margin = np.minimum(q[valid] - jr[valid, 0], jr[valid, 1] - q[valid])
                arm_min_limit_margin = min(arm_min_limit_margin, float(np.min(margin)))
            if grasp_assist and glued:
                hp, hq = d.xpos[ctrl.hand], d.xquat[ctrl.hand]
                d.qpos[pj:pj + 3] = hp + Q.quat_apply(hq, rel_pos)
                d.qpos[pj + 3:pj + 7] = Q.quat_mul(hq, rel_quat)
                d.qvel[ctrl.peg_dof:ctrl.peg_dof + 6] = 0

        ph = peg_in_hole_pose(d, ctrl)
        dist3 = np.linalg.norm(ph[:3] - np.array([0.0, 0.0, ASSEMBLED_Z]))
        rot = np.linalg.norm(ph[3:])
        physical_rot = square_symmetry_rot_error(d.xquat[ctrl.hole], d.xquat[ctrl.peg])
        best = min(best, dist3)
        best_rot = min(best_rot, rot)
        best_physical_rot = min(best_physical_rot, physical_rot)
        legacy_strict_pose = bool(dist3 < SUC_POS and rot < SUC_ROT)
        physical_strict_pose = bool(dist3 < SUC_POS and physical_rot < SUC_ROT)
        robot_contact = ctrl.peg_robot_contact(d)
        if legacy_strict_pose:  # retained for model_4550 baseline parity
            success = True
        if physical_strict_pose:
            physical_success = True
        if ever_closed and not close and physical_strict_pose and not robot_contact:
            stable_release_streak += 1
        else:
            stable_release_streak = 0
        max_stable_release_streak = max(max_stable_release_streak, stable_release_streak)
        if stable_release_streak >= STABLE_RELEASE_STEPS:
            stable_success = True
        if render:
            imgs = []
            for c in ("cam_front", "cam_side"):
                cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c)
                rends[c].update_scene(d, camera=cid)
                imgs.append(rends[c].render())
            frames.append(np.concatenate(imgs, axis=1))

    ph = peg_in_hole_pose(d, ctrl)
    final_dist3 = np.linalg.norm(ph[:3] - np.array([0.0, 0.0, ASSEMBLED_Z]))
    final_rot = np.linalg.norm(ph[3:])
    final_physical_rot = square_symmetry_rot_error(d.xquat[ctrl.hole], d.xquat[ctrl.peg])
    final_success = bool(final_dist3 < SUC_POS and final_rot < SUC_ROT)
    physical_final_success = bool(final_dist3 < SUC_POS and final_physical_rot < SUC_ROT)
    return {"success": success, "final_lat": float(np.hypot(ph[0], ph[1])),
            "final_z": float(ph[2]), "final_dist3": float(final_dist3),
            "final_rot": float(final_rot), "final_physical_rot": float(final_physical_rot),
            "best": float(best), "best_rot": float(best_rot),
            "best_physical_rot": float(best_physical_rot),
            "physical_success": bool(physical_success),
            "final_success": final_success, "stable_success": bool(stable_success),
            "physical_final_success": physical_final_success,
            "stable_release_required_steps": STABLE_RELEASE_STEPS,
            "max_stable_release_steps": int(max_stable_release_streak),
            "release_step": release_step, "final_robot_contact": bool(ctrl.peg_robot_contact(d)),
            "closed_steps": int(closed_steps), "grasp_lost_after_close": bool(grasp_lost_after_close),
            "arm_min_limit_margin": float(arm_min_limit_margin), "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="datasets/franka_valset.zarr")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--render", type=int, default=0, help="episode idx to render a video of, -1=none")
    ap.add_argument("--profile", choices=PROFILE_CHOICES, default="legacy_stage1",
                    help="effective sim profile: legacy default, B3 center, B3 ADR sampled, or B3 box-hole ablation")
    ap.add_argument("--sysid_metadata", default=B3_SYSID_METADATA,
                    help="B3 sysid YAML consumed by stage2_b3_* profiles")
    ap.add_argument("--controller_profile", choices=["profile", "stage1", "stage2_deploy"],
                    default="profile", help="single-variable override for OSC controller/action scaling")
    ap.add_argument("--sysid_mode", choices=["profile", "off", "center", "sampled"],
                    default="profile", help="single-variable override for B3 arm sysid")
    ap.add_argument("--gripper_profile", choices=["profile", "legacy", "b3"], default="profile",
                    help="single-variable override for legacy tuned vs B3-aligned gripper actuator")
    ap.add_argument("--gripper_tendon_stiffness", type=float, default=None)
    ap.add_argument("--gripper_tendon_damping", type=float, default=None)
    ap.add_argument("--gripper_tendon_force_limit", type=float, default=None)
    ap.add_argument("--gripper_joint_friction", type=float, default=None)
    ap.add_argument("--gripper_joint_armature", type=float, default=None)
    ap.add_argument("--gripper_contact_timeconst", type=float, default=None,
                    help="override peg-finger explicit-pair solref time constant")
    ap.add_argument("--gripper_contact_dampratio", type=float, default=None,
                    help="override positive-format peg-finger contact damping ratio")
    ap.add_argument("--gripper_contact_stiffness", type=float, default=None,
                    help="use direct-format peg-finger contact stiffness")
    ap.add_argument("--gripper_contact_damping", type=float, default=None,
                    help="direct-format contact damping; default is critical damping")
    ap.add_argument("--gripper_contact_impedance", type=float, default=None,
                    help="constant peg-finger solimp d0=dwidth; must be in (0, 1)")
    ap.add_argument("--adr_seed", type=int, default=0,
                    help="seed for per-episode B3 ADR sampled sysid/gain scales")
    ap.add_argument("--peg_mass", type=float, default=None,
                    help="override peg mass; profile default is 0.02 legacy, 0.001 B3")
    ap.add_argument("--peg_collision", choices=["profile", "box", "mesh"], default="profile",
                    help="override peg collision; profile default is box legacy, exported STL mesh B3")
    ap.add_argument("--hole_collision", choices=["profile", "box_ring", "box_ring_big", "mesh_sdf", "mesh_sdf_big"], default="profile",
                    help="override hole collision; profile default is box-ring legacy, exported STL SDF B3")
    ap.add_argument("--physics_substeps", type=int, default=None,
                    help="MuJoCo substeps per Isaac 1/120-s physics step; policy period stays 0.1 s")
    ap.add_argument("--grasp_assist", action="store_true",
                    help="idealize the grasp (weld peg to hand in capture zone) to isolate "
                         "grasp-contact physics from the rest of the transfer pipeline")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--eval_gains", action="store_true",
                    help="use Stage-2 deploy OSC gains (FRANKA_FR3_RELATIVE_OSC_EVAL: Kp=1000/50, "
                         "scale 0.01/0.002) — required for the gain-robust model_6800 expert")
    ap.add_argument("--big_pad", action="store_true",
                    help="ablation: replace 5 small fingertip pads with one large flat pad per finger")
    ap.add_argument("--box_hole", action="store_true",
                    help="ablation: primitive box-ring hole collision (33mm opening) instead of SDF mesh")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base_profile = make_base_profile(args)

    if args.big_pad:
        global BIG_PAD
        BIG_PAD = True
        print("[eval] BIG_PAD ablation: one large flat pad per finger")
    if args.box_hole:
        print("[eval] BOX_HOLE ablation: primitive box-ring hole collision")
    print(f"[eval] profile={args.profile} controller={base_profile['controller']} "
          f"peg_mass={base_profile['peg_mass']} peg_collision={base_profile['peg_collision']} "
          f"hole_collision={base_profile['hole_collision']} "
          f"gripper={base_profile['gripper']['profile']}")

    policy = FrankaPolicy.load_from_checkpoint(args.checkpoint)
    print(f"[eval] policy iter={policy.ckpt_iter}")

    z = zarr.open(args.zarr, mode="r")
    raw = np.asarray(z["data/raw_state"])
    ends = np.asarray(z["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    n = min(args.episodes, len(starts))

    effective_config = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "script": os.path.abspath(__file__),
        "profile": args.profile,
        "base_profile": base_profile,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_md5": file_md5(args.checkpoint),
        "policy_iter": int(policy.ckpt_iter),
        "zarr": os.path.abspath(args.zarr),
        "zarr_signature": zarr_signature(z),
        "runtime_versions": runtime_versions(),
        "episodes_requested": int(args.episodes),
        "episodes_resolved": int(n),
        "success_metric": {
            "legacy_ever_hit": {"dist3_m_lt": SUC_POS, "raw_quaternion_rot_rad_lt": SUC_ROT},
            "physical_ever_hit": {"dist3_m_lt": SUC_POS, "square_symmetry_rot_rad_lt": SUC_ROT},
            "legacy_final": {"dist3_m_lt": SUC_POS, "raw_quaternion_rot_rad_lt": SUC_ROT},
            "physical_final": {"dist3_m_lt": SUC_POS, "square_symmetry_rot_rad_lt": SUC_ROT},
            "stable_after_release": {
                "dist3_m_lt": SUC_POS,
                "square_symmetry_rot_rad_lt": SUC_ROT,
                "consecutive_policy_steps": STABLE_RELEASE_STEPS,
                "duration_s": STABLE_RELEASE_STEPS * DECIM * SIM_DT,
                "requires_gripper_open": True,
                "requires_no_robot_contact": True,
            },
            "assembled_z": ASSEMBLED_Z,
        },
        "git": {"sim2sim_cotrain": git_state(ROOT)},
        "eval_snapshot_id": open(
            os.path.join(EVAL_SNAPSHOT, "SNAPSHOT_ID"), encoding="utf-8"
        ).read().strip(),
        "alignment_status": [
            {"field": "real fingertips", "status": "MATCH", "value": MIMIC_FINGER},
            {"field": "strict success metric", "status": "MATCH", "value": "2.5mm and 0.025rad"},
            {"field": "anti-hack success metric", "status": "MATCH",
             "value": "released + no robot contact + strict pose held for 0.5s"},
            {"field": "gripper actuator", "status": "MATCH" if args.profile.startswith("stage2_b3")
             and base_profile["gripper"]["profile"] == "b3_training_center" else "PROFILE_DEPENDENT",
             "value": base_profile["gripper"]},
            {"field": "B3 sysid", "status": "MATCH" if base_profile["apply_sysid"] else "NOT_APPLIED",
             "value": ("ADR sampled per episode" if base_profile["sample_sysid"] else "center")
             if base_profile["apply_sysid"] else "n/a"},
            {"field": "motor delay", "status": "INTENDED_OMISSION", "value": base_profile["motor_delay_note"]},
        ],
    }
    with open(os.path.join(args.out, "effective_config.json"), "w") as f:
        json.dump(to_jsonable(effective_config), f, indent=2, sort_keys=True)

    results = []
    csv_rows = []
    episode_configs = []
    for e in range(n):
        episode_cfg = make_episode_config(base_profile, e, args.adr_seed)
        ctrl_vals = episode_cfg["controller_values"]
        set_controller_gains(ctrl_vals["scale"], ctrl_vals["kp"], ctrl_vals["zeta"])
        episode_configs.append(to_jsonable(episode_cfg))
        s0 = int(starts[e])
        r0 = raw[s0]
        if episode_cfg["robot_root_frame_init"]:
            root_pos, root_quat = r0[18:21], r0[21:25]
            root_quat_inv = Q.quat_inv(root_quat)
            peg_pos, peg_quat = Q.subtract_frame_transforms(root_pos, root_quat, r0[31:34], r0[34:38])
            hole_pos, hole_quat = Q.subtract_frame_transforms(root_pos, root_quat, r0[44:47], r0[47:51])
            peg_vel = np.concatenate([
                Q.quat_apply(root_quat_inv, r0[38:41] - r0[25:28]),
                Q.quat_apply(root_quat_inv, r0[41:44] - r0[28:31]),
            ])
        else:
            peg_pos, peg_quat = r0[31:34].copy(), r0[34:38].copy()
            hole_pos, hole_quat = r0[44:47].copy(), r0[47:51].copy()
            peg_vel = np.zeros(6)
        init = {
            "q9": r0[0:9].copy(),
            "qvel9": r0[9:18].copy() if episode_cfg["robot_root_frame_init"] else np.zeros(9),
            "peg_pos": peg_pos,
            "peg_quat": peg_quat,
            "peg_vel": peg_vel,
            "hole_pos": hole_pos,
            "hole_quat": hole_quat,
        }
        m = build_model(init["hole_pos"], init["hole_quat"], episode_cfg)
        ctrl = Controller(m)
        obb = ObsBuilder(ctrl)
        do_render = (e == args.render)
        res = run_episode(
            m,
            ctrl,
            obb,
            policy,
            init,
            render=do_render,
            grasp_assist=args.grasp_assist,
            finger_velocity_limit=episode_cfg["finger_velocity_limit"],
        )
        results.append(res)
        print(f"[eval] ep{e:2d}: success={res['success']}  final_lat={res['final_lat']:.4f} "
              f"final_z={res['final_z']:.4f} final_rot={res['final_rot']:.4f} "
              f"physical_rot={res['final_physical_rot']:.4f} "
              f"best_seat={res['best']:.4f} final={res['final_success']} "
              f"stable={res['stable_success']} closed_steps={res['closed_steps']}")
        row = {k: v for k, v in res.items() if k != "frames"}
        row["episode"] = e
        row["start_index"] = s0
        row["profile"] = args.profile
        row["kp"] = json.dumps(to_jsonable(KP))
        row["zeta"] = json.dumps(to_jsonable(ZETA))
        row["sysid_armature"] = json.dumps(to_jsonable(episode_cfg["sysid"]["armature"])) if episode_cfg["sysid"] is not None else ""
        row["sysid_static_friction"] = json.dumps(to_jsonable(episode_cfg["sysid"]["static_friction"])) if episode_cfg["sysid"] is not None else ""
        csv_rows.append(row)
        if do_render and res["frames"]:
            vid = os.path.join(args.out, f"closedloop_ep{e}.mp4")
            with imageio.get_writer(vid, fps=20, macro_block_size=1) as wr:
                for fr in res["frames"]:
                    wr.append_data(fr)
            print(f"[eval] video -> {vid}")

    sr = np.mean([r["success"] for r in results])
    physical_sr = np.mean([r["physical_success"] for r in results])
    final_sr = np.mean([r["final_success"] for r in results])
    physical_final_sr = np.mean([r["physical_final_success"] for r in results])
    stable_sr = np.mean([r["stable_success"] for r in results])
    if csv_rows:
        csv_path = os.path.join(args.out, "episode_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[eval] per-episode results -> {csv_path}")
    summary = {
        "profile": args.profile,
        "checkpoint": os.path.abspath(args.checkpoint),
        "episodes": int(n),
        "successes": int(sum(r["success"] for r in results)),
        "sr": float(sr),
        "physical_successes": int(sum(r["physical_success"] for r in results)),
        "physical_sr": float(physical_sr),
        "final_successes": int(sum(r["final_success"] for r in results)),
        "final_sr": float(final_sr),
        "physical_final_successes": int(sum(r["physical_final_success"] for r in results)),
        "physical_final_sr": float(physical_final_sr),
        "stable_successes": int(sum(r["stable_success"] for r in results)),
        "stable_sr": float(stable_sr),
        "episode_configs": episode_configs,
        "episode_results": [{k: v for k, v in r.items() if k != "frames"} for r in results],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(to_jsonable(summary), f, indent=2, sort_keys=True)
    effective_config["summary"] = summary
    with open(os.path.join(args.out, "effective_config.json"), "w") as f:
        json.dump(to_jsonable(effective_config), f, indent=2, sort_keys=True)
    print(f"\n[eval] ===== ever-hit SR = {sr:.3f} ({sum(r['success'] for r in results)}/{n}) =====")
    print(f"[eval] ===== physical ever-hit SR = {physical_sr:.3f} "
          f"({sum(r['physical_success'] for r in results)}/{n}) =====")
    print(f"[eval] ===== final SR = {final_sr:.3f} ({sum(r['final_success'] for r in results)}/{n}) =====")
    print(f"[eval] ===== physical final SR = {physical_final_sr:.3f} "
          f"({sum(r['physical_final_success'] for r in results)}/{n}) =====")
    print(f"[eval] ===== stable-after-release SR = {stable_sr:.3f} "
          f"({sum(r['stable_success'] for r in results)}/{n}) =====")
    print(f"[eval] (IsaacLab task_0 teacher reference ~0.90)")


if __name__ == "__main__":
    main()
