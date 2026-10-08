"""Audit direct Isaac-to-MuJoCo replay for the CupCake-on-Plate task.

The source Zarr is authoritative: every episode supplies the exact reset,
controller gains, actuator properties, body inertias, and contact materials.
Recorded policy actions are replayed without policy inference or training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop_eval as CL
import compare_peginsert_continuous_mujoco as B3
import cupcake_mujoco_model as CupCake
import quat_utils as Q


RUNTIME_FIELDS = (
    "gripper_joint_stiffness",
    "gripper_joint_damping",
    "gripper_joint_effort_limit",
    "gripper_joint_armature",
    "gripper_joint_friction_static",
    "gripper_joint_friction_dynamic",
    "gripper_joint_friction_viscous",
)


def quaternion_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)))


def parse_episodes(spec: str, count: int) -> list[int]:
    if spec == "all":
        return list(range(count))
    result: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            begin_text, end_text = item.split(":", 1)
            begin = int(begin_text) if begin_text else 0
            end = int(end_text) if end_text else count
            result.extend(range(begin, end))
        else:
            result.append(int(item))
    result = list(dict.fromkeys(result))
    invalid = [episode for episode in result if episode < 0 or episode >= count]
    if not result or invalid:
        raise ValueError(f"invalid --episodes={spec!r}; count={count}, invalid={invalid}")
    return result


def constant_field(store, key: str, start: int, end: int) -> np.ndarray:
    path = f"data/{key}"
    if path not in store:
        raise KeyError(f"missing runtime field: {path}")
    values = np.asarray(store[path][start:end], dtype=np.float64)
    drift = float(np.max(np.abs(values - values[0])))
    if drift > 1.0e-6:
        raise ValueError(f"{path} changes within episode: max drift={drift}")
    return values[0]


def configure_timing(store) -> tuple[float, int, float]:
    physics_dt = float(store.attrs["physics_dt_s"])
    decimation = int(store.attrs["decimation"])
    policy_dt = float(store.attrs["policy_dt_s"])
    if abs(physics_dt * decimation - policy_dt) > 1.0e-9:
        raise ValueError("source physics_dt * decimation != policy_dt")
    CL.SIM_DT = physics_dt
    CL.DECIM = decimation
    return physics_dt, decimation, policy_dt


def validate_source_asset_contract(store) -> None:
    if store.attrs.get("task_pair") != "CupCake__Plate":
        raise ValueError(f"expected task_pair=CupCake__Plate, got {store.attrs.get('task_pair')}")
    insertive = str(store.attrs.get("insertive_object_usd", ""))
    receptive = str(store.attrs.get("receptive_object_usd", ""))
    if "cupcake" not in insertive.lower() or "plate" not in receptive.lower():
        raise ValueError(
            "source rollout was not generated with CupCake/Plate assets: "
            f"insertive={insertive or '<missing>'}, receptive={receptive or '<missing>'}"
        )


def audit_source_action_contract(store) -> dict:
    required = (
        "data/action",
        "data/arm_raw_action",
        "data/arm_processed_action",
        "data/arm_scale",
        "data/arm_ee_pos",
        "data/arm_ee_quat",
        "data/arm_desired_pos",
        "data/arm_desired_quat",
    )
    missing = [key for key in required if key not in store]
    if missing:
        return {"complete": False, "missing": missing}
    action = np.asarray(store["data/action"][:, :6], dtype=np.float64)
    raw = np.asarray(store["data/arm_raw_action"], dtype=np.float64)
    processed = np.asarray(store["data/arm_processed_action"], dtype=np.float64)
    scale = np.asarray(store["data/arm_scale"], dtype=np.float64)
    ee_pos = np.asarray(store["data/arm_ee_pos"], dtype=np.float64)
    ee_quat = np.asarray(store["data/arm_ee_quat"], dtype=np.float64)
    desired_pos = np.asarray(store["data/arm_desired_pos"], dtype=np.float64)
    desired_quat = np.asarray(store["data/arm_desired_quat"], dtype=np.float64)
    predicted_quat = np.asarray(
        [
            Q.quat_mul(CL.quat_from_aa(delta[3:6]), quaternion)
            for delta, quaternion in zip(processed, ee_quat)
        ]
    )
    quaternion_errors = np.asarray(
        [quaternion_angle(actual, expected) for actual, expected in zip(predicted_quat, desired_quat)]
    )
    return {
        "complete": True,
        "raw_action_max_abs_error": float(np.max(np.abs(raw - action))),
        "processed_action_max_abs_error": float(np.max(np.abs(processed - raw * scale))),
        "desired_position_max_abs_error_m": float(
            np.max(np.abs(desired_pos - (ee_pos + processed[:, :3])))
        ),
        "desired_orientation_max_error_rad": float(np.max(quaternion_errors)),
    }


def set_fixed_scene_poses(model: mujoco.MjModel, raw: np.ndarray) -> None:
    plate_pos, plate_quat = CupCake.pose_in_robot_root(
        raw, slice(44, 47), slice(47, 51)
    )
    plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    model.body_pos[plate] = plate_pos
    model.body_quat[plate] = plate_quat

    table_pos, table_quat = Q.subtract_frame_transforms(
        raw[18:21],
        raw[21:25],
        CupCake.TABLE_WORLD_POS,
        CupCake.TABLE_WORLD_QUAT,
    )
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    model.body_pos[table] = table_pos
    model.body_quat[table] = table_quat


def configure_gripper_runtime(
    model: mujoco.MjModel,
    ctrl: CupCake.Controller,
    store,
    start: int,
    end: int,
) -> dict:
    runtime = {key: constant_field(store, key, start, end) for key in RUNTIME_FIELDS}
    for key, values in runtime.items():
        if values.shape != (2,):
            raise ValueError(f"{key} must have two finger values, got {values.shape}")
        if not np.allclose(values, values[0], atol=1.0e-6):
            raise ValueError(f"tendon mapping requires symmetric fingers: {key}={values}")

    # One split tendon coordinate drives two symmetric finger joints, hence
    # the tendon PD/limit is twice the exported per-finger PhysX value.
    stiffness = 2.0 * float(runtime["gripper_joint_stiffness"][0])
    damping = 2.0 * float(runtime["gripper_joint_damping"][0])
    force_limit = 2.0 * float(runtime["gripper_joint_effort_limit"][0])
    actuator = ctrl.grip_act
    model.actuator_gainprm[actuator] = 0.0
    model.actuator_gainprm[actuator, 0] = stiffness * 0.04 / 255.0
    model.actuator_biasprm[actuator] = 0.0
    model.actuator_biasprm[actuator, 1] = -stiffness
    model.actuator_biasprm[actuator, 2] = -damping
    model.actuator_forcerange[actuator] = [-force_limit, force_limit]
    model.dof_armature[7:9] = runtime["gripper_joint_armature"]
    model.dof_frictionloss[7:9] = runtime["gripper_joint_friction_static"]
    model.dof_damping[7:9] = runtime["gripper_joint_friction_viscous"]
    return {
        **runtime,
        "tendon_stiffness": stiffness,
        "tendon_damping": damping,
        "tendon_force_limit": force_limit,
    }


def apply_cupcake_materials(model: mujoco.MjModel, scene: dict) -> dict:
    applied = B3.apply_runtime_contact_materials(model, scene)
    mapping = {}
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name and name.startswith("cupcake_collision"):
            mapping[name] = "insertive_object_material_properties"
        elif name and name.startswith("plate_collision"):
            mapping[name] = "receptive_object_material_properties"
    for geom_name, source_key in mapping.items():
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        coefficient = float(np.asarray(scene[source_key]).reshape(-1, 3)[0, 1])
        model.geom_friction[geom, 0] = coefficient
        applied[geom_name] = coefficient
    table_materials = np.asarray(scene["table_material_properties"]).reshape(-1, 3)
    if len(table_materials) != len(CupCake.TABLE_COLLISION_BOXES):
        raise ValueError(
            "table material count does not match pat_vention collision boxes: "
            f"{len(table_materials)} != {len(CupCake.TABLE_COLLISION_BOXES)}"
        )
    for index, material in enumerate(table_materials):
        geom_name = "table_top" if index == 0 else f"table_collision_{index}"
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        coefficient = float(material[1])
        model.geom_friction[geom, 0] = coefficient
        applied[geom_name] = coefficient
    return applied


def apply_pair_friction_combine(
    model: mujoco.MjModel, scene: dict, mode: str
) -> dict:
    if mode not in {"legacy_b3", "physx_average"}:
        raise ValueError(f"unknown contact friction combine mode: {mode}")
    cupcake_mu = float(
        np.asarray(scene["insertive_object_material_properties"])
        .reshape(-1, 3)[0, 1]
    )
    robot_materials = np.asarray(scene["robot_material_properties"]).reshape(-1, 3)
    robot_names = list(scene["robot_body_names"])
    robot_mu = {
        name: float(robot_materials[index, 1])
        for index, name in enumerate(robot_names[: len(robot_materials)])
    }
    source_mu = {
        "table_top": float(
            np.asarray(scene["table_material_properties"]).reshape(-1, 3)[0, 1]
        ),
        "plate_collision": float(
            np.asarray(scene["receptive_object_material_properties"])
            .reshape(-1, 3)[0, 1]
        ),
        "left_finger_mimic": robot_mu["panda_leftfinger"],
        "right_finger_mimic": robot_mu["panda_rightfinger"],
        "hand_mimic": robot_mu["panda_hand"],
    }
    applied = {}
    for pair in range(model.npair):
        geom1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[pair])
        )
        geom2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[pair])
        )
        if geom1.startswith("cupcake_collision"):
            other = geom2
        elif geom2.startswith("cupcake_collision"):
            other = geom1
        else:
            continue
        source_key = "plate_collision" if other.startswith("plate_collision") else other
        if source_key not in source_mu:
            continue
        other_mu = source_mu[source_key]
        if mode == "legacy_b3" and other.endswith("finger_mimic"):
            coefficient = 2.0
        elif mode == "legacy_b3":
            coefficient = max(cupcake_mu, other_mu)
        else:
            coefficient = 0.5 * (cupcake_mu + other_mu)
        model.pair_friction[pair] = [
            coefficient,
            coefficient,
            0.0,
            0.0,
            0.0,
        ]
        applied[other] = {
            "cupcake_dynamic_friction": cupcake_mu,
            "other_dynamic_friction": other_mu,
            "effective_tangential_friction": coefficient,
            "torsional_and_rolling_friction": [0.0, 0.0, 0.0],
        }
    return applied


def apply_contact_offsets(
    model: mujoco.MjModel, ctrl: CupCake.Controller
) -> dict:
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body in ctrl.robot_body_ids:
            model.geom_margin[geom] = CupCake.ROBOT_CONTACT_OFFSET
            model.geom_gap[geom] = CupCake.ROBOT_CONTACT_OFFSET
        elif body in {ctrl.insertive, ctrl.receptive} or mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body
        ) == "table":
            model.geom_margin[geom] = CupCake.CUPCAKE_CONTACT_OFFSET
            model.geom_gap[geom] = CupCake.CUPCAKE_CONTACT_OFFSET

    applied_pairs = {}
    for pair in range(model.npair):
        geom1 = int(model.pair_geom1[pair])
        geom2 = int(model.pair_geom2[pair])
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if ctrl.insertive not in {body1, body2}:
            continue
        other_body = body2 if body1 == ctrl.insertive else body1
        margin = (
            CupCake.ROBOT_OBJECT_CONTACT_MARGIN
            if other_body in ctrl.robot_body_ids
            else CupCake.OBJECT_OBJECT_CONTACT_MARGIN
        )
        model.pair_margin[pair] = margin
        model.pair_gap[pair] = margin
        names = sorted(
            [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            ]
        )
        applied_pairs["+".join(names)] = margin
    return {
        "cupcake_plate_table_shape_offset_m": CupCake.CUPCAKE_CONTACT_OFFSET,
        "robot_shape_offset_m": CupCake.ROBOT_CONTACT_OFFSET,
        "pair_margins_m": applied_pairs,
        "rest_surface_offset_m": 0.0,
    }


def configure_episode(
    model: mujoco.MjModel,
    ctrl: CupCake.Controller,
    store,
    start: int,
    end: int,
    raw0: np.ndarray,
    friction_combine: str = "physx_average",
    arm_friction: str = "dynamic",
    arm_armature: str = "source",
    arm_armature_scale: float = 1.0,
) -> dict:
    runtime = B3.load_isaac_runtime(store, start, end)
    zeta = runtime["arm_kd"] / (2.0 * np.sqrt(runtime["arm_kp"]))
    CL.set_controller_gains(runtime["arm_scale"], runtime["arm_kp"], zeta)
    CL.TAU_MAX = runtime["arm_torque_max"].copy()
    if arm_armature not in {"source", "none"}:
        raise ValueError(f"unknown arm armature mapping: {arm_armature}")
    model.dof_armature[:7] = (
        arm_armature_scale * runtime["arm_joint_armature"]
        if arm_armature == "source"
        else 0.0
    )
    if arm_friction not in {"physx", "dynamic", "static", "none"}:
        raise ValueError(f"unknown arm friction mapping: {arm_friction}")
    if arm_friction == "none":
        model.dof_frictionloss[:7] = 0.0
        model.dof_damping[:7] = 0.0
    else:
        coulomb_profile = "static" if arm_friction == "physx" else arm_friction
        model.dof_frictionloss[:7] = runtime[f"arm_joint_friction_{coulomb_profile}"]
        model.dof_damping[:7] = runtime["arm_joint_friction_viscous"]

    gripper = configure_gripper_runtime(model, ctrl, store, start, end)
    scene = B3.load_runtime_scene_properties(store, start, end)
    set_fixed_scene_poses(model, raw0)
    cupcake_visual = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cupcake_visual"
    )
    if int(model.geom_sameframe[cupcake_visual]) in {
        int(mujoco.mjtSameFrame.mjSAMEFRAME_INERTIA),
        int(mujoco.mjtSameFrame.mjSAMEFRAME_INERTIAROT),
    }:
        # The compiler optimized this mass-carrying mesh against its original
        # inertial frame. Runtime PhysX inertia replacement must not rotate it.
        model.geom_sameframe[cupcake_visual] = (
            mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        )
    masses = B3.apply_runtime_body_masses(model, ctrl, scene, include_robot=True)
    materials = apply_cupcake_materials(model, scene)
    pair_friction = apply_pair_friction_combine(model, scene, friction_combine)
    contact_offsets = apply_contact_offsets(model, ctrl)
    return {
        "arm": runtime,
        "arm_armature_mapping": arm_armature,
        "arm_armature_scale": arm_armature_scale,
        "arm_friction_mapping": arm_friction,
        "gripper": gripper,
        "cupcake_visual_sameframe": int(model.geom_sameframe[cupcake_visual]),
        "masses_applied_kg": masses,
        "contact_materials_applied": materials,
        "pair_friction_combine": friction_combine,
        "pair_friction_applied": pair_friction,
        "contact_offsets_applied": contact_offsets,
    }


def source_object_state(raw: np.ndarray) -> tuple[np.ndarray, ...]:
    position, quaternion = CupCake.pose_in_robot_root(
        raw, slice(31, 34), slice(34, 38)
    )
    root_quat_inv = Q.quat_inv(raw[21:25])
    linear_velocity = Q.quat_apply(root_quat_inv, raw[38:41] - raw[25:28])
    angular_velocity = Q.quat_apply(root_quat_inv, raw[41:44] - raw[28:31])
    return position, quaternion, linear_velocity, angular_velocity


def measure(
    data: mujoco.MjData,
    ctrl: CupCake.Controller,
    reference: np.ndarray,
    source_position_error: float,
    source_orientation_error: float,
    source_success: bool,
    live_close: bool,
    recorded_close: bool,
) -> dict:
    ref_pos, ref_quat, ref_linvel, ref_angvel = source_object_state(reference)
    qpos = ctrl.insertive_qpos
    dof = ctrl.insertive_dof
    metrics = CupCake.success_metrics(data, ctrl)
    return {
        "arm_joint_pos_l2_rad": float(np.linalg.norm(data.qpos[:7] - reference[:7])),
        "arm_joint_pos_max_rad": float(np.max(np.abs(data.qpos[:7] - reference[:7]))),
        "arm_joint_vel_l2_radps": float(
            np.linalg.norm(data.qvel[:7] - reference[9:16])
        ),
        "finger_pos_l2_m": float(np.linalg.norm(data.qpos[7:9] - reference[7:9])),
        "finger_vel_l2_mps": float(
            np.linalg.norm(data.qvel[7:9] - reference[16:18])
        ),
        "cupcake_pos_m": float(np.linalg.norm(data.qpos[qpos : qpos + 3] - ref_pos)),
        "cupcake_rot_rad": quaternion_angle(
            data.qpos[qpos + 3 : qpos + 7], ref_quat
        ),
        "cupcake_linvel_l2_mps": float(
            np.linalg.norm(data.qvel[dof : dof + 3] - ref_linvel)
        ),
        "cupcake_angvel_l2_radps": float(
            np.linalg.norm(data.qvel[dof + 3 : dof + 6] - ref_angvel)
        ),
        "mujoco_position_error_m": float(metrics["position_error_m"]),
        "source_position_error_m": float(source_position_error),
        "task_position_error_delta_m": float(
            metrics["position_error_m"] - source_position_error
        ),
        "mujoco_orientation_xy_error_rad": float(
            metrics["orientation_xy_error_rad"]
        ),
        "source_orientation_xy_error_rad": float(source_orientation_error),
        "task_orientation_error_delta_rad": float(
            metrics["orientation_xy_error_rad"] - source_orientation_error
        ),
        "mujoco_strict_success": int(metrics["strict_success"]),
        "source_strict_success": int(source_success),
        "strict_success_match": int(bool(metrics["strict_success"]) == source_success),
        "live_grasp_guard_close": int(live_close),
        "recorded_gripper_close": int(recorded_close),
        "grasp_guard_match": int(live_close == recorded_close),
    }


def summary_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"transitions": 0}
    output: dict[str, object] = {"transitions": len(rows)}
    excluded = {
        "episode",
        "step_zero_based",
        "mujoco_strict_success",
        "source_strict_success",
        "strict_success_match",
        "live_grasp_guard_close",
        "recorded_gripper_close",
        "grasp_guard_match",
        "robot_cupcake_contact",
        "robot_external_contact",
        "robot_self_contact",
        "robot_external_contact_bodies",
    }
    for key in rows[0]:
        if key in excluded:
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[key] = {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    output["strict_success_match_rate"] = float(
        np.mean([row["strict_success_match"] for row in rows])
    )
    output["grasp_guard_match_rate"] = float(
        np.mean([row["grasp_guard_match"] for row in rows])
    )
    return output


def contact_partitioned_summary(rows: list[dict]) -> dict:
    return {
        "all": summary_stats(rows),
        "robot_external_contact": summary_stats(
            [row for row in rows if row["robot_external_contact"]]
        ),
        "no_robot_external_contact": summary_stats(
            [row for row in rows if not row["robot_external_contact"]]
        ),
    }


def robot_contact_state(
    model: mujoco.MjModel, data: mujoco.MjData, ctrl: CupCake.Controller
) -> tuple[bool, bool, bool, set[str]]:
    cupcake_contact = False
    external_contact = False
    self_contact = False
    external_bodies = set()
    for contact in data.contact[: data.ncon]:
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        robot1 = body1 in ctrl.robot_body_ids
        robot2 = body2 in ctrl.robot_body_ids
        if robot1 and robot2:
            self_contact = True
            continue
        if not (robot1 or robot2):
            continue
        external_contact = True
        other = body2 if robot1 else body1
        cupcake_contact = cupcake_contact or other == ctrl.insertive
        external_bodies.add(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
        )
    return cupcake_contact, external_contact, self_contact, external_bodies


def step_recorded_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: CupCake.Controller,
    torque_substeps: np.ndarray,
    gripper_target: np.ndarray,
    source_q_substeps: np.ndarray,
    source_dq_substeps: np.ndarray,
    source_next: np.ndarray,
    arm_velocity_limits: np.ndarray | None,
    physx_static_friction: np.ndarray | None = None,
    physx_dynamic_friction: np.ndarray | None = None,
    friction_slip_velocity: float = 1.0e-4,
) -> dict:
    target = np.asarray(gripper_target, dtype=np.float64)
    if target.shape != (2,) or not np.allclose(target, target[0], atol=1.0e-6):
        raise ValueError(f"tendon replay requires symmetric finger targets: {target}")
    data.ctrl[ctrl.grip_act] = float(np.clip(target.mean() / 0.04 * 255.0, 0.0, 255.0))
    physics_substeps = max(1, int(round(CL.SIM_DT / model.opt.timestep)))
    q_errors = []
    dq_errors = []
    cupcake_contact, external_contact, self_contact, external_bodies = (
        robot_contact_state(model, data, ctrl)
    )
    for tick, torque in enumerate(np.asarray(torque_substeps, dtype=np.float64)):
        # The exported substep state is the state at which this effort target
        # was evaluated, before the corresponding 1/120-s PhysX step.
        q_errors.append(float(np.linalg.norm(data.qpos[:7] - source_q_substeps[tick])))
        dq_errors.append(float(np.linalg.norm(data.qvel[:7] - source_dq_substeps[tick])))
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[:7] = torque
        if physx_static_friction is not None:
            sliding = np.abs(data.qvel[:7]) > friction_slip_velocity
            data.qfrc_applied[:7] += np.where(
                sliding,
                np.sign(data.qvel[:7])
                * (physx_static_friction - physx_dynamic_friction),
                0.0,
            )
        for _ in range(physics_substeps):
            mujoco.mj_step(model, data)
            tick_contact = robot_contact_state(model, data, ctrl)
            cupcake_contact = cupcake_contact or tick_contact[0]
            external_contact = external_contact or tick_contact[1]
            self_contact = self_contact or tick_contact[2]
            external_bodies.update(tick_contact[3])
        if arm_velocity_limits is not None:
            np.clip(
                data.qvel[:7],
                -arm_velocity_limits,
                arm_velocity_limits,
                out=data.qvel[:7],
            )
    q_errors.append(float(np.linalg.norm(data.qpos[:7] - source_next[:7])))
    dq_errors.append(float(np.linalg.norm(data.qvel[:7] - source_next[9:16])))
    return {
        "arm_joint_tick_pos_l2_median_rad": float(np.median(q_errors)),
        "arm_joint_tick_pos_l2_p95_rad": float(np.percentile(q_errors, 95)),
        "arm_joint_tick_pos_l2_max_rad": float(np.max(q_errors)),
        "arm_joint_tick_vel_l2_median_radps": float(np.median(dq_errors)),
        "arm_joint_tick_vel_l2_p95_radps": float(np.percentile(dq_errors, 95)),
        "arm_joint_tick_vel_l2_max_radps": float(np.max(dq_errors)),
        "robot_cupcake_contact": int(cupcake_contact),
        "robot_external_contact": int(external_contact),
        "robot_self_contact": int(self_contact),
        "robot_external_contact_bodies": ";".join(sorted(external_bodies)),
    }


def render_reference_pair(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    reference_data: mujoco.MjData,
    replay_data: mujoco.MjData,
    camera: str,
    label: str,
) -> np.ndarray:
    renderer.update_scene(reference_data, camera=camera)
    left = renderer.render().copy()
    renderer.update_scene(replay_data, camera=camera)
    right = renderer.render().copy()
    gap = np.zeros((left.shape[0], 6, 3), dtype=left.dtype)
    image = np.concatenate([left, gap, right], axis=1)
    try:
        from PIL import Image, ImageDraw

        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas.width, 30], fill=(0, 0, 0))
        draw.text((8, 7), label, fill=(255, 255, 255))
        return np.asarray(canvas)
    except Exception:
        return image


def reset_geometry_audit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: CupCake.Controller,
    raw: np.ndarray,
    render_path: Path | None,
) -> dict:
    CupCake.set_raw_state(model, data, ctrl, raw)
    cupcake_ref, cupcake_quat_ref, _, _ = source_object_state(raw)
    plate_ref, plate_quat_ref = CupCake.pose_in_robot_root(
        raw, slice(44, 47), slice(47, 51)
    )
    contacts: dict[str, int] = {}
    distances = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        names = sorted(
            [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
            ]
        )
        key = "+".join(names)
        contacts[key] = contacts.get(key, 0) + 1
        distances.append(float(contact.dist))

    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=480, width=640)
        try:
            renderer.update_scene(data, camera="cam_front")
            imageio.imwrite(render_path, renderer.render())
        finally:
            renderer.close()

    return {
        "cupcake_position_reset_error_m": float(
            np.linalg.norm(data.xpos[ctrl.insertive] - cupcake_ref)
        ),
        "cupcake_rotation_reset_error_rad": quaternion_angle(
            data.xquat[ctrl.insertive], cupcake_quat_ref
        ),
        "plate_position_reset_error_m": float(
            np.linalg.norm(data.xpos[ctrl.receptive] - plate_ref)
        ),
        "plate_rotation_reset_error_rad": quaternion_angle(
            data.xquat[ctrl.receptive], plate_quat_ref
        ),
        "arm_joint_reset_max_error_rad": float(
            np.max(np.abs(data.qpos[:7] - raw[:7]))
        ),
        "finger_reset_max_error_m": float(
            np.max(np.abs(data.qpos[7:9] - raw[7:9]))
        ),
        "contact_count": int(data.ncon),
        "contact_body_pairs": contacts,
        "minimum_contact_distance_m": min(distances) if distances else None,
        "table_top_world_z_m": float(
            data.geom_xpos[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"), 2
            ]
            + model.geom_size[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"), 2
            ]
        ),
        "table_top_world_z_error_m": float(
            data.geom_xpos[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"), 2
            ]
            + model.geom_size[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"), 2
            ]
            - CupCake.TABLE_TOP_WORLD_Z
        ),
        "render": str(render_path) if render_path is not None else None,
    }


def run_mode(
    mode: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: CupCake.Controller,
    store,
    episode: int,
    start: int,
    end: int,
    limit_steps: int,
    video_path: Path | None = None,
    width: int = 480,
    height: int = 360,
    fps: int = 10,
    arm_velocity_limits: np.ndarray | None = None,
    action_source: str = "recorded_action",
    arm_friction: str = "dynamic",
) -> list[dict]:
    raw = np.asarray(store["data/raw_state"][start:end], dtype=np.float64)
    next_raw = np.asarray(store["data/next_raw_state"][start:end], dtype=np.float64)
    actions = np.asarray(store["data/action"][start:end], dtype=np.float64)
    valid = np.asarray(store["data/transition_valid"][start:end], dtype=bool)
    gripper_processed = np.asarray(
        store["data/gripper_processed_action"][start:end], dtype=np.float64
    )
    if action_source in {"recorded_processed_action", "recorded_target"}:
        fields = ("arm_processed_action", "arm_desired_pos", "arm_desired_quat")
        missing = [key for key in fields if f"data/{key}" not in store]
        if missing:
            raise KeyError(f"{action_source} requires source fields: {missing}")
        arm_processed = np.asarray(
            store["data/arm_processed_action"][start:end], dtype=np.float64
        )
        desired_pos = np.asarray(
            store["data/arm_desired_pos"][start:end], dtype=np.float64
        )
        desired_quat = np.asarray(
            store["data/arm_desired_quat"][start:end], dtype=np.float64
        )
    static_friction = constant_field(store, "arm_joint_friction_static", start, end)
    dynamic_friction = constant_field(store, "arm_joint_friction_dynamic", start, end)
    if action_source == "recorded_torque":
        telemetry_fields = (
            "arm_joint_torque_substeps",
            "arm_joint_pos_substeps",
            "arm_joint_vel_substeps",
        )
        missing = [key for key in telemetry_fields if f"data/{key}" not in store]
        if missing:
            raise KeyError(f"recorded torque replay requires telemetry fields: {missing}")
        torque_substeps = np.asarray(
            store["data/arm_joint_torque_substeps"][start:end], dtype=np.float64
        )
        q_substeps = np.asarray(
            store["data/arm_joint_pos_substeps"][start:end], dtype=np.float64
        )
        dq_substeps = np.asarray(
            store["data/arm_joint_vel_substeps"][start:end], dtype=np.float64
        )
    source_position = np.asarray(
        store["data/position_error_next_m"][start:end], dtype=np.float64
    )
    source_orientation = np.asarray(
        store["data/orientation_xy_error_next_rad"][start:end], dtype=np.float64
    )
    source_success = np.asarray(
        store["data/strict_success_next"][start:end], dtype=bool
    )
    indices = np.flatnonzero(valid)
    if limit_steps > 0:
        indices = indices[:limit_steps]
    if mode == "continuous":
        expected = np.arange(len(indices))
        if not np.array_equal(indices, expected):
            raise ValueError(
                f"continuous replay requires a valid prefix; episode={episode}, "
                f"valid indices begin {indices[:10].tolist()}"
            )
        CupCake.set_raw_state(model, data, ctrl, raw[0])

    renderer = None
    writer = None
    reference_data = None
    if video_path is not None:
        renderer = mujoco.Renderer(model, height=height, width=width)
        writer = imageio.get_writer(video_path, fps=fps, macro_block_size=1)
        reference_data = mujoco.MjData(model)

    rows = []
    try:
        for step in indices:
            if mode == "onestep":
                CupCake.set_raw_state(model, data, ctrl, raw[step])
            live_close = ctrl.grasp_close(data)
            recorded_close = bool(np.mean(gripper_processed[step]) < 0.02)
            if action_source == "recorded_torque":
                tick_metrics = step_recorded_torque(
                    model,
                    data,
                    ctrl,
                    torque_substeps[step],
                    gripper_processed[step],
                    q_substeps[step],
                    dq_substeps[step],
                    next_raw[step],
                    arm_velocity_limits,
                    physx_static_friction=(
                        static_friction if arm_friction == "physx" else None
                    ),
                    physx_dynamic_friction=(
                        dynamic_friction if arm_friction == "physx" else None
                    ),
                )
            else:
                contact_state = robot_contact_state(model, data, ctrl)
                cupcake_contact = contact_state[0]
                external_contact = contact_state[1]
                self_contact = contact_state[2]
                external_bodies = set(contact_state[3])

                def capture_contact(**_kwargs):
                    nonlocal cupcake_contact, external_contact, self_contact
                    tick_contact = robot_contact_state(model, data, ctrl)
                    cupcake_contact = cupcake_contact or tick_contact[0]
                    external_contact = external_contact or tick_contact[1]
                    self_contact = self_contact or tick_contact[2]
                    external_bodies.update(tick_contact[3])

                arm_command = actions[step]
                processed_action = False
                desired_ee_target = None
                if action_source == "recorded_processed_action":
                    arm_command = arm_processed[step]
                    processed_action = True
                elif action_source == "recorded_target":
                    desired_ee_target = np.concatenate(
                        [desired_pos[step], desired_quat[step]]
                    )
                B3.step_action(
                    model,
                    data,
                    ctrl,
                    arm_command,
                    finger_velocity_limits=None,
                    independent_gripper=False,
                    gripper_close_override=recorded_close,
                    jacobian_point="physx_com",
                    nullspace_stiffness=0.0,
                    physx_static_friction=(
                        static_friction if arm_friction == "physx" else None
                    ),
                    physx_dynamic_friction=(
                        dynamic_friction if arm_friction == "physx" else None
                    ),
                    physx_viscous_friction=None,
                    action_reference_blend=0.0,
                    bias_compensation_scale=0.0,
                    effort_scale=1.0,
                    arm_velocity_limits=arm_velocity_limits,
                    tick_callback=capture_contact,
                    processed_action=processed_action,
                    desired_ee_target=desired_ee_target,
                )
                tick_metrics = {
                    "robot_cupcake_contact": int(cupcake_contact),
                    "robot_external_contact": int(external_contact),
                    "robot_self_contact": int(self_contact),
                    "robot_external_contact_bodies": ";".join(
                        sorted(external_bodies)
                    ),
                }
            row = measure(
                data,
                ctrl,
                next_raw[step],
                source_position[step],
                source_orientation[step],
                bool(source_success[step]),
                live_close,
                recorded_close,
            )
            row["episode"] = episode
            row["step_zero_based"] = int(step)
            row.update(tick_metrics)
            rows.append(row)
            if writer is not None:
                CupCake.set_raw_state(model, reference_data, ctrl, next_raw[step])
                label = (
                    f"LEFT Isaac reference | RIGHT MuJoCo {action_source} {mode} | "
                    f"frame={int(step) + 1}/{len(indices)} "
                    f"arm={row['arm_joint_pos_l2_rad']:.3f}rad "
                    f"cupcake={row['cupcake_pos_m'] * 1000.0:.1f}mm"
                )
                writer.append_data(
                    render_reference_pair(
                        model,
                        renderer,
                        reference_data,
                        data,
                        "cam_front",
                        label,
                    )
                )
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(Path(args.zarr).resolve()), mode="r")
    validate_source_asset_contract(store)
    required = (
        "data/raw_state",
        "data/next_raw_state",
        "data/action",
        "data/transition_valid",
        "data/gripper_processed_action",
    )
    missing = [key for key in required if key not in store]
    if missing:
        raise KeyError(f"source Zarr is not rich-replay compatible: {missing}")
    if store["data/raw_state"].shape[1] != 57:
        raise ValueError("expected canonical 57-D raw state")

    timing = configure_timing(store)
    ends = np.asarray(store["meta/episode_ends"], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    episodes = parse_episodes(args.episodes, len(ends))
    first_start, first_end = int(starts[episodes[0]]), int(ends[episodes[0]])
    first_raw = np.asarray(store["data/raw_state"][first_start], dtype=np.float64)
    profile = B3.b3_center_profile(physics_substeps=args.physics_substeps)
    profile["cupcake_collision"] = args.cupcake_collision
    profile["cupcake_plate_collision"] = args.cupcake_plate_collision
    profile["hand_collision"] = args.hand_collision
    profile["finger_collision"] = args.finger_collision
    profile["integrator"] = args.integrator
    profile["contact_solref"] = [args.object_contact_timeconst, 1.0]
    profile["robot_contact_solref"] = [args.robot_contact_timeconst, 1.0]
    model = CupCake.build_model(first_raw, profile_cfg=profile)
    ctrl = CupCake.Controller(model)
    data = mujoco.MjData(model)

    modes = ("onestep", "continuous") if args.mode == "both" else (args.mode,)
    all_rows = {mode: [] for mode in modes}
    videos = []
    episode_audits = []
    for ordinal, episode in enumerate(episodes):
        start, end = int(starts[episode]), int(ends[episode])
        raw0 = np.asarray(store["data/raw_state"][start], dtype=np.float64)
        effective = configure_episode(
            model,
            ctrl,
            store,
            start,
            end,
            raw0,
            friction_combine=args.friction_combine,
            arm_friction=args.arm_friction,
            arm_armature=args.arm_armature,
            arm_armature_scale=args.arm_armature_scale,
        )
        render_path = out / f"geometry_reset_ep{episode:03d}.png" if ordinal == 0 else None
        geometry = reset_geometry_audit(model, data, ctrl, raw0, render_path)
        episode_audits.append(
            {
                "episode": episode,
                "source_indices": [start, end],
                "geometry": geometry,
                "effective_runtime": effective,
            }
        )
        print(
            f"[cupcake-replay] episode={episode} geometry "
            f"cupcake={geometry['cupcake_position_reset_error_m']:.3e}m "
            f"plate={geometry['plate_position_reset_error_m']:.3e}m "
            f"contacts={geometry['contact_count']}",
            flush=True,
        )
        for mode in modes:
            video_path = (
                out / f"{mode}_{args.action_source}_isaac_left_mujoco_right_ep{episode:03d}.mp4"
                if episode == args.render_episode
                else None
            )
            rows = run_mode(
                mode,
                model,
                data,
                ctrl,
                store,
                episode,
                start,
                end,
                args.limit_steps,
                video_path=video_path,
                width=args.width,
                height=args.height,
                fps=args.fps,
                arm_velocity_limits=(
                    CL.VEL_MAX if args.arm_velocity_limit == "source" else None
                ),
                action_source=args.action_source,
                arm_friction=args.arm_friction,
            )
            if video_path is not None:
                videos.append(str(video_path))
            all_rows[mode].extend(rows)
            stats = summary_stats(rows)
            print(
                f"[cupcake-replay] episode={episode} mode={mode} "
                f"steps={len(rows)} arm_p95="
                f"{stats['arm_joint_pos_l2_rad']['p95']:.4g}rad cupcake_p95="
                f"{stats['cupcake_pos_m']['p95'] * 1000.0:.3g}mm",
                flush=True,
            )

    summaries = {}
    for mode, rows in all_rows.items():
        write_csv(out / f"{mode}_errors.csv", rows)
        summaries[mode] = contact_partitioned_summary(rows)

    result = {
        "definition": (
            "exact CupCake Isaac resets and recorded actions replayed in MuJoCo; "
            "no policy inference or training"
        ),
        "zarr": str(Path(args.zarr).resolve()),
        "zarr_signature": CL.zarr_signature(store),
        "checkpoint": store.attrs.get("checkpoint"),
        "checkpoint_sha256": store.attrs.get("checkpoint_sha256"),
        "reset_state_sha256": store.attrs.get("reset_state_sha256"),
        "episodes": episodes,
        "mode": args.mode,
        "limit_steps": args.limit_steps,
        "source_timing": {
            "physics_dt_s": timing[0],
            "decimation": timing[1],
            "policy_dt_s": timing[2],
        },
        "physics_substeps_per_source_tick": args.physics_substeps,
        "videos": videos,
        "physics_profile": {
            "cupcake_collision": args.cupcake_collision,
            "cupcake_plate_collision": args.cupcake_plate_collision,
            "plate_collision": "48-piece CoACD convex decomposition",
            "hand_collision": args.hand_collision,
            "finger_collision": args.finger_collision,
            "friction_combine": args.friction_combine,
            "arm_velocity_limit": args.arm_velocity_limit,
            "arm_friction": args.arm_friction,
            "arm_armature": args.arm_armature,
            "arm_armature_scale": args.arm_armature_scale,
            "action_source": args.action_source,
            "integrator": args.integrator,
            "object_contact_solref": [args.object_contact_timeconst, 1.0],
            "robot_contact_solref": [args.robot_contact_timeconst, 1.0],
        },
        "collision_audit": {
            "sdf_geom_count": int(
                np.count_nonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_SDF)
            ),
            "plate_convex_piece_count": int(
                sum(
                    (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith(
                        "plate_collision"
                    )
                    for geom in range(model.ngeom)
                )
            ),
        },
        "source_action_contract": audit_source_action_contract(store),
        "controller_contract": {
            "arm_command": {
                "recorded_action": "recorded raw policy action through audited B3 OSC",
                "recorded_processed_action": "recorded scaled Cartesian delta through audited B3 OSC",
                "recorded_target": "recorded absolute source EE target through audited B3 OSC",
                "recorded_torque": "recorded 12x120Hz Isaac effort targets",
            }[args.action_source],
            "action_scale": "recorded runtime",
            "kp_kd_torque_limits": "recorded runtime",
            "jacobian_point": "PhysX panda_hand COM",
            "action_reference_blend": 0.0,
            "nullspace_stiffness": 0.0,
            "gripper_command": "recorded processed actuator position target",
            "arm_dynamic_and_viscous_friction": "recorded runtime",
            "robot_and_cupcake_spatial_inertia": "recorded runtime",
            "contact_sliding_friction": "recorded dynamic material coefficient",
            "finger_cupcake_pair": "B3 calibrated contact profile",
            "arm_velocity_limits_radps": (
                CL.VEL_MAX.tolist()
                if args.arm_velocity_limit == "source"
                else None
            ),
            "arm_velocity_limit_enforcement": (
                "post-source-physics-tick qvel projection"
                if args.arm_velocity_limit == "source"
                else "disabled"
            ),
        },
        "episode_audits": episode_audits,
        "summaries": summaries,
    }
    summary_path = out / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(CL.to_jsonable(result), file, indent=2, sort_keys=True)
    print(f"[cupcake-replay] summary -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zarr",
        default="datasets/cupcake_sim2sim_t0_20260815/isaac_fit_b3center.zarr",
    )
    parser.add_argument(
        "--episodes",
        default="0",
        help="comma-separated IDs, Python-style ranges such as 0:8, or all",
    )
    parser.add_argument(
        "--mode", choices=("onestep", "continuous", "both"), default="onestep"
    )
    parser.add_argument("--limit_steps", type=int, default=20)
    parser.add_argument("--physics_substeps", type=int, default=1)
    parser.add_argument("--render_episode", type=int, default=-1)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--integrator",
        choices=("euler", "implicitfast", "implicit", "rk4"),
        default="implicitfast",
    )
    parser.add_argument(
        "--action_source",
        choices=(
            "recorded_action",
            "recorded_processed_action",
            "recorded_target",
            "recorded_torque",
        ),
        default="recorded_action",
    )
    parser.add_argument("--object_contact_timeconst", type=float, default=0.02)
    parser.add_argument("--robot_contact_timeconst", type=float, default=0.005)
    parser.add_argument(
        "--cupcake_collision",
        choices=(
            "convex_decomposition",
            "radial32",
            "coacd",
            "sdf",
            "convex_mesh",
            "compound",
        ),
        default="convex_decomposition",
    )
    parser.add_argument(
        "--cupcake_plate_collision",
        choices=("convex_hull", "radial16", "base_cylinder"),
        default="base_cylinder",
    )
    parser.add_argument(
        "--hand_collision",
        choices=("menagerie", "source_usd", "disabled"),
        default="source_usd",
    )
    parser.add_argument(
        "--finger_collision",
        choices=("mimic", "menagerie", "menagerie_mesh_only"),
        default="mimic",
    )
    parser.add_argument(
        "--friction_combine",
        choices=("legacy_b3", "physx_average"),
        default="physx_average",
    )
    parser.add_argument(
        "--arm_velocity_limit",
        choices=("source", "disabled"),
        default="source",
        help="map the source PhysX Franka velocity limits at each 120 Hz tick",
    )
    parser.add_argument(
        "--arm_friction",
        choices=("physx", "dynamic", "static", "none"),
        default="dynamic",
    )
    parser.add_argument(
        "--arm_armature",
        choices=("source", "none"),
        default="source",
    )
    parser.add_argument("--arm_armature_scale", type=float, default=1.0)
    parser.add_argument(
        "--out", default="log/active/cupcake_sim2sim_20260815/replay_smoke"
    )
    args = parser.parse_args()
    if args.physics_substeps <= 0:
        parser.error("--physics_substeps must be positive")
    if args.object_contact_timeconst <= 0.0 or args.robot_contact_timeconst <= 0.0:
        parser.error("contact time constants must be positive")
    if args.arm_armature_scale < 0.0:
        parser.error("--arm_armature_scale must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
