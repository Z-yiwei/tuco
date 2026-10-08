"""Direct IsaacSim-to-MuJoCo action replay for the B3 PegInsert policy.

The source zarr stores Isaac states before each policy action. This script runs
the recorded action sequence through the B3 MuJoCo profile in two modes:

* onestep: reset MuJoCo to Isaac state(t), execute action(t), compare state(t+1)
* continuous: initialize once at state(0), then execute all actions open-loop

No policy inference or training is involved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop_eval as CL
import obs_reconstruct as R
import quat_utils as Q


USD_FINGER_INERTIA = {
    "mass": 0.02969999983906746,
    "com": np.array([0.0, 0.01140000019222498, 0.023099999874830246]),
    "diagonal": np.array([7.552000170107931e-06, 7.375549557764316e-06, 2.135450131390826e-06]),
    "principal_axes": np.array([0.994766116142273, 0.1021781712770462, 0.0, 0.0]),
}
USD_FINGER_VEL_MAX = np.array([0.05, 0.04], dtype=np.float64)
# PhysX reports the panda_hand body Jacobian at its center of mass, while the
# UWLab OSC action computes pose error at the panda_hand link origin.  The
# panda_hand COM in the link-local frame is exactly (-10, 0, +30) mm.  Keeping
# this seemingly odd hybrid is required for action-level parity with the
# controller that actually generated the Isaac rollout.
ISAAC_HAND_COM_LOCAL = np.array([-0.01, 0.0, 0.03], dtype=np.float64)
B3_NULLSPACE_DEFAULT_POS = np.array(
    [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0],
    dtype=np.float64,
)


def quat_angle(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)))


def b3_center_profile(
    physics_substeps=None,
    hole_collision=None,
    gripper_contact_friction=None,
):
    args = SimpleNamespace(
        profile="stage2_b3_center",
        sysid_metadata=CL.B3_SYSID_METADATA,
        eval_gains=False,
        controller_profile="profile",
        sysid_mode="profile",
        gripper_profile="profile",
        gripper_tendon_stiffness=None,
        gripper_tendon_damping=None,
        gripper_tendon_force_limit=None,
        gripper_joint_friction=None,
        gripper_joint_armature=None,
        gripper_contact_timeconst=None,
        gripper_contact_dampratio=None,
        gripper_contact_stiffness=None,
        gripper_contact_damping=None,
        gripper_contact_impedance=None,
        peg_mass=None,
        peg_collision="profile",
        hole_collision=hole_collision or "profile",
        box_hole=False,
        physics_substeps=physics_substeps,
    )
    base = CL.make_base_profile(args)
    if gripper_contact_friction is not None:
        gripper = dict(base["gripper"])
        gripper["finger_contact_friction"] = np.asarray(
            gripper_contact_friction, dtype=np.float64
        )
        base["gripper"] = gripper
    return CL.make_episode_config(base, episode_idx=0, seed=0)


def pose_in_robot_root(raw):
    root_pos, root_quat = raw[18:21], raw[21:25]
    peg_pos, peg_quat = Q.subtract_frame_transforms(root_pos, root_quat, raw[31:34], raw[34:38])
    hole_pos, hole_quat = Q.subtract_frame_transforms(root_pos, root_quat, raw[44:47], raw[47:51])
    return peg_pos, peg_quat, hole_pos, hole_quat


def set_raw_state(model, data, ctrl, raw):
    mujoco.mj_resetData(model, data)
    data.qpos[:9] = raw[:9]
    data.qvel[:9] = raw[9:18]
    peg_pos, peg_quat, _, _ = pose_in_robot_root(raw)
    data.qpos[ctrl.peg_qpos:ctrl.peg_qpos + 3] = peg_pos
    data.qpos[ctrl.peg_qpos + 3:ctrl.peg_qpos + 7] = peg_quat

    root_quat_inv = Q.quat_inv(raw[21:25])
    root_lin_vel = raw[25:28]
    root_ang_vel = raw[28:31]
    data.qvel[ctrl.peg_dof:ctrl.peg_dof + 3] = Q.quat_apply(
        root_quat_inv, raw[38:41] - root_lin_vel
    )
    data.qvel[ctrl.peg_dof + 3:ctrl.peg_dof + 6] = Q.quat_apply(
        root_quat_inv, raw[41:44] - root_ang_vel
    )
    mujoco.mj_forward(model, data)


def jac_arm_at_isaac_hand_com(model, data, ctrl):
    """Match the PhysX body Jacobian point used by UWLab's OSC action term."""
    point = data.xpos[ctrl.hand] + Q.quat_apply(data.xquat[ctrl.hand], ISAAC_HAND_COM_LOCAL)
    mujoco.mj_jac(model, data, ctrl.jacp, ctrl.jacr, point, ctrl.hand)
    return np.vstack([ctrl.jacp[:, :7], ctrl.jacr[:, :7]])


def selected_arm_jacobian(model, data, ctrl, jacobian_point):
    if jacobian_point == "link_origin":
        return ctrl.jac_arm(data).copy()
    if jacobian_point == "physx_com":
        return jac_arm_at_isaac_hand_com(model, data, ctrl).copy()
    raise ValueError(f"unknown Jacobian point: {jacobian_point}")


def physx_joint_friction_effort(
    model, data, static, dynamic, viscous, iterations
):
    """Approximate PhysX's per-axis accumulated friction-impulse solve."""
    # qacc is the unconstrained velocity change from the live controller,
    # gravity and passive model.  The arm friction terms themselves are
    # disabled in the MuJoCo model for this mapping.
    mujoco.mj_forward(model, data)
    full_mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, full_mass, data.qM)
    response = np.linalg.inv(full_mass)[:7, :7]
    dt = float(model.opt.timestep)
    free_velocity = data.qvel[:7] + dt * data.qacc[:7]
    impulse = np.zeros(7, dtype=np.float64)
    static_bound = dt * static
    for _ in range(iterations):
        for joint in range(7):
            velocity = free_velocity[joint] + response[joint] @ impulse
            candidate = impulse[joint] - velocity / response[joint, joint]
            if abs(candidate) <= static_bound[joint]:
                impulse[joint] = candidate
            else:
                dynamic_bound = dt * (
                    dynamic[joint] + viscous[joint] * abs(velocity)
                )
                impulse[joint] = -np.sign(velocity) * dynamic_bound
    return impulse / dt


def step_action(
    model,
    data,
    ctrl,
    action,
    finger_velocity_limits=None,
    independent_gripper=False,
    gripper_close_override=None,
    jacobian_point="physx_com",
    nullspace_stiffness=0.0,
    nullspace_damping_ratio=1.0,
    physx_static_friction=None,
    physx_dynamic_friction=None,
    physx_viscous_friction=None,
    physx_friction_iterations=4,
    friction_slip_velocity=1.0e-4,
    action_reference_blend=0.0,
    bias_compensation_scale=0.0,
    effort_scale=None,
    arm_velocity_limits=None,
    tick_callback=None,
    processed_action=False,
    desired_ee_target=None,
):
    ee_pos, ee_quat = ctrl.ee_root(data)
    if desired_ee_target is not None:
        desired_ee_target = np.asarray(desired_ee_target, dtype=np.float64)
        if desired_ee_target.shape != (7,):
            raise ValueError(
                f"desired_ee_target must have shape (7,), got {desired_ee_target.shape}"
            )
        ctrl.action_reference_pos = desired_ee_target[:3].copy()
        ctrl.action_reference_quat = desired_ee_target[3:7].copy()
        ctrl.action_reference_quat /= np.linalg.norm(ctrl.action_reference_quat)
    else:
        scaled = np.asarray(action[:6], dtype=np.float64)
        if not processed_action:
            scaled = scaled * CL.SCALE
        blend = action_reference_blend
        if blend > 0.0:
            reference_pos = (1.0 - blend) * ee_pos + blend * ctrl.action_reference_pos
            previous_quat = ctrl.action_reference_quat
            if np.dot(ee_quat, previous_quat) < 0.0:
                previous_quat = -previous_quat
            reference_quat = (1.0 - blend) * ee_quat + blend * previous_quat
            reference_quat /= np.linalg.norm(reference_quat)
        else:
            reference_pos = ee_pos
            reference_quat = ee_quat
        ctrl.action_reference_pos = reference_pos + scaled[:3]
        ctrl.action_reference_quat = Q.quat_mul(
            CL.quat_from_aa(scaled[3:6]), reference_quat
        )
    ee_pos_des = ctrl.action_reference_pos.copy()
    ee_quat_des = ctrl.action_reference_quat.copy()
    close = ctrl.grasp_close(data)
    if gripper_close_override is not None:
        close = bool(gripper_close_override)
    gripper_target = 0.0 if close else 0.04
    if not independent_gripper:
        data.ctrl[ctrl.grip_act] = 0.0 if close else 255.0

    physics_substeps = max(1, int(round(CL.SIM_DT / model.opt.timestep)))
    # Isaac evaluates the action term once per 1/120-s physics tick.  MuJoCo's
    # extra substeps are integration-only: the torque must be held, not
    # recomputed at 1920 Hz.
    for tick in range(CL.DECIM):
        q_before = data.qpos[:7].copy()
        dq_before = data.qvel[:7].copy()
        ee_pos, ee_quat = ctrl.ee_root(data)
        jac = selected_arm_jacobian(model, data, ctrl, jacobian_point)
        ee_vel = jac @ data.qvel[:7]
        pos_err = ee_pos_des - ee_pos
        quat_err = Q.quat_mul(ee_quat_des, Q.quat_inv(ee_quat))
        pose_err = np.concatenate([pos_err, Q.axis_angle_from_quat(quat_err)])
        task_force = CL.KP * pose_err + CL.KD * (-ee_vel)
        joint_torque = jac.T @ task_force
        if bias_compensation_scale:
            mujoco.mj_forward(model, data)
            joint_torque += bias_compensation_scale * data.qfrc_bias[:7]
        if nullspace_stiffness > 0.0:
            nullspace_damping = (
                2.0 * np.sqrt(nullspace_stiffness) * nullspace_damping_ratio
            )
            joint_torque += (
                nullspace_stiffness * (B3_NULLSPACE_DEFAULT_POS - data.qpos[:7])
                - nullspace_damping * data.qvel[:7]
            )
        # Match UWLab exactly: task and nullspace torques are summed first,
        # then the total joint effort is clipped once.
        effort_command = np.clip(joint_torque, -CL.TAU_MAX, CL.TAU_MAX)
        data.qfrc_applied[:7] = effort_command * effort_scale
        if physx_static_friction is not None:
            # MuJoCo's frictionloss constraint supplies the PhysX static bound.
            # Once a joint is sliding, cancel only Ts-Td so the net Coulomb
            # effort is Td.  This is a physical engine mapping, independent of
            # the recorded trajectory; it never reads a source state.
            sliding = np.abs(data.qvel[:7]) > friction_slip_velocity
            data.qfrc_applied[:7] += np.where(
                sliding,
                np.sign(data.qvel[:7])
                * (physx_static_friction - physx_dynamic_friction),
                0.0,
            )
        elif physx_viscous_friction is not None:
            data.qfrc_applied[:7] += physx_joint_friction_effort(
                model,
                data,
                physx_dynamic_friction[0],
                physx_dynamic_friction[1],
                physx_viscous_friction,
                physx_friction_iterations,
            )
        if independent_gripper:
            finger_force = 1000.0 * (gripper_target - data.qpos[7:9]) - 14.0 * data.qvel[7:9]
            finger_force = np.clip(finger_force, -60.0, 60.0)
        else:
            finger_force = np.zeros(2, dtype=np.float64)
        for _ in range(physics_substeps):
            # Stop only the component of actuator drive that would accelerate
            # farther past the source joint's max velocity.  Opposing force and
            # the full contact/grasp position-PD remain active.  This maps the
            # actuator limit without ever writing qpos/qvel.
            if finger_velocity_limits is not None:
                data.qfrc_applied[7:9] = velocity_limited_finger_drive(
                    model, data, finger_force, finger_velocity_limits
                )
            else:
                data.qfrc_applied[7:9] = finger_force
            mujoco.mj_step(model, data)
        if arm_velocity_limits is not None:
            np.clip(
                data.qvel[:7],
                -arm_velocity_limits,
                arm_velocity_limits,
                out=data.qvel[:7],
            )
        if tick_callback is not None:
            tick_callback(
                tick=tick,
                q_before=q_before,
                dq_before=dq_before,
                effort_command=effort_command.copy(),
                q_after=data.qpos[:7].copy(),
                dq_after=data.qvel[:7].copy(),
            )
    return bool(close)


def velocity_limited_finger_drive(model, data, force, limits, iterations=4):
    """Project actuator force so its next-step velocity stays within bounds.

    The projection can reduce the commanded drive but never adds a braking
    impulse or changes simulator state.  Contact can still exceed the bound,
    just as the recorded PhysX fingers occasionally do.
    """
    dofs = np.array([7, 8], dtype=np.int32)
    force = np.asarray(force, dtype=np.float64).copy()
    limits = np.asarray(limits, dtype=np.float64)
    dt = float(model.opt.timestep)

    # Compute acceleration with no finger drive.  Arm torque, bias, friction,
    # contacts, and all other current physics remain present.
    data.qfrc_applied[dofs] = 0.0
    mujoco.mj_forward(model, data)
    acceleration_zero = data.qacc[dofs].copy()

    # Measure the current contact-aware response to each commanded finger
    # force.  A blocked fingertip has a much smaller response than a free one;
    # this preserves full grasp force while still limiting free-space motion.
    response = np.zeros((2, 2), dtype=np.float64)
    for actuator in range(2):
        if abs(force[actuator]) <= 1.0e-12:
            continue
        data.qfrc_applied[dofs] = 0.0
        data.qfrc_applied[dofs[actuator]] = force[actuator]
        mujoco.mj_forward(model, data)
        response[:, actuator] = data.qacc[dofs] - acceleration_zero

    scales = np.ones(2, dtype=np.float64)

    for _ in range(iterations):
        predicted = data.qvel[dofs] + dt * (acceleration_zero + response @ scales)
        changed = False
        for joint in range(2):
            if predicted[joint] > limits[joint]:
                target = limits[joint]
            elif predicted[joint] < -limits[joint]:
                target = -limits[joint]
            else:
                continue
            row = dt * response[joint]
            norm_sq = float(row @ row)
            if norm_sq > 0.0:
                scales += ((target - predicted[joint]) / norm_sq) * row
                scales = np.clip(scales, 0.0, 1.0)
                changed = True
        if not changed:
            break
    return force * scales


def _constant_runtime_array(store, key, start, end):
    values = np.asarray(store[f"data/{key}"])[start:end].astype(np.float64)
    drift = float(np.max(np.abs(values - values[0])))
    if drift > 1e-6:
        raise ValueError(f"Isaac runtime field {key} changed within episode: max drift={drift}")
    return values[0], drift


def load_isaac_runtime(store, start, end):
    """Load what Isaac actually used, instead of trusting intended config."""
    fields = (
        "arm_scale",
        "arm_kp",
        "arm_kd",
        "arm_torque_max",
        "arm_joint_armature",
        "arm_joint_friction_static",
        "arm_joint_friction_dynamic",
        "arm_joint_friction_viscous",
    )
    runtime = {}
    drift = {}
    for key in fields:
        runtime[key], drift[key] = _constant_runtime_array(store, key, start, end)
    runtime["max_within_episode_drift"] = drift
    return runtime


def load_runtime_scene_properties(store, start, end):
    keys = (
        "robot_material_properties",
        "robot_body_masses",
        "robot_body_coms",
        "robot_body_inertias",
        "insertive_object_material_properties",
        "insertive_object_body_masses",
        "insertive_object_body_coms",
        "insertive_object_body_inertias",
        "receptive_object_material_properties",
        "receptive_object_body_masses",
        "table_material_properties",
        "table_body_masses",
    )
    scene = {}
    for key in keys:
        path = f"data/{key}"
        if path not in store:
            raise KeyError(f"runtime scene property missing from source zarr: {path}")
        scene[key], _ = _constant_runtime_array(store, key, start, end)
    scene["robot_body_names"] = list(store.attrs["robot_body_names"])
    return scene


def _set_mujoco_body_inertia_from_physx(model, body, mass, com, inertia):
    """Apply a PhysX body-frame inertia tensor to one MuJoCo body."""
    inertia = np.asarray(inertia, dtype=np.float64).reshape(3, 3, order="F")
    inertia = 0.5 * (inertia + inertia.T)
    principal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0.0:
        axes[:, 0] *= -1.0
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, axes.reshape(-1))
    model.body_mass[body] = float(mass)
    model.body_ipos[body] = np.asarray(com, dtype=np.float64)[:3]
    model.body_inertia[body] = np.maximum(principal, 1.0e-12)
    model.body_iquat[body] = quat


def apply_runtime_body_masses(model, ctrl, scene, include_robot):
    source_names = scene["robot_body_names"]
    source_masses = scene["robot_body_masses"]
    source_coms = scene["robot_body_coms"].reshape(-1, 7)
    source_inertias = scene["robot_body_inertias"].reshape(-1, 9)
    source_index = {name: i for i, name in enumerate(source_names)}
    name_map = {
        **{f"link{i}": f"panda_link{i}" for i in range(8)},
        "hand": "panda_hand",
        "left_finger": "panda_leftfinger",
        "right_finger": "panda_rightfinger",
    }
    applied = {}
    if include_robot:
        for mujoco_name, source_name in name_map.items():
            body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name
            )
            if body < 0 or source_name not in source_index:
                continue
            source_id = source_index[source_name]
            new_mass = float(source_masses[source_id])
            new_com = source_coms[source_id]
            new_inertia = source_inertias[source_id]
            # Isaac's FR3 asset contains a fixed force_sensor rigid body between
            # panda_link7 and panda_hand.  The MuJoCo model has no corresponding
            # body, so merge its spatial inertia into link7.  Both source bodies
            # use the same rigid-body-prim frame and COM location in this asset.
            if mujoco_name == "link7" and "force_sensor" in source_index:
                sensor_id = source_index["force_sensor"]
                sensor_com = source_coms[sensor_id]
                if not np.allclose(new_com[:3], sensor_com[:3], atol=1.0e-7):
                    raise ValueError(
                        "force_sensor and panda_link7 COM positions differ; "
                        "a parallel-axis merge is required"
                    )
                new_mass += float(source_masses[sensor_id])
                new_inertia = new_inertia + source_inertias[sensor_id]
            _set_mujoco_body_inertia_from_physx(
                model, body, new_mass, new_com, new_inertia
            )
            applied[mujoco_name] = new_mass
    peg_mass = float(scene["insertive_object_body_masses"][0])
    _set_mujoco_body_inertia_from_physx(
        model,
        ctrl.peg,
        peg_mass,
        scene["insertive_object_body_coms"].reshape(-1, 7)[0],
        scene["insertive_object_body_inertias"].reshape(-1, 9)[0],
    )
    applied["peg"] = peg_mass
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return applied


def apply_runtime_contact_materials(model, scene):
    """Map recorded PhysX moving-friction coefficients to MuJoCo geoms."""
    robot_materials = scene["robot_material_properties"].reshape(-1, 3)
    source_names = scene["robot_body_names"]
    source_material = {
        name: robot_materials[i]
        for i, name in enumerate(source_names[: len(robot_materials)])
    }
    body_map = {
        **{f"link{i}": f"panda_link{i}" for i in range(8)},
        "hand": "panda_hand",
        "left_finger": "panda_leftfinger",
        "right_finger": "panda_rightfinger",
    }
    applied = {}
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        body = int(model.geom_bodyid[geom])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        source_name = body_map.get(body_name)
        if source_name in source_material:
            coefficient = float(source_material[source_name][1])
            model.geom_friction[geom, 0] = coefficient
            applied[body_name] = coefficient
    named = {
        "peg_geom": scene["insertive_object_material_properties"],
        "hole_col_g": scene["receptive_object_material_properties"],
        "table_top": scene["table_material_properties"][:3],
    }
    for geom_name, material in named.items():
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom >= 0:
            coefficient = float(np.asarray(material).reshape(-1, 3)[0, 1])
            model.geom_friction[geom, 0] = coefficient
            applied[geom_name] = coefficient
    receptive_coefficient = float(
        np.asarray(scene["receptive_object_material_properties"])
        .reshape(-1, 3)[0, 1]
    )
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        body = int(model.geom_bodyid[geom])
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) == "peghole":
            model.geom_friction[geom, 0] = receptive_coefficient
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            applied[geom_name] = receptive_coefficient
    return applied


def audit_recorded_osc(
    model,
    ctrl,
    raw,
    store,
    start,
    limit,
    runtime,
    jacobian_point,
    nullspace_stiffness,
    nullspace_damping_ratio,
):
    """Compare both Jacobian conventions and first-substep torque on recorded states."""
    data = mujoco.MjData(model)
    recorded_jac = np.asarray(store["data/arm_jacobian"])[start:start + limit]
    recorded_pose_error = np.asarray(store["data/arm_pose_error"])[start:start + limit]
    # The first actual controller write in each policy step is authoritative.
    # Older exporter revisions computed arm_joint_torque_first diagnostically
    # and omitted the optional nullspace term; arm_joint_torque_substeps is
    # captured from the action term after it writes the real effort target.
    recorded_tau = np.asarray(store["data/arm_joint_torque_substeps"])[
        start:start + limit, 0
    ]
    origin_jac_l2 = []
    com_jac_l2 = []
    com_tau_l2 = []
    com_tau_max = []
    selected_tau_l2 = []
    selected_tau_max = []
    for step in range(limit):
        set_raw_state(model, data, ctrl, raw[step])
        jac_origin = ctrl.jac_arm(data).copy()
        jac_com = jac_arm_at_isaac_hand_com(model, data, ctrl).copy()
        jac_selected = jac_origin if jacobian_point == "link_origin" else jac_com
        ee_vel = jac_selected @ data.qvel[:7]
        task_force = (
            runtime["arm_kp"] * recorded_pose_error[step]
            + runtime["arm_kd"] * (-ee_vel)
        )
        selected_torque_unclipped = jac_selected.T @ task_force
        if nullspace_stiffness > 0.0:
            nullspace_damping = (
                2.0 * np.sqrt(nullspace_stiffness) * nullspace_damping_ratio
            )
            selected_torque_unclipped += (
                nullspace_stiffness * (B3_NULLSPACE_DEFAULT_POS - data.qpos[:7])
                - nullspace_damping * data.qvel[:7]
            )
        selected_torque = np.clip(
            selected_torque_unclipped,
            -runtime["arm_torque_max"],
            runtime["arm_torque_max"],
        )
        com_ee_vel = jac_com @ data.qvel[:7]
        com_task_force = (
            runtime["arm_kp"] * recorded_pose_error[step]
            + runtime["arm_kd"] * (-com_ee_vel)
        )
        com_torque_unclipped = jac_com.T @ com_task_force
        if nullspace_stiffness > 0.0:
            com_torque_unclipped += (
                nullspace_stiffness * (B3_NULLSPACE_DEFAULT_POS - data.qpos[:7])
                - nullspace_damping * data.qvel[:7]
            )
        com_torque = np.clip(
            com_torque_unclipped,
            -runtime["arm_torque_max"],
            runtime["arm_torque_max"],
        )
        origin_jac_l2.append(float(np.linalg.norm(jac_origin - recorded_jac[step])))
        com_jac_l2.append(float(np.linalg.norm(jac_com - recorded_jac[step])))
        delta_tau = com_torque - recorded_tau[step]
        com_tau_l2.append(float(np.linalg.norm(delta_tau)))
        com_tau_max.append(float(np.max(np.abs(delta_tau))))
        selected_delta_tau = selected_torque - recorded_tau[step]
        selected_tau_l2.append(float(np.linalg.norm(selected_delta_tau)))
        selected_tau_max.append(float(np.max(np.abs(selected_delta_tau))))

    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        return {"mean": float(values.mean()), "max": float(values.max())}

    return {
        "pose_error_point": "panda_hand link origin",
        "selected_jacobian_point": jacobian_point,
        "nullspace_stiffness": float(nullspace_stiffness),
        "nullspace_damping_ratio": float(nullspace_damping_ratio),
        "nullspace_default_pos_rad": B3_NULLSPACE_DEFAULT_POS.copy(),
        "hand_com_local_m": ISAAC_HAND_COM_LOCAL.copy(),
        "origin_jacobian_fro_error": stats(origin_jac_l2),
        "com_jacobian_fro_error": stats(com_jac_l2),
        "com_first_substep_torque_l2_error_nm": stats(com_tau_l2),
        "com_first_substep_torque_max_abs_error_nm": stats(com_tau_max),
        "selected_first_substep_torque_l2_error_nm": stats(selected_tau_l2),
        "selected_first_substep_torque_max_abs_error_nm": stats(selected_tau_max),
    }


def _tick_error_row(
    policy_step,
    tick,
    q,
    dq,
    tau,
    q_ref,
    dq_ref,
    tau_ref,
    q_after=None,
    dq_after=None,
    q_after_ref=None,
    dq_after_ref=None,
):
    q_error = np.asarray(q) - np.asarray(q_ref)
    dq_error = np.asarray(dq) - np.asarray(dq_ref)
    tau_error = np.asarray(tau) - np.asarray(tau_ref)
    row = {
        "policy_step_zero_based": int(policy_step),
        "physics_tick_zero_based": int(tick),
        "global_tick_one_based": int(policy_step * CL.DECIM + tick + 1),
        "q_l2_before_rad": float(np.linalg.norm(q_error)),
        "q_max_before_rad": float(np.max(np.abs(q_error))),
        "dq_l2_before_radps": float(np.linalg.norm(dq_error)),
        "dq_max_before_radps": float(np.max(np.abs(dq_error))),
        "tau_l2_nm": float(np.linalg.norm(tau_error)),
        "tau_max_nm": float(np.max(np.abs(tau_error))),
    }
    for joint in range(7):
        suffix = joint + 1
        row[f"q_error_j{suffix}_rad"] = float(q_error[joint])
        row[f"dq_error_j{suffix}_radps"] = float(dq_error[joint])
        row[f"tau_error_j{suffix}_nm"] = float(tau_error[joint])
    if q_after is not None:
        q_after_error = np.asarray(q_after) - np.asarray(q_after_ref)
        dq_after_error = np.asarray(dq_after) - np.asarray(dq_after_ref)
        row.update(
            {
                "q_l2_after_rad": float(np.linalg.norm(q_after_error)),
                "q_max_after_rad": float(np.max(np.abs(q_after_error))),
                "dq_l2_after_radps": float(np.linalg.norm(dq_after_error)),
                "dq_max_after_radps": float(np.max(np.abs(dq_after_error))),
            }
        )
    return row


def summarize_tick_trace(rows):
    if not rows:
        return {"ticks": 0}
    output = {"ticks": len(rows)}
    thresholds = {
        "q_l2_before_rad": 1.0e-3,
        "dq_l2_before_radps": 0.05,
        "tau_l2_nm": 1.0,
    }
    for key in (
        "q_l2_before_rad",
        "dq_l2_before_radps",
        "tau_l2_nm",
        "q_l2_after_rad",
        "dq_l2_after_radps",
    ):
        if key not in rows[0]:
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        stats = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "argmax_global_tick_one_based": int(values.argmax() + 1),
        }
        if key in thresholds:
            crossing = np.flatnonzero(values > thresholds[key])
            stats["threshold"] = thresholds[key]
            stats["first_over_threshold_global_tick_one_based"] = (
                int(crossing[0] + 1) if len(crossing) else None
            )
        output[key] = stats
    return output


def audit_recorded_osc_ticks(
    model,
    ctrl,
    raw,
    actions,
    q_ticks,
    dq_ticks,
    tau_ticks,
    max_ticks,
    jacobian_point,
    nullspace_stiffness,
    nullspace_damping_ratio,
):
    """Teacher-force source states to audit controller math at physics-tick rate."""
    data = mujoco.MjData(model)
    rows = []
    for policy_step in range(len(actions)):
        if len(rows) >= max_ticks:
            break
        set_raw_state(model, data, ctrl, raw[policy_step])
        data.qpos[:7] = q_ticks[policy_step, 0]
        data.qvel[:7] = dq_ticks[policy_step, 0]
        mujoco.mj_forward(model, data)
        ee_pos, ee_quat = ctrl.ee_root(data)
        scaled = np.asarray(actions[policy_step, :6], dtype=np.float64) * CL.SCALE
        ee_pos_des = ee_pos + scaled[:3]
        ee_quat_des = Q.quat_mul(CL.quat_from_aa(scaled[3:6]), ee_quat)
        for tick in range(CL.DECIM):
            if len(rows) >= max_ticks:
                break
            data.qpos[:7] = q_ticks[policy_step, tick]
            data.qvel[:7] = dq_ticks[policy_step, tick]
            mujoco.mj_forward(model, data)
            ee_pos, ee_quat = ctrl.ee_root(data)
            jac = selected_arm_jacobian(model, data, ctrl, jacobian_point)
            ee_vel = jac @ data.qvel[:7]
            pos_error = ee_pos_des - ee_pos
            quat_error = Q.quat_mul(ee_quat_des, Q.quat_inv(ee_quat))
            pose_error = np.concatenate(
                [pos_error, Q.axis_angle_from_quat(quat_error)]
            )
            task_force = CL.KP * pose_error + CL.KD * (-ee_vel)
            torque = jac.T @ task_force
            if nullspace_stiffness > 0.0:
                nullspace_damping = (
                    2.0
                    * np.sqrt(nullspace_stiffness)
                    * nullspace_damping_ratio
                )
                torque += (
                    nullspace_stiffness * (B3_NULLSPACE_DEFAULT_POS - data.qpos[:7])
                    - nullspace_damping * data.qvel[:7]
                )
            torque = np.clip(torque, -CL.TAU_MAX, CL.TAU_MAX)
            rows.append(
                _tick_error_row(
                    policy_step,
                    tick,
                    q_ticks[policy_step, tick],
                    dq_ticks[policy_step, tick],
                    torque,
                    q_ticks[policy_step, tick],
                    dq_ticks[policy_step, tick],
                    tau_ticks[policy_step, tick],
                )
            )
    return summarize_tick_trace(rows), rows


def measure(data, ctrl, raw_ref, fk, close):
    peg_pos_ref, peg_quat_ref, _, _ = pose_in_robot_root(raw_ref)
    ee_pos_ref, ee_quat_ref = fk.hand_in_root(raw_ref[:9])
    ee_pos, ee_quat = ctrl.ee_root(data)
    peg_pos = data.qpos[ctrl.peg_qpos:ctrl.peg_qpos + 3].copy()
    peg_quat = data.qpos[ctrl.peg_qpos + 3:ctrl.peg_qpos + 7].copy()
    return {
        "joint_pos_l2_rad": float(np.linalg.norm(data.qpos[:7] - raw_ref[:7])),
        "joint_pos_max_rad": float(np.max(np.abs(data.qpos[:7] - raw_ref[:7]))),
        "joint_vel_l2_radps": float(np.linalg.norm(data.qvel[:7] - raw_ref[9:16])),
        "finger_pos_l2_m": float(np.linalg.norm(data.qpos[7:9] - raw_ref[7:9])),
        "finger_left_m": float(data.qpos[7]),
        "finger_right_m": float(data.qpos[8]),
        "finger_ref_left_m": float(raw_ref[7]),
        "finger_ref_right_m": float(raw_ref[8]),
        "ee_pos_m": float(np.linalg.norm(ee_pos - ee_pos_ref)),
        "ee_rot_rad": quat_angle(ee_quat, ee_quat_ref),
        "peg_pos_m": float(np.linalg.norm(peg_pos - peg_pos_ref)),
        "peg_rot_rad": quat_angle(peg_quat, peg_quat_ref),
        "close_command": int(close),
        "robot_contact": int(ctrl.peg_robot_contact(data)),
    }


def roll_pitch_error(quat):
    """UWLab ProgressContext orientation metric: |roll| + |pitch|, yaw ignored."""
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    return float(abs(roll) + abs(pitch))


def assembly_metrics(data, ctrl, hole_pos, hole_quat):
    """Official B3 metric plus the historical square-yaw diagnostic."""
    peg_pos = data.qpos[ctrl.peg_qpos:ctrl.peg_qpos + 3]
    peg_quat = data.qpos[ctrl.peg_qpos + 3:ctrl.peg_qpos + 7]
    peg_in_hole = Q.pose_in_root(hole_pos, hole_quat, peg_pos, peg_quat)
    position_error = float(
        np.linalg.norm(peg_in_hole[:3] - np.array([0.0, 0.0, CL.ASSEMBLED_Z]))
    )
    _, relative_quat = Q.subtract_frame_transforms(
        hole_pos, hole_quat, peg_pos, peg_quat
    )
    official_orientation_error = roll_pitch_error(relative_quat)
    square_yaw_error = CL.square_symmetry_rot_error(hole_quat, peg_quat)
    official_pose = (
        position_error < CL.SUC_POS and official_orientation_error < CL.SUC_ROT
    )
    square_yaw_pose = position_error < CL.SUC_POS and square_yaw_error < CL.SUC_ROT
    return (
        position_error,
        official_orientation_error,
        square_yaw_error,
        bool(official_pose),
        bool(square_yaw_pose),
        peg_in_hole[:3].copy(),
    )


def summarize_task(rows):
    position = np.asarray([row["assembly_pos_m"] for row in rows], dtype=np.float64)
    orientation = np.asarray(
        [row["assembly_roll_pitch_rad"] for row in rows], dtype=np.float64
    )
    square_yaw = np.asarray(
        [row["assembly_square_yaw_rot_rad"] for row in rows], dtype=np.float64
    )
    official = np.asarray([row["official_pose"] for row in rows], dtype=bool)
    square_pose = np.asarray([row["square_yaw_pose"] for row in rows], dtype=bool)
    official_steps = np.flatnonzero(official)
    square_steps = np.flatnonzero(square_pose)
    return {
        "success_thresholds": {
            "position_m_lt": CL.SUC_POS,
            "official_abs_roll_plus_abs_pitch_rad_lt": CL.SUC_ROT,
            "yaw": "ignored by UWLab ProgressContext",
            "stable_release_steps_gte": CL.STABLE_RELEASE_STEPS,
        },
        "final_position_error_m": float(position[-1]),
        "final_official_orientation_error_rad": float(orientation[-1]),
        "final_official_pose": bool(official[-1]),
        "best_position_error_m": float(position.min()),
        "best_official_orientation_error_rad": float(orientation.min()),
        "official_pose_steps": int(official.sum()),
        "first_official_pose_step": (
            int(official_steps[0] + 1) if len(official_steps) else None
        ),
        "square_yaw_diagnostic": {
            "threshold_rad_lt": CL.SUC_ROT,
            "final_error_rad": float(square_yaw[-1]),
            "best_error_rad": float(square_yaw.min()),
            "pose_steps": int(square_pose.sum()),
            "first_pose_step": int(square_steps[0] + 1) if len(square_steps) else None,
        },
        "max_stable_release_steps": int(max(row["stable_release_streak"] for row in rows)),
        "stable_release_success": bool(
            max(row["stable_release_streak"] for row in rows) >= CL.STABLE_RELEASE_STEPS
        ),
        "final_gripper_close": bool(rows[-1]["close_command"]),
        "final_robot_contact": bool(rows[-1]["robot_contact"]),
    }


def summarize_reference_task(raw, limit):
    """Evaluate recorded Isaac states with official B3 and square-yaw metrics."""
    position = []
    orientation = []
    square_yaw = []
    official = []
    square_pose = []
    for state in raw[1:limit + 1]:
        peg_pos, peg_quat, hole_pos, hole_quat = pose_in_robot_root(state)
        peg_in_hole = Q.pose_in_root(hole_pos, hole_quat, peg_pos, peg_quat)
        pos_error = float(
            np.linalg.norm(peg_in_hole[:3] - np.array([0.0, 0.0, CL.ASSEMBLED_Z]))
        )
        _, relative_quat = Q.subtract_frame_transforms(
            hole_pos, hole_quat, peg_pos, peg_quat
        )
        orientation_error = roll_pitch_error(relative_quat)
        square_error = CL.square_symmetry_rot_error(hole_quat, peg_quat)
        position.append(pos_error)
        orientation.append(orientation_error)
        square_yaw.append(square_error)
        official.append(pos_error < CL.SUC_POS and orientation_error < CL.SUC_ROT)
        square_pose.append(pos_error < CL.SUC_POS and square_error < CL.SUC_ROT)
    official_steps = np.flatnonzero(official)
    square_steps = np.flatnonzero(square_pose)
    return {
        "final_position_error_m": position[-1],
        "final_official_orientation_error_rad": orientation[-1],
        "final_official_pose": bool(official[-1]),
        "best_position_error_m": min(position),
        "best_official_orientation_error_rad": min(orientation),
        "official_pose_steps": int(sum(official)),
        "first_official_pose_step": (
            int(official_steps[0] + 1) if len(official_steps) else None
        ),
        "square_yaw_diagnostic": {
            "threshold_rad_lt": CL.SUC_ROT,
            "final_error_rad": square_yaw[-1],
            "best_error_rad": min(square_yaw),
            "pose_steps": int(sum(square_pose)),
            "first_pose_step": int(square_steps[0] + 1) if len(square_steps) else None,
        },
    }


def effective_model_config(model, ctrl):
    """Dump the compiled MuJoCo values, not merely the intended profile."""
    grip = ctrl.grip_act
    peg_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    return {
        "timestep_s": float(model.opt.timestep),
        "solver_iterations": int(model.opt.iterations),
        "noslip_iterations": int(model.opt.noslip_iterations),
        "arm_dof_armature": model.dof_armature[:7].copy(),
        "arm_dof_static_frictionloss": model.dof_frictionloss[:7].copy(),
        "arm_dof_viscous_damping": model.dof_damping[:7].copy(),
        "gripper_actuator_gainprm": model.actuator_gainprm[grip].copy(),
        "gripper_actuator_biasprm": model.actuator_biasprm[grip].copy(),
        "gripper_actuator_forcerange": model.actuator_forcerange[grip].copy(),
        "finger_dof_armature": model.dof_armature[7:9].copy(),
        "finger_dof_static_frictionloss": model.dof_frictionloss[7:9].copy(),
        "finger_dof_viscous_damping": model.dof_damping[7:9].copy(),
        "peg_mass_kg": float(model.body_mass[peg_body]),
    }


def render_pair(model, ref_data, replay_data, renderer, camera, label):
    renderer.update_scene(ref_data, camera=camera)
    left = renderer.render()
    renderer.update_scene(replay_data, camera=camera)
    right = renderer.render()
    gap = np.zeros((left.shape[0], 6, 3), dtype=left.dtype)
    image = np.concatenate([left, gap, right], axis=1)
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, pil.width, 24], fill=(0, 0, 0))
        draw.text((8, 5), label, fill=(255, 255, 255))
        return np.asarray(pil)
    except Exception:
        return image


def first_crossing(values, threshold):
    indices = np.flatnonzero(np.asarray(values) > threshold)
    return int(indices[0] + 1) if len(indices) else None


def apply_usd_finger_inertia(model):
    for name in ("left_finger", "right_finger"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        model.body_mass[body_id] = USD_FINGER_INERTIA["mass"]
        model.body_ipos[body_id] = USD_FINGER_INERTIA["com"]
        model.body_inertia[body_id] = USD_FINGER_INERTIA["diagonal"]
        model.body_iquat[body_id] = USD_FINGER_INERTIA["principal_axes"]
    mujoco.mj_setConst(model, mujoco.MjData(model))


def disable_tendon_gripper(model):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    model.actuator_gainprm[actuator_id] = 0.0
    model.actuator_biasprm[actuator_id] = 0.0
    for eq_id in range(model.neq):
        if model.eq_type[eq_id] == mujoco.mjtEq.mjEQ_JOINT:
            model.eq_active0[eq_id] = 0


def summarize(mode, rows):
    summary = {"mode": mode, "steps": len(rows)}
    thresholds = {
        "joint_pos_l2_rad": 0.01,
        "ee_pos_m": 0.005,
        "ee_rot_rad": np.deg2rad(5.0),
        "peg_pos_m": 0.005,
        "peg_rot_rad": np.deg2rad(5.0),
    }
    for key in (
        "joint_pos_l2_rad",
        "joint_pos_max_rad",
        "joint_vel_l2_radps",
        "finger_pos_l2_m",
        "ee_pos_m",
        "ee_rot_rad",
        "peg_pos_m",
        "peg_rot_rad",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "argmax_step": int(values.argmax() + 1),
        }
        if key in thresholds:
            summary[key]["first_over_threshold_step"] = first_crossing(values, thresholds[key])
            summary[key]["threshold"] = float(thresholds[key])
    return summary


def run_mode(
    mode,
    model,
    ctrl,
    raw,
    actions,
    limit,
    out_dir,
    render,
    width,
    height,
    fps,
    finger_velocity_limits,
    independent_gripper,
    recorded_gripper_close,
    jacobian_point,
    nullspace_stiffness,
    nullspace_damping_ratio,
    physx_static_friction,
    physx_dynamic_friction,
    physx_viscous_friction,
    physx_friction_iterations,
    friction_slip_velocity,
    action_reference_blend,
    bias_compensation_scale,
    effort_scale,
    q_ticks,
    dq_ticks,
    tau_ticks,
    tick_trace_ticks,
):
    data = mujoco.MjData(model)
    ref_data = mujoco.MjData(model)
    fk = R.FrankaFK()
    rows = []
    tick_rows = []
    writer = None
    renderer = None
    camera = None
    if render:
        renderer = mujoco.Renderer(model, height, width)
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_front")
        writer = imageio.get_writer(out_dir / f"{mode}_isaac_left_mujoco_right.mp4", fps=fps, macro_block_size=1)

    if mode == "continuous":
        set_raw_state(model, data, ctrl, raw[0])
        if action_reference_blend > 0.0:
            (
                ctrl.action_reference_pos,
                ctrl.action_reference_quat,
            ) = ctrl.ee_root(data)
    elif action_reference_blend > 0.0:
        raise ValueError(
            "integrated action references are stateful and only support continuous mode"
        )

    _, _, hole_pos, hole_quat = pose_in_robot_root(raw[0])
    ever_closed = False
    stable_release_streak = 0

    try:
        for step in range(limit):
            if mode == "onestep":
                set_raw_state(model, data, ctrl, raw[step])

            tick_callback = None
            if mode == "continuous" and len(tick_rows) < tick_trace_ticks:
                def collect_tick(
                    *, tick, q_before, dq_before, effort_command, q_after, dq_after
                ):
                    if len(tick_rows) >= tick_trace_ticks:
                        return
                    if tick + 1 < CL.DECIM:
                        q_after_ref = q_ticks[step, tick + 1]
                        dq_after_ref = dq_ticks[step, tick + 1]
                    else:
                        q_after_ref = raw[step + 1, :7]
                        dq_after_ref = raw[step + 1, 9:16]
                    tick_rows.append(
                        _tick_error_row(
                            step,
                            tick,
                            q_before,
                            dq_before,
                            effort_command,
                            q_ticks[step, tick],
                            dq_ticks[step, tick],
                            tau_ticks[step, tick],
                            q_after=q_after,
                            dq_after=dq_after,
                            q_after_ref=q_after_ref,
                            dq_after_ref=dq_after_ref,
                        )
                    )

                tick_callback = collect_tick
            close = step_action(
                model,
                data,
                ctrl,
                actions[step],
                finger_velocity_limits,
                independent_gripper,
                gripper_close_override=(
                    bool(recorded_gripper_close[step])
                    if recorded_gripper_close is not None
                    else None
                ),
                jacobian_point=jacobian_point,
                nullspace_stiffness=nullspace_stiffness,
                nullspace_damping_ratio=nullspace_damping_ratio,
                physx_static_friction=physx_static_friction,
                physx_dynamic_friction=physx_dynamic_friction,
                physx_viscous_friction=physx_viscous_friction,
                physx_friction_iterations=physx_friction_iterations,
                friction_slip_velocity=friction_slip_velocity,
                action_reference_blend=action_reference_blend,
                bias_compensation_scale=bias_compensation_scale,
                effort_scale=effort_scale,
                tick_callback=tick_callback,
            )
            ever_closed = ever_closed or close
            metrics = measure(data, ctrl, raw[step + 1], fk, close)
            (
                assembly_pos,
                assembly_roll_pitch,
                assembly_square_yaw,
                official_pose,
                square_yaw_pose,
                peg_in_hole_pos,
            ) = assembly_metrics(data, ctrl, hole_pos, hole_quat)
            if ever_closed and not close and official_pose and not metrics["robot_contact"]:
                stable_release_streak += 1
            else:
                stable_release_streak = 0
            metrics.update({
                "assembly_pos_m": assembly_pos,
                "assembly_roll_pitch_rad": assembly_roll_pitch,
                "assembly_square_yaw_rot_rad": assembly_square_yaw,
                "peg_in_hole_x_m": float(peg_in_hole_pos[0]),
                "peg_in_hole_y_m": float(peg_in_hole_pos[1]),
                "peg_in_hole_z_m": float(peg_in_hole_pos[2]),
                "official_pose": int(official_pose),
                "square_yaw_pose": int(square_yaw_pose),
                "stable_release_streak": stable_release_streak,
            })
            metrics["step"] = step + 1
            rows.append(metrics)

            if writer is not None:
                set_raw_state(model, ref_data, ctrl, raw[step + 1])
                label = (
                    f"LEFT recorded Isaac state (MuJoCo render) | RIGHT MuJoCo {mode} | "
                    f"step={step + 1}/{limit} "
                    f"joint={metrics['joint_pos_l2_rad']:.3f}rad "
                    f"ee={metrics['ee_pos_m'] * 1000:.1f}mm "
                    f"peg={metrics['peg_pos_m'] * 1000:.1f}mm "
                    f"assembly={metrics['assembly_pos_m'] * 1000:.1f}mm/"
                    f"rp={metrics['assembly_roll_pitch_rad']:.3f}rad/"
                    f"yawdiag={metrics['assembly_square_yaw_rot_rad']:.3f}rad"
                )
                writer.append_data(render_pair(model, ref_data, data, renderer, camera, label))
    finally:
        if writer is not None:
            writer.close()

    csv_path = out_dir / f"{mode}_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    if tick_rows:
        tick_csv_path = out_dir / f"{mode}_physics_tick_trace.csv"
        with tick_csv_path.open("w", newline="", encoding="utf-8") as file:
            csv_writer = csv.DictWriter(file, fieldnames=list(tick_rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(tick_rows)
    summary = summarize(mode, rows)
    summary["task_outcome"] = summarize_task(rows)
    summary["physics_tick_trace"] = summarize_tick_trace(tick_rows)
    return summary


def run(args):
    if not 0.0 <= args.action_reference_blend <= 1.0:
        raise ValueError("action_reference_blend must be in [0, 1]")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.open(args.zarr, mode="r")
    source_physics_dt = float(store.attrs.get("physics_dt_s", 1.0 / 120.0))
    source_decimation = int(store.attrs.get("decimation", 12))
    source_policy_dt = float(
        store.attrs.get("policy_dt_s", source_physics_dt * source_decimation)
    )
    if abs(source_physics_dt * source_decimation - source_policy_dt) > 1e-9:
        raise ValueError(
            "source timing is inconsistent: "
            f"physics_dt={source_physics_dt}, decimation={source_decimation}, "
            f"policy_dt={source_policy_dt}"
        )
    CL.SIM_DT = source_physics_dt
    CL.DECIM = source_decimation
    all_raw = np.asarray(store["data/raw_state"])
    all_actions = np.asarray(store["data/action"])
    ends = np.asarray(store["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    start, end = int(starts[args.episode]), int(ends[args.episode])
    if args.start_step < 0 or start + args.start_step >= end - 1:
        raise ValueError("start_step must leave at least one action and successor state")
    start += args.start_step
    raw = all_raw[start:end]
    actions = all_actions[start:end]
    limit = min(len(raw) - 1, args.limit_steps if args.limit_steps > 0 else len(raw) - 1)
    recorded_gripper_close = None
    if args.recorded_gripper_command:
        processed_key = "data/gripper_processed_action"
        if processed_key not in store:
            raise KeyError(
                f"--recorded_gripper_command requires {processed_key} in the source zarr"
            )
        processed_gripper = np.asarray(store[processed_key])[start:start + limit]
        if processed_gripper.ndim != 2 or processed_gripper.shape[1] != 2:
            raise ValueError(
                f"expected {processed_key} shape [T, 2], got {processed_gripper.shape}"
            )
        # Isaac records the actuator position command actually applied at each
        # policy step: [0, 0] is closed and [0.04, 0.04] is open.
        recorded_gripper_close = processed_gripper[:, 0] < 0.02
    q_ticks = np.asarray(store["data/arm_joint_pos_substeps"])[start:start + limit]
    dq_ticks = np.asarray(store["data/arm_joint_vel_substeps"])[start:start + limit]
    tau_ticks = np.asarray(store["data/arm_joint_torque_substeps"])[start:start + limit]
    if q_ticks.shape[1] != source_decimation:
        raise ValueError(
            f"source tick telemetry has {q_ticks.shape[1]} ticks per action, "
            f"metadata declares {source_decimation}"
        )

    runtime = load_isaac_runtime(store, start, end)
    runtime_scene = (
        load_runtime_scene_properties(store, start, end)
        if args.runtime_scene_masses or args.runtime_peg_mass
        else None
    )
    profile = b3_center_profile(
        args.physics_substeps,
        args.hole_collision,
        args.gripper_contact_friction,
    )
    # Runtime telemetry is authoritative.  In particular, this prevents a
    # nominal YAML/profile value from silently drifting away from the rollout.
    source_zeta = runtime["arm_kd"] / (2.0 * np.sqrt(runtime["arm_kp"]))
    zeta = (
        np.full(6, args.damping_ratio, dtype=np.float64)
        if args.damping_ratio is not None
        else source_zeta
    )
    effective_kp = runtime["arm_kp"] * args.kp_scale
    CL.set_controller_gains(runtime["arm_scale"], effective_kp, zeta)
    CL.TAU_MAX = runtime["arm_torque_max"].copy()
    _, _, hole_pos, hole_quat = pose_in_robot_root(raw[0])
    model = CL.build_model(hole_pos, hole_quat, profile)
    model.opt.integrator = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }[args.integrator]
    ctrl = CL.Controller(model)
    ctrl.peg_qpos = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg")]
    if args.armature_values is not None:
        model.dof_armature[:7] = np.asarray(args.armature_values, dtype=np.float64)
    elif args.armature_mode == "runtime":
        model.dof_armature[:7] = runtime["arm_joint_armature"] * args.armature_scale
    else:
        model.dof_armature[:7] = 0.0
    if args.physx_friction_impulse:
        if args.viscous_values is not None:
            raise ValueError(
                "--physx_friction_impulse uses recorded B3 viscous friction and "
                "cannot be combined with --viscous_values"
            )
        model.dof_damping[:7] = 0.0
    elif args.viscous_values is not None:
        model.dof_damping[:7] = np.asarray(args.viscous_values, dtype=np.float64)
    else:
        model.dof_damping[:7] = runtime["arm_joint_friction_viscous"] * args.viscous_scale
    if args.real_finger_inertia and not profile["real_finger_inertia"]:
        apply_usd_finger_inertia(model)
    runtime_masses_applied = (
        apply_runtime_body_masses(
            model, ctrl, runtime_scene, include_robot=args.runtime_scene_masses
        )
        if runtime_scene is not None
        else None
    )
    runtime_contact_materials_applied = (
        apply_runtime_contact_materials(model, runtime_scene)
        if args.runtime_contact_materials
        else None
    )
    if args.independent_gripper:
        disable_tendon_gripper(model)
    if args.peg_contact_timeconst is not None:
        model.geom_solref[ctrl.peg_geom, :2] = [args.peg_contact_timeconst, 1.0]
    if args.physx_friction_transition or args.physx_friction_impulse:
        if args.friction_values is not None:
            raise ValueError(
                "the PhysX friction mappings use recorded B3 Ts/Td and cannot "
                "be combined with --friction_values"
            )
        if args.physx_friction_transition:
            model.dof_frictionloss[:7] = (
                runtime["arm_joint_friction_static"] * args.friction_scale
            )
        else:
            model.dof_frictionloss[:7] = 0.0
    elif args.friction_values is not None:
        model.dof_frictionloss[:7] = np.asarray(args.friction_values, dtype=np.float64)
    elif args.arm_friction_mode == "dynamic":
        model.dof_frictionloss[:7] = (
            runtime["arm_joint_friction_dynamic"] * args.friction_scale
        )
    elif args.arm_friction_mode == "off":
        model.dof_frictionloss[:7] = 0.0
    else:
        model.dof_frictionloss[:7] = (
            runtime["arm_joint_friction_static"] * args.friction_scale
        )

    osc_audit = audit_recorded_osc(
        model,
        ctrl,
        raw,
        store,
        start,
        limit,
        runtime,
        args.jacobian_point,
        args.nullspace_stiffness,
        args.nullspace_damping_ratio,
    )
    print("[peginsert-replay] recorded OSC audit")
    print(json.dumps(CL.to_jsonable(osc_audit), indent=2))
    tick_osc_audit, tick_osc_rows = audit_recorded_osc_ticks(
        model,
        ctrl,
        raw,
        actions[:limit],
        q_ticks,
        dq_ticks,
        tau_ticks,
        args.tick_trace_ticks,
        args.jacobian_point,
        args.nullspace_stiffness,
        args.nullspace_damping_ratio,
    )
    if tick_osc_rows:
        tick_osc_path = out_dir / "teacher_forced_osc_physics_tick_audit.csv"
        with tick_osc_path.open("w", newline="", encoding="utf-8") as file:
            csv_writer = csv.DictWriter(file, fieldnames=list(tick_osc_rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(tick_osc_rows)
    print("[peginsert-replay] teacher-forced physics-tick OSC audit")
    print(json.dumps(CL.to_jsonable(tick_osc_audit), indent=2))

    modes = ("onestep", "continuous") if args.mode == "both" else (args.mode,)
    if args.finger_velocity_limit is not None:
        finger_velocity_limits = np.full(2, args.finger_velocity_limit, dtype=np.float64)
    elif profile["finger_velocity_limit"] is not None and not args.clip_finger_velocity:
        finger_velocity_limits = np.full(2, profile["finger_velocity_limit"], dtype=np.float64)
    else:
        finger_velocity_limits = USD_FINGER_VEL_MAX if args.clip_finger_velocity else None
    summaries = {}
    effort_scale = (
        np.asarray(args.effort_scale_values, dtype=np.float64)
        if args.effort_scale_values is not None
        else np.full(7, args.effort_scale, dtype=np.float64)
    )
    if args.physx_friction_transition:
        physx_static_friction = (
            runtime["arm_joint_friction_static"] * args.friction_scale
        )
        physx_dynamic_friction = (
            runtime["arm_joint_friction_dynamic"] * args.friction_scale
        )
        physx_viscous_friction = None
    elif args.physx_friction_impulse:
        physx_static_friction = None
        physx_dynamic_friction = (
            runtime["arm_joint_friction_static"] * args.friction_scale,
            runtime["arm_joint_friction_dynamic"] * args.friction_scale,
        )
        physx_viscous_friction = (
            runtime["arm_joint_friction_viscous"] * args.viscous_scale
        )
    else:
        physx_static_friction = None
        physx_dynamic_friction = None
        physx_viscous_friction = None
    for mode in modes:
        print(f"[peginsert-replay] mode={mode} episode={args.episode} steps={limit}")
        summaries[mode] = run_mode(
            mode,
            model,
            ctrl,
            raw,
            actions,
            limit,
            out_dir,
            args.render,
            args.width,
            args.height,
            args.fps,
            finger_velocity_limits,
            args.independent_gripper,
            recorded_gripper_close,
            args.jacobian_point,
            args.nullspace_stiffness,
            args.nullspace_damping_ratio,
            physx_static_friction,
            physx_dynamic_friction,
            physx_viscous_friction,
            args.physx_friction_iterations,
            args.friction_slip_velocity,
            args.action_reference_blend,
            args.bias_compensation_scale,
            effort_scale,
            q_ticks,
            dq_ticks,
            tau_ticks,
            args.tick_trace_ticks,
        )
        print(json.dumps(summaries[mode], indent=2))

    result = {
        "definition": "fixed B3 checkpoint rollout actions replayed directly in MuJoCo; no policy inference or training",
        "zarr": str(Path(args.zarr).resolve()),
        "episode": args.episode,
        "source_indices": [start, end],
        "steps_compared": limit,
        "source_physics_dt_s": source_physics_dt,
        "source_decimation": source_decimation,
        "source_policy_dt_s": source_policy_dt,
        "zarr_signature": CL.zarr_signature(store),
        "source_progress_success_steps": int(np.asarray(store["data/success"])[start:start + limit].sum()),
        "source_reference_task_outcome": summarize_reference_task(raw, limit),
        "isaac_runtime": CL.to_jsonable(runtime),
        "runtime_scene_properties": CL.to_jsonable(runtime_scene),
        "runtime_masses_applied_kg": runtime_masses_applied,
        "runtime_contact_materials_applied": runtime_contact_materials_applied,
        "recorded_gripper_command": bool(args.recorded_gripper_command),
        "gripper_command_semantics": (
            "open-loop replay of Isaac gripper_processed_action actuator position command"
            if args.recorded_gripper_command
            else "target-side live privileged grasp guard"
        ),
        "recorded_osc_audit": CL.to_jsonable(osc_audit),
        "teacher_forced_physics_tick_osc_audit": CL.to_jsonable(tick_osc_audit),
        "profile": CL.to_jsonable(profile),
        "effective_compiled_mujoco": CL.to_jsonable(effective_model_config(model, ctrl)),
        "peg_contact_timeconst_override_s": args.peg_contact_timeconst,
        "arm_friction_mode": args.arm_friction_mode,
        "physx_friction_transition": args.physx_friction_transition,
        "physx_friction_impulse": args.physx_friction_impulse,
        "physx_friction_iterations": args.physx_friction_iterations,
        "friction_slip_velocity_radps": args.friction_slip_velocity,
        "action_reference_blend": args.action_reference_blend,
        "bias_compensation_scale": args.bias_compensation_scale,
        "kp_scale": args.kp_scale,
        "damping_ratio_override": args.damping_ratio,
        "effort_scale": CL.to_jsonable(effort_scale),
        "armature_mode": args.armature_mode,
        "armature_scale": args.armature_scale,
        "friction_scale": args.friction_scale,
        "viscous_scale": args.viscous_scale,
        "armature_values_override": args.armature_values,
        "friction_values_override": args.friction_values,
        "viscous_values_override": args.viscous_values,
        "integrator": args.integrator,
        "real_finger_inertia": bool(profile["real_finger_inertia"] or args.real_finger_inertia),
        "usd_finger_inertia": CL.to_jsonable(USD_FINGER_INERTIA)
        if profile["real_finger_inertia"] or args.real_finger_inertia
        else None,
        "finger_velocity_limits_mps": CL.to_jsonable(finger_velocity_limits),
        "finger_velocity_limit_enforcement": (
            "projected actuator drive force; no qpos/qvel writes"
            if finger_velocity_limits is not None
            else "not applied"
        ),
        "independent_gripper_pd": args.independent_gripper,
        "robot_root_frame_conversion": True,
        "jacobian_point": args.jacobian_point,
        "nullspace_stiffness": args.nullspace_stiffness,
        "nullspace_damping_ratio": args.nullspace_damping_ratio,
        "nullspace_default_pos_rad": CL.to_jsonable(B3_NULLSPACE_DEFAULT_POS),
        "isaac_hand_com_jacobian_reproduced": args.jacobian_point == "physx_com",
        "initial_joint_and_object_velocities_applied": True,
        "summaries": summaries,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    print(f"[peginsert-replay] summary -> {out_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="datasets/franka_b3_model11200_valset.zarr")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--mode", choices=("onestep", "continuous", "both"), default="both")
    parser.add_argument("--limit_steps", type=int, default=-1)
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="diagnostic suffix start within the source episode; state is still restored once",
    )
    parser.add_argument("--out", default="log/active/peg_b3_model11200_direct_replay_20260714")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--peg_contact_timeconst",
        type=float,
        default=None,
        help="single-variable override for general peg contacts; profile default is 0.02 s",
    )
    parser.add_argument(
        "--arm_friction_mode",
        choices=("static", "dynamic", "off"),
        default="static",
        help="mapping of PhysX joint friction to MuJoCo frictionloss",
    )
    parser.add_argument(
        "--physx_friction_transition",
        action="store_true",
        help=(
            "map recorded PhysX static/dynamic friction: MuJoCo frictionloss "
            "provides Ts while sliding effort is reduced to Td"
        ),
    )
    parser.add_argument(
        "--physx_friction_impulse",
        action="store_true",
        help=(
            "apply the recorded PhysX Ts/Td/viscous values through a per-axis "
            "accumulated impulse solve"
        ),
    )
    parser.add_argument("--physx_friction_iterations", type=int, default=4)
    parser.add_argument("--action_reference_blend", type=float, default=0.0)
    parser.add_argument("--bias_compensation_scale", type=float, default=0.0)
    parser.add_argument("--kp_scale", type=float, default=1.0)
    parser.add_argument("--damping_ratio", type=float, default=None)
    parser.add_argument("--effort_scale", type=float, default=1.0)
    parser.add_argument("--effort_scale_values", type=float, nargs=7, default=None)
    parser.add_argument("--friction_slip_velocity", type=float, default=1.0e-4)
    parser.add_argument(
        "--armature_mode",
        choices=("runtime", "off"),
        default="runtime",
        help="runtime maps the Isaac property directly; off tests PhysX mass-matrix parity.",
    )
    parser.add_argument(
        "--armature_scale",
        type=float,
        default=1.0,
        help="MuJoCo engine-mapping multiplier on the recorded Isaac armature.",
    )
    parser.add_argument(
        "--friction_scale",
        type=float,
        default=1.0,
        help="MuJoCo engine-mapping multiplier on the selected Coulomb friction.",
    )
    parser.add_argument(
        "--viscous_scale",
        type=float,
        default=1.0,
        help="MuJoCo engine-mapping multiplier on recorded viscous friction.",
    )
    parser.add_argument(
        "--armature_values",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="explicit per-joint MuJoCo armature override for controlled sysid tests",
    )
    parser.add_argument(
        "--friction_values",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="explicit per-joint MuJoCo frictionloss override for controlled sysid tests",
    )
    parser.add_argument(
        "--viscous_values",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="explicit per-joint MuJoCo viscous damping override for controlled sysid tests",
    )
    parser.add_argument(
        "--integrator",
        choices=("euler", "rk4", "implicit", "implicitfast"),
        default="implicitfast",
    )
    parser.add_argument("--real_finger_inertia", action="store_true")
    parser.add_argument("--runtime_scene_masses", action="store_true")
    parser.add_argument("--runtime_peg_mass", action="store_true")
    parser.add_argument(
        "--hole_collision",
        choices=("mesh_sdf_big", "box_ring_big", "mesh_sdf", "box_ring"),
        default=None,
        help="explicit source-geometry collision representation for controlled engine-mapping tests",
    )
    parser.add_argument(
        "--runtime_contact_materials",
        action="store_true",
        help="map recorded PhysX dynamic material friction to MuJoCo contact geoms",
    )
    parser.add_argument("--clip_finger_velocity", action="store_true")
    parser.add_argument("--finger_velocity_limit", type=float, default=None)
    parser.add_argument("--independent_gripper", action="store_true")
    parser.add_argument(
        "--gripper_contact_friction",
        type=float,
        nargs=5,
        default=None,
        metavar=("SLIDE1", "SLIDE2", "TORSION", "ROLL1", "ROLL2"),
        help="explicit MuJoCo peg-fingertip pair friction for contact-model alignment",
    )
    parser.add_argument(
        "--recorded_gripper_command",
        action="store_true",
        help=(
            "replay Isaac's recorded gripper_processed_action open-loop instead of "
            "recomputing the privileged grasp guard from the diverged MuJoCo state"
        ),
    )
    parser.add_argument(
        "--jacobian_point",
        choices=("physx_com", "link_origin"),
        default="physx_com",
        help="must match the Isaac controller that generated the source trajectory",
    )
    parser.add_argument(
        "--nullspace_stiffness",
        type=float,
        default=0.0,
        help="joint-space posture stiffness; must match the Isaac source controller",
    )
    parser.add_argument(
        "--nullspace_damping_ratio",
        type=float,
        default=1.0,
        help="critical-damping ratio used by both source and replay controller",
    )
    parser.add_argument(
        "--physics_substeps",
        type=int,
        default=None,
        help="MuJoCo integration substeps per Isaac 1/120-s tick; profile default is 16.",
    )
    parser.add_argument(
        "--tick_trace_ticks",
        type=int,
        default=36,
        help="number of initial source physics ticks to audit without state injection",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
