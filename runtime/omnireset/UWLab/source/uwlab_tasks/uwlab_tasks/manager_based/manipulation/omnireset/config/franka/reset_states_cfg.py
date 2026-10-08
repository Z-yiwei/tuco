# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda reset-state recording configurations."""

from __future__ import annotations

import os
import numpy as np
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR, resolve_cloud_path
from uwlab_tasks.manager_based.manipulation.factory_extension.factory_assets_cfg import FRANKA_PANDA_CFG

# Local mirror for CupCake/Plate (and optional cubes) to avoid flaky HF fetches
# during bulk reset collection. Override via UWLAB_LOCAL_ASSETS_DIR.
LOCAL_ASSETS_DIR = os.environ.get(
    "UWLAB_LOCAL_ASSETS_DIR",
    os.path.abspath(os.path.join(os.getcwd(), "Datasets/local_assets")),
)

from .actions import FRANKA_FR3_RELATIVE_OSC

from ... import mdp as task_mdp


_OBJECT_GREY = (142.0 / 255.0, 144.0 / 255.0, 137.0 / 255.0)  # #8E9089


@configclass
class FrankaResetStatesSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Franka reset states recording."""

    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0005),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_OBJECT_GREY),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole_big.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_OBJECT_GREY),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=10000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---------------------------------------------------------------------------
# Event configs for each reset type
# ---------------------------------------------------------------------------

@configclass
class FrankaResetStatesBaseEventCfg:
    """Base event config shared by all Franka reset state types."""

    reset_robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.5, 0.5),
            "dynamic_friction_range": (0.5, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (2.0, 2.0),
            "dynamic_friction_range": (2.0, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    reset_robot_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfgs": {"robot": SceneEntityCfg("robot")},
        },
    )

    reset_receptive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.1, 0.3),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-np.pi / 12, np.pi / 12),
            },
            "velocity_range": {},
            "asset_cfgs": {"receptive_object": SceneEntityCfg("receptive_object")},
        },
    )


@configclass
class FrankaObjectAnywhereEEAnywhereEventCfg(FrankaResetStatesBaseEventCfg):
    """Object placed randomly, EE placed randomly via IK."""

    reset_insertive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.3),
                "roll": (-np.pi, np.pi),
                "pitch": (-np.pi, np.pi),
                "yaw": (-np.pi, np.pi),
            },
            "velocity_range": {},
            "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.65),
                "y": (-0.3, 0.3),
                "z": (0.1, 0.5),
                # Franka panda_hand: Z+ local ≈ -Z world at home.
                # Keep pitch near π (fingers down), small roll/yaw variation.
                "roll": (-np.pi / 6, np.pi / 6),
                "pitch": (2 * np.pi / 3, 4 * np.pi / 3),
                "yaw": (-np.pi, np.pi),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["panda_joint.*"], body_names="panda_hand"
            ),
        },
    )


@configclass
class FrankaObjectRestingEEGraspedEventCfg(FrankaResetStatesBaseEventCfg):
    """Object resting on table (upright) + EE in grasp pose from grasps.pt.

    Diverges from UR5e original: UR5e loads Step 1 states (ObjectAnywhereEEAnywhere)
    via MultiResetManager. For Franka that yielded 98% lying-peg input → grasp ⊕
    object_world produces a *horizontal* gripper, which then collides with the
    ground plane during physics settle (collision_free analyzers don't include
    ground). We sidestep the issue by spawning peg upright on table here, which
    is also semantically what "Resting" should mean.
    """

    reset_insertive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.2, 0.2),
                # Peg in *air* near the table (not on table), so gravity acts
                # as a filter: if grasp fails, peg falls > 1cm → max_pos_dev
                # rejects.  Peg-on-table (no gravity filter) was Step 2 v4's
                # failure mode — measure_grasp_distance.py showed only 3% of
                # recorded states actually had peg in gripper.
                "z": (0.05, 0.06),
                "roll": (-np.pi / 18, np.pi / 18),
                "pitch": (-np.pi / 18, np.pi / 18),
                "yaw": (-np.pi, np.pi),
            },
            "velocity_range": {},
            "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
        },
    )

    # Reset robot joints to home before IK to ensure top-down convergence.
    reset_robot_joints_to_home = EventTerm(
        func=task_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["panda_joint.*"], body_names="panda_hand"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            "pose_range_b": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
                "roll": (-np.pi / 16, np.pi / 16),
                "pitch": (-np.pi / 16, np.pi / 16),
                "yaw": (-np.pi / 16, np.pi / 16),
            },
            # Snap peg to gripper's post-IK pose so the recorded grasp-relative
            # pose holds regardless of IK error / peg drift. Without this the
            # recorder was capturing fingers-closed-on-empty-space (only ~3%
            # of states had peg actually in the gripper, per
            # measure_grasp_distance.py).
            "snap_object_after_ik": True,
            "write_gripper_state": True,
        },
    )


@configclass
class FrankaObjectAnywhereEEGraspedEventCfg(FrankaResetStatesBaseEventCfg):
    """Object placed randomly, EE in grasp pose from grasps.pt."""

    reset_insertive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.3),
                "roll": (-np.pi, np.pi),
                "pitch": (-np.pi, np.pi),
                "yaw": (-np.pi, np.pi),
            },
            "velocity_range": {},
            "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
        },
    )

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["panda_joint.*"], body_names="panda_hand"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "snap_object_after_ik": True,
            # Use dataset's recorded ~17mm finger pose (sized to the peg).
            # Combined with close_command_expr=0.015 in the action cfg (just
            # under the dataset's 17mm) the gripper holds the peg without
            # squeezing it out laterally.
            "write_gripper_state": True,
        },
    )


@configclass
class FrankaObjectPartiallyAssembledEEAnywhereEventCfg(FrankaResetStatesBaseEventCfg):
    """Insertive object partially inserted into receptive, EE placed randomly via IK."""

    reset_insertive_object_pose_from_partial_assembly_dataset = EventTerm(
        func=task_mdp.reset_insertive_object_from_partial_assembly_dataset,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.65),
                "y": (-0.3, 0.3),
                "z": (0.3, 0.5),
                # Franka panda_hand convention: pitch near π = fingers down.
                "roll": (-np.pi / 6, np.pi / 6),
                "pitch": (2 * np.pi / 3, 4 * np.pi / 3),
                "yaw": (-np.pi, np.pi),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["panda_joint.*"], body_names="panda_hand"
            ),
        },
    )


@configclass
class FrankaObjectPartiallyAssembledEEGraspedEventCfg(FrankaResetStatesBaseEventCfg):
    """Insertive object partially inserted into receptive, EE in grasp pose from grasps.pt."""

    reset_insertive_object_pose_from_partial_assembly_dataset = EventTerm(
        func=task_mdp.reset_insertive_object_from_partial_assembly_dataset,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["panda_joint.*"], body_names="panda_hand"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            "pose_range_b": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-np.pi / 32, np.pi / 32),
                "pitch": (-np.pi / 32, np.pi / 32),
                "yaw": (-np.pi / 32, np.pi / 32),
            },
            # Snap peg to gripper's post-IK pose; otherwise gripper would
            # close on empty space while peg is held in place only by the
            # partial-assembly fixture (peg in hole), masking the grasp bug.
            "snap_object_after_ik": True,
            "write_gripper_state": True,
        },
    )


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

@configclass
class FrankaResetStatesTerminationCfg:
    """Termination config for Franka reset state recording."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    success = DoneTerm(
        func=task_mdp.check_reset_state_success,
        params={
            "object_cfgs": [SceneEntityCfg("insertive_object"), SceneEntityCfg("receptive_object")],
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_body_name": "panda_hand",
            "collision_analyzer_cfgs": [
                # Robot vs insertive object — relaxed for panda palm geometry
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=-0.005,
                    asset_cfg=SceneEntityCfg("robot", body_names=["panda_hand"]),
                    obstacle_cfgs=[SceneEntityCfg("insertive_object")],
                ),
                # Robot vs receptive object — strict (should not collide)
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=0.0,
                    asset_cfg=SceneEntityCfg("robot"),
                    obstacle_cfgs=[SceneEntityCfg("receptive_object")],
                ),
                # Insertive vs receptive — slight tolerance
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=-0.001,
                    asset_cfg=SceneEntityCfg("insertive_object"),
                    obstacle_cfgs=[SceneEntityCfg("receptive_object")],
                ),
            ],
            "max_robot_pos_deviation": 0.1,
            "max_object_pos_deviation": MISSING,
            "pos_z_threshold": -0.02,
            # Stability: gripper damping fix (factory_assets_cfg.py:95) removed
            # the finger oscillation that previously forced ang_vel_threshold=10.
            # Now back to default-ish values, just slightly tighter than the
            # IsaacLab defaults to filter wobbly grasps.
            "consecutive_stability_steps": 10,
            "stability_lin_vel_threshold": 0.05,
            "stability_ang_vel_threshold": 1.0,
        },
        time_out=True,
    )


@configclass
class FrankaResetStatesObservationsCfg:
    pass


@configclass
class FrankaResetStatesRewardsCfg:
    pass


# ---------------------------------------------------------------------------
# Object variant factories (same as UR5)
# ---------------------------------------------------------------------------

def make_insertive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_cloud_path(usd_path),
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0005),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_OBJECT_GREY),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_cloud_path(usd_path),
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_OBJECT_GREY),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.insertive_object": {
        "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
        "cupcake": make_insertive_object(f"{LOCAL_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cupcake_half": make_insertive_object(f"{LOCAL_ASSETS_DIR}/Props/Custom/CupCakeHalf/cupcake.usd"),
    },
    "scene.receptive_object": {
        "peghole": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole_big.usd"),
        "peghole_big": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole_big.usd"),
        "plate": make_receptive_object(f"{LOCAL_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
        "cube": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/ReceptiveCube/receptive_cube.usd"),
        "wall": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"),
    },
}

variants["scene.insertive_object"]["cupcake_half"].spawn.scale = (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Action config for reset state recording (OSC + binary gripper)
# ---------------------------------------------------------------------------

@configclass
class FrankaResetStatesAction:
    body = FRANKA_FR3_RELATIVE_OSC
    gripper = task_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        # close target just under the grasp dataset's ~17mm finger pose, so
        # the implicit-PD actuator gently tightens onto the peg instead of
        # squeezing it out (closing to 0 was the original failure mode for
        # ObjectAnywhereEEGrasped — Step 3 had ~36% peg-outside-gripper).
        close_command_expr={"panda_finger_.*": 0.015},
    )


# ---------------------------------------------------------------------------
# Top-level env configs per reset type
# ---------------------------------------------------------------------------

@configclass
class FrankaPandaResetStatesCfg(ManagerBasedRLEnvCfg):
    """Base reset states config for Franka Panda."""

    scene: FrankaResetStatesSceneCfg = FrankaResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    events: FrankaResetStatesBaseEventCfg = MISSING
    terminations: FrankaResetStatesTerminationCfg = FrankaResetStatesTerminationCfg()
    observations: FrankaResetStatesObservationsCfg = FrankaResetStatesObservationsCfg()
    actions: FrankaResetStatesAction = FrankaResetStatesAction()
    rewards: FrankaResetStatesRewardsCfg = FrankaResetStatesRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 3.0
        self.sim.dt = 1 / 120.0

        # Arm joint stiffness to hold position (prevents drooping/shaking)
        self.scene.robot.actuators["panda_arm1"].stiffness = 1000.0
        self.scene.robot.actuators["panda_arm1"].damping = 100.0
        self.scene.robot.actuators["panda_arm2"].stiffness = 1000.0
        self.scene.robot.actuators["panda_arm2"].damping = 100.0
        # Fast gripper closing with near-critical damping (no oscillation)
        self.scene.robot.actuators["panda_hand"].stiffness = 5000.0
        self.scene.robot.actuators["panda_hand"].damping = 50.0
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 400.0

        # High friction for grasp stability
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
            friction_combine_mode="max",
        )

        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31


@configclass
class FrankaObjectAnywhereEEAnywhereResetStatesCfg(FrankaPandaResetStatesCfg):
    events: FrankaObjectAnywhereEEAnywhereEventCfg = FrankaObjectAnywhereEEAnywhereEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = np.inf


@configclass
class FrankaObjectRestingEEGraspedResetStatesCfg(FrankaPandaResetStatesCfg):
    events: FrankaObjectRestingEEGraspedEventCfg = FrankaObjectRestingEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # max_pos_dev=0.01: tightening to 0.005 (v6) didn't improve quality
        # (53% vs 57% in_gripper); IK precision is the bottleneck, not the
        # position threshold. Sticking with 0.01 keeps recording feasible
        # (~74 min vs 2-3 hr) and produces 100 states.
        self.terminations.success.params["max_object_pos_deviation"] = 0.01
        self.episode_length_s = 5.0


@configclass
class FrankaObjectAnywhereEEGraspedResetStatesCfg(FrankaPandaResetStatesCfg):
    events: FrankaObjectAnywhereEEGraspedEventCfg = FrankaObjectAnywhereEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.05


@configclass
class FrankaObjectPartiallyAssembledEEAnywhereResetStatesCfg(FrankaPandaResetStatesCfg):
    events: FrankaObjectPartiallyAssembledEEAnywhereEventCfg = FrankaObjectPartiallyAssembledEEAnywhereEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.025
        self.terminations.success.params["insertive_asset_cfg"] = SceneEntityCfg("insertive_object")
        self.terminations.success.params["receptive_asset_cfg"] = SceneEntityCfg("receptive_object")
        self.terminations.success.params["assembly_success_prob"] = 0.5
        self.terminations.success.params["assembly_threshold_scale"] = 1.5


from .cupcake_reset_spec import CUPCAKE_POSE_RANGE, PLATE_POSE_RANGE


@configclass
class FrankaCupCakeSideLyingFront3cmEventCfg(FrankaObjectAnywhereEEAnywhereEventCfg):
    """CupCake side-lying near the robot, with the plate farther forward."""

    # FixedHome alignment: the robot is reset to its canonical root/joint
    # state.  The joint targets must be synchronized as well; otherwise the
    # implicit-PD controller immediately pulls the robot toward stale targets.
    reset_end_effector_pose = None

    reset_robot_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfgs": {"robot": SceneEntityCfg("robot")},
        },
    )

    reset_robot_joints_to_home = EventTerm(
        func=task_mdp.reset_joints_to_default_and_hold,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )

    reset_insertive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": CUPCAKE_POSE_RANGE,
            "velocity_range": {},
            "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
        },
    )

    reset_receptive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": PLATE_POSE_RANGE,
            "velocity_range": {},
            "asset_cfgs": {"receptive_object": SceneEntityCfg("receptive_object")},
        },
    )


@configclass
class FrankaCupCakeSideLyingFront3cmResetStatesCfg(FrankaPandaResetStatesCfg):
    events: FrankaCupCakeSideLyingFront3cmEventCfg = FrankaCupCakeSideLyingFront3cmEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = np.inf
        self.terminations.success.params["side_lying_asset_cfg"] = SceneEntityCfg(
            "insertive_object"
        )
        self.terminations.success.params[
            "side_lying_local_z_world_z_range"
        ] = (float(np.cos(np.deg2rad(78.0))), float(np.cos(np.deg2rad(70.0))))
        self.actions.body = None
        self.episode_length_s = 4.0


@configclass
class FrankaObjectPartiallyAssembledEEGraspedResetStatesCfg(FrankaPandaResetStatesCfg):
    events: FrankaObjectPartiallyAssembledEEGraspedEventCfg = FrankaObjectPartiallyAssembledEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.025
        self.terminations.success.params["insertive_asset_cfg"] = SceneEntityCfg("insertive_object")
        self.terminations.success.params["receptive_asset_cfg"] = SceneEntityCfg("receptive_object")
        self.terminations.success.params["assembly_success_prob"] = 0.5
        self.terminations.success.params["assembly_threshold_scale"] = 1.5
