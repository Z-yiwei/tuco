# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Official Franka Research 3 asset used by the explicit FR3 migration tasks."""

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .research3_mimic_fingertips import spawn_research3_with_mimic_fingertips


FRANKA_RESEARCH3_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.00871,
            "fr3_joint2": -0.10368,
            "fr3_joint3": -0.00794,
            "fr3_joint4": -1.49139,
            "fr3_joint5": -0.00083,
            "fr3_joint6": 1.38774,
            "fr3_joint7": 0.0,
            "fr3_finger_joint.*": 0.04,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    # Keep these dictionary keys stable. Existing Stage-2 SysID and runtime
    # overrides address actuator groups by key; only the USD joint names change.
    actuators={
        "panda_arm1": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.1,
            effort_limit_sim=87,
            velocity_limit_sim=2.62,
        ),
        "panda_arm2": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.1,
            effort_limit_sim=12,
            velocity_limit_sim=4.18,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint[1-2]"],
            effort_limit_sim=400.0,
            stiffness=5000.0,
            damping=50.0,
            friction=0.5,
            armature=0.0,
        ),
    },
)


# Explicit compatibility asset: official FR3 kinematics/visuals with only the
# historical Factory fingertip convex hulls substituted for contact replay.
FRANKA_RESEARCH3_MIMIC_FINGERTIP_CFG = FRANKA_RESEARCH3_CFG.replace(
    spawn=FRANKA_RESEARCH3_CFG.spawn.replace(func=spawn_research3_with_mimic_fingertips)
)
