"""Franka Panda config for RoboMimic Lift with Joint Velocity control.

Matches robosuite's default Panda controller:
  - JOINT_VELOCITY: input [-1,1] → output [-0.5, 0.5] rad/s
  - Gripper: binary open(0.04) / close(0.0)
"""

from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from ... import mdp
from ...lift_robomimic_env_cfg import LiftRobomimicEnvCfg


@configclass
class FrankaLiftRobomimicEnvCfg(LiftRobomimicEnvCfg):
    """Panda + Joint Velocity control, aligned with RoboMimic defaults."""

    def __post_init__(self):
        super().__post_init__()

        # --- Robot ---
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Match RoboMimic initial joint angles:
        #   [0, π/16, 0, -π/2-π/3, 0, π-0.2, π/4]
        #   ≈ [0, 0.196, 0, -1.833, 0, 2.942, 0.785]
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": 0.1963,
            "panda_joint3": 0.0,
            "panda_joint4": -1.8326,
            "panda_joint5": 0.0,
            "panda_joint6": 2.9416,
            "panda_joint7": 0.7854,
            "panda_finger_joint.*": 0.04,
        }

        # --- Actions ---
        # Arm: Joint Velocity (RoboMimic default controller)
        # Input [-1, 1] scaled to [-0.5, 0.5] rad/s
        self.actions.arm_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=False,
        )
        # Gripper: Binary open/close
        # RoboMimic: finger range [0, 0.04] m
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )


@configclass
class FrankaLiftRobomimicEnvCfg_PLAY(FrankaLiftRobomimicEnvCfg):
    """Playback config with fewer envs and no obs corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
