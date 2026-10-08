"""Gymnasium registration for RoboMimic Lift (Panda)."""

import gymnasium as gym

from . import agents

# Joint Velocity control (matches RoboMimic default)
gym.register(
    id="UW-Lift-Robomimic-Panda-JointVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_vel_env_cfg:FrankaLiftRobomimicEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-Lift-Robomimic-Panda-JointVel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_vel_env_cfg:FrankaLiftRobomimicEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
