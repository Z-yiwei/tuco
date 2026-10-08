"""Observation functions matching RoboMimic's 19-dim low_dim observation space.

RoboMimic obs layout:
  robot0_eef_pos    (3)  - end-effector position in world frame
  robot0_eef_quat   (4)  - end-effector orientation (x, y, z, w)
  robot0_gripper_qpos(2) - finger joint positions
  object            (10) - cube pos(3) + quat(4) + lin_vel(3)
  Total: 19
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def eef_pos_in_env_frame(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """End-effector position relative to environment origin. Shape: (N, 3)."""
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names)
    ee_pos_w = asset.data.body_pos_w[:, body_ids[0], :]  # (N, 3)
    return ee_pos_w - env.scene.env_origins


def eef_quat(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """End-effector quaternion (w, x, y, z) in world frame. Shape: (N, 4)."""
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names)
    return asset.data.body_quat_w[:, body_ids[0], :]  # (N, 4)


def gripper_joint_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
) -> torch.Tensor:
    """Gripper finger joint positions. Shape: (N, 2)."""
    asset = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
    return asset.data.joint_pos[:, joint_ids]  # (N, 2)


def object_state_in_env_frame(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object state: pos(3) + quat(4) + lin_vel(3) = 10D, relative to env origin."""
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins  # (N, 3)
    quat = obj.data.root_quat_w  # (N, 4)
    vel = obj.data.root_lin_vel_w  # (N, 3)
    return torch.cat([pos, quat, vel], dim=-1)  # (N, 10)
