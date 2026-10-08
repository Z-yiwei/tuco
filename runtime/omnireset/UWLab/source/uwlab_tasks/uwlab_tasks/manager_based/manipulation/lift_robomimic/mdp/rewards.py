"""Reward functions for RoboMimic Lift task.

Supports both sparse (default, matching RoboMimic) and dense variants.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def lift_success(
    env: ManagerBasedEnv,
    minimal_height: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse reward: 1.0 if object is lifted above table surface + minimal_height.

    In RoboMimic, success = cube_z > table_z + 0.04.
    Here we use object height relative to env origin (table surface is at z~0).
    """
    obj = env.scene[object_cfg.name]
    object_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (object_height > minimal_height).float()


def reach_object(
    env: ManagerBasedEnv,
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reaching reward: 1 - tanh(distance / std). Matches RoboMimic dense reward."""
    robot = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(robot_cfg.body_names)
    ee_pos = robot.data.body_pos_w[:, body_ids[0], :]

    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w

    distance = torch.norm(ee_pos - obj_pos, dim=-1)
    return 1.0 - torch.tanh(distance / std)
