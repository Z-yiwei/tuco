"""Opt-in reset isolation for the new paper-cup collection contract."""
import torch

from ...mdp.events import reset_end_effector_round_fixed_asset


class IsolatedResetEndEffector(reset_end_effector_round_fixed_asset):
    """Keep IK's batched temporary targets from moving non-reset environments."""

    def __call__(self, env, env_ids, fixed_asset_cfg, fixed_asset_offset,
                 pose_range_b, robot_ik_cfg):
        others = None
        previous = None
        if env_ids is not None and len(env_ids) < self.robot.num_instances:
            mask = torch.ones(self.robot.num_instances, device=self.robot.device, dtype=torch.bool)
            mask[env_ids] = False
            others = mask.nonzero().flatten()
            previous = self.robot.data.joint_pos_target[others].clone()
        try:
            return super().__call__(env, env_ids, fixed_asset_cfg, fixed_asset_offset,
                                    pose_range_b, robot_ik_cfg)
        finally:
            if others is not None:
                self.robot.set_joint_position_target(previous, env_ids=others)
