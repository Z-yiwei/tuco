"""Opt-in alignment reward. Reads state only; leaves guard/actions/physics intact."""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .cupcake_pregrasp_alignment_math import (
    AlignmentParameters, AlignmentProgressTracker, alignment_quality,
)


class CupCakePregraspAlignmentProgress(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.params = AlignmentParameters(**cfg.params)
        if cfg.weight != 1.0 or not math.isclose(env.step_dt, .1):
            raise ValueError('alignment uses weight=1, dt=.1; tracker returns actual step reward')
        success = env.cfg.rewards.success_reward.weight*env.step_dt
        if self.params.step_cap > .5*success:
            raise ValueError('alignment step cap must not exceed half a success step')
        self.robot = env.scene['robot']
        self.obj = env.scene['insertive_object']
        self.grip = env.action_manager.get_term('gripper')
        self.hand = self.robot.find_bodies('fr3_hand')[0][0]
        self.fingers = self.robot.find_joints(['fr3_finger_joint1', 'fr3_finger_joint2'], preserve_order=True)[0]
        self.offset = torch.tensor(self.grip.cfg.tcp_offset, device=env.device).expand(env.num_envs, -1)
        self.jaw_axis = torch.tensor([0., 1., 0.], device=env.device).expand(env.num_envs, -1)
        self.tracker = AlignmentProgressTracker(env.num_envs, env.device, self.params)

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)

    def __call__(self, env, target_height_m, lateral_std_m, height_std_m,
                 pad_level_std_m, open_half_width_m, coarse_std_m, near_full_m,
                 near_zero_m, coarse_fraction, eligible_width_m, episode_budget,
                 step_cap):
        # IsaacLab validates explicit parameter names; **kwargs is not accepted
        # by this version's ManagerBase signature checker.
        supplied = locals()
        if any(supplied[name] != value for name, value in vars(self.params).items()):
            raise ValueError('alignment parameters changed after initialization')
        rot = self.robot.data.body_link_quat_w[:, self.hand]
        tcp = self.robot.data.body_link_pos_w[:, self.hand]+quat_apply(rot, self.offset)
        tcp_local = quat_apply_inverse(self.obj.data.root_quat_w, tcp-self.obj.data.root_pos_w)
        jaw_local = quat_apply_inverse(self.obj.data.root_quat_w, quat_apply(rot, self.jaw_axis))
        quality, valid = alignment_quality(tcp_local, jaw_local, self.params)
        success = env.reward_manager.get_term_cfg('progress_context').func.success
        actual_return = self.tracker.step(quality, valid,
            self.robot.data.joint_pos[:, self.fingers].sum(-1),
            self.grip._capture_command, success)
        # RewardManager multiplies by weight*dt. Do NOT silently scale the
        # episode budget down by another factor of ten.
        return actual_return/env.step_dt


def verify_alignment_startup(env, env_ids):
    from .cupcake_success400_phase_reset import verify_phase_startup
    verify_phase_startup(env, env_ids)
    path = Path(os.environ['CUPCAKE_ALIGNMENT_CONTRACT'])
    contract = json.loads(path.read_text())
    assert contract['version'] == 'cupcake_open_alignment_progress_v1'
    for name, digest in contract['code_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    term = env.reward_manager.get_term_cfg('pregrasp_alignment_progress')
    assert type(term.func) is CupCakePregraspAlignmentProgress and term.weight == 1.
    assert term.params == contract['reward_params'] == asdict(AlignmentParameters())
    report = dict(version=contract['version'], reward_params=term.params,
        step_cap_actual=term.func.params.step_cap, episode_budget_actual=term.func.params.episode_budget,
        success_step_actual=env.cfg.rewards.success_reward.weight*env.step_dt,
        scope='only additive bounded open-jaw geometry progress; no grasp success claim',
        code_sha256=contract['code_sha256'])
    output = Path(env.cfg.log_dir)/f'alignment_contract_rank{os.environ.get("RANK", "0")}.json'
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
