# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import (
    BehaviorCloningCfg,
    OffPolicyAlgorithmCfg,
    RslRlFancyActorCriticCfg,
    RslRlFancyPpoAlgorithmCfg,
)


def my_experts_observation_func(env):
    """Expert observation getter for behavior cloning — reads ``expert_obs`` from env buf."""
    return env.unwrapped.obs_buf["expert_obs"]


@configclass
class Base_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    resume = False
    experiment_name = "franka_fr3_gripper_omnireset_agent"
    policy = RslRlFancyActorCriticCfg(
        # Matched to UR5e default (1.0). Earlier 3.0 setting combined with
        # scale_xyz_axisangle=0.08 produced 12× UR5e's single-step displacement,
        # blowing past the ±2 cm capture zone every step.
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Base_DAggerRunnerCfg(Base_PPORunnerCfg):
    """For collect_demos.py / eval_distilled_policy.py — wraps a JIT-traced
    state-RL expert as the BC teacher. ``experts_path`` is overridden via
    Hydra at run time (e.g. ``agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path='["exported/policy.pt"]'``).
    """

    algorithm = RslRlFancyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        offline_algorithm_cfg=OffPolicyAlgorithmCfg(
            behavior_cloning_cfg=BehaviorCloningCfg(
                experts_path=[""],
                experts_loader="torch.jit.load",
                experts_observation_group_cfg=(
                    "uwlab_tasks.manager_based.manipulation.omnireset.config.franka.rl_state_cfg"
                    ":ObservationsCfg.PolicyCfg"
                ),
                experts_observation_func=my_experts_observation_func,
                experts_action_group_cfg=(
                    "uwlab_tasks.manager_based.manipulation.omnireset.config.franka.actions"
                    ":FrankaFr3GripperRelativeOSCAction"
                ),
                cloning_loss_coeff=1.0,
                loss_decay=1.0,
            )
        ),
    )
