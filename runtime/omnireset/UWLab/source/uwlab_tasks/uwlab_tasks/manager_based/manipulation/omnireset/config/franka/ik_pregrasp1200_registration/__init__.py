"""Versioned task; historical tasks and their frozen contracts are unchanged."""
import gymnasium as gym

gym.register(
    id='OmniReset-CupCakeHalf-FrankaResearch3-GuardedGripper-IKPregrasp1200-DiffIK-State-v0',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point':'uwlab_tasks.manager_based.manipulation.omnireset.config.franka.cupcake_ik_pregrasp1200_cfg:CupCakeIKPregrasp1200TrainCfg',
        'rsl_rl_cfg_entry_point':'uwlab_tasks.manager_based.manipulation.omnireset.config.franka.agents.rsl_rl_cfg:Base_PPORunnerCfg',
    },
)
