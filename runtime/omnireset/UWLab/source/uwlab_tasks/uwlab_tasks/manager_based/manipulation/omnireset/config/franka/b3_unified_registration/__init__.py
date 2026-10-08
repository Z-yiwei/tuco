"""Isolated task registration; do not mutate historical CupCake tasks."""
import gymnasium as gym

gym.register(
    id='OmniReset-CupCakeHalf-FrankaResearch3-B3Unified-IKPregrasp1200-DiffIK-State-v0',
    entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': 'uwlab_tasks.manager_based.manipulation.omnireset.config.franka.cupcake_b3_unified_cfg:CupCakeB3UnifiedTrainCfg',
        'rsl_rl_cfg_entry_point': 'uwlab_tasks.manager_based.manipulation.omnireset.config.franka.agents.rsl_rl_cfg:Base_PPORunnerCfg',
    },
)
