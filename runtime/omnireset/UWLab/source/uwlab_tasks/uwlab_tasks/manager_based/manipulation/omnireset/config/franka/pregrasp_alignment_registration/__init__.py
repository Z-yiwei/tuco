"""Opt-in task registration; never replaces the existing running task."""
import gymnasium as gym

gym.register(
    id='OmniReset-CupCakeHalf-FrankaResearch3-GuardedGripper-OpenAlignment-DiffIK-State-v0',
    entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': 'uwlab_tasks.manager_based.manipulation.omnireset.config.franka.cupcake_pregrasp_alignment_cfg:CupCakePregraspAlignmentTrainCfg',
        'rsl_rl_cfg_entry_point': 'uwlab_tasks.manager_based.manipulation.omnireset.config.franka.agents.rsl_rl_cfg:Base_PPORunnerCfg',
    },
)
