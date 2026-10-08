"""Separate registration; historical registry and task remain untouched."""
import gymnasium as gym

gym.register(
    id='OmniReset-CupCakeHalf-FrankaResearch3-GuardedGripper-Success400Phase-DiffIK-State-v0',
    entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': 'uwlab_tasks.manager_based.manipulation.omnireset.config.franka.cupcake_success400_phase_cfg:CupCakeSuccess400PhaseTrainCfg',
        'rsl_rl_cfg_entry_point': 'uwlab_tasks.manager_based.manipulation.omnireset.config.franka.agents.rsl_rl_cfg:Base_PPORunnerCfg',
    },
)
