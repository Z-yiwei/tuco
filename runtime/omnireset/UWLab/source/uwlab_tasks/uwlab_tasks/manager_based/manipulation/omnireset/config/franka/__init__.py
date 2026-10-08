# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka OmniReset task registrations."""

import gymnasium as gym

# Use string paths for agent cfgs to avoid importing rsl_rl/tensordict at discovery time.
# This prevents conflicts when --enable_cameras is used.
_AGENTS = f"{__name__}.agents"

# Register grasp sampling environment
gym.register(
    id="OmniReset-FrankaPanda-GraspSampling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.grasp_sampling_cfg:FrankaPandaGraspSamplingCfg"},
    disable_env_checker=True,
)

# Register RL state environments
gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaFr3GripperRelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaFr3GripperRelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Joint-position PPO teacher for guard-free joint-target BC collection.
gym.register(
    id="OmniReset-FrankaFr3Gripper-RelJointTarget-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaFr3GripperRelativeJointTargetTrainCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelJointTarget-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaFr3GripperRelativeJointTargetEvalCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Joint Stage 2: arm-only ADR from the identified B3 nominal to +/-20%.
gym.register(
    id="OmniReset-FrankaFr3Gripper-RelJointTarget-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:FrankaFr3GripperRelativeJointTargetFinetuneCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelJointTarget-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:FrankaFr3GripperRelativeJointTargetFinetuneEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Cartesian-output PPO teacher executed through DLS IK and an absolute
# joint-position plant.  These task IDs intentionally keep the policy action
# shape compatible with the canonical CupCake OSC checkpoint.
gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperRelCartesianDiffIKJointTargetTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperRelCartesianDiffIKJointTargetEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-XY5T0-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperRelCartesianDiffIKJointTargetXY5T0TrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-XY5T0-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperRelCartesianDiffIKJointTargetXY5T0EvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperRelCartesianDiffIKJointTargetFinetuneCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperRelCartesianDiffIKJointTargetFinetuneEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Peg-specific Cartesian-output / DiffIK joint-target tasks.  These retain the
# successful table0 OSC expert's big-hole, soft-gripper and fixed-scene
# contract instead of silently inheriting the CupCake/StackCube gripper plant.
gym.register(
    id="OmniReset-FrankaFr3Gripper-PegRelCartesianDiffIKJointTarget-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperPegRelCartesianDiffIKJointTargetTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-PegRelCartesianDiffIKJointTarget-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperPegRelCartesianDiffIKJointTargetEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-PegRelCartesianDiffIKJointTarget-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperPegRelCartesianDiffIKJointTargetFinetuneCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-PegRelCartesianDiffIKJointTarget-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "FrankaFr3GripperPegRelCartesianDiffIKJointTargetFinetuneEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-AbsoluteJointTarget-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaFr3GripperAbsoluteJointTargetEvalCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Finetune (Stage 2): curriculum-ramped OSC gains + action scale.
gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaFr3GripperRelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:FrankaFr3GripperRelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# --- RGB environments (mirrors UR5e Ur5eRobotiq2f85 RGB tasks) ---
gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianOSC-RGB-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_rgb_cfg:FrankaFr3GripperDataCollectionRGBRelCartesianOSCCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianOSC-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_rgb_cfg:FrankaFr3GripperEvalRGBRelCartesianOSCCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-AbsoluteJointTarget-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_rgb_cfg:FrankaFr3GripperEvalRGBAbsoluteJointTargetCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaFr3Gripper-RelCartesianOSC-RGB-OOD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_rgb_cfg:FrankaFr3GripperEvalRGBRelCartesianOSCOODCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

# Explicit official Franka Research 3 migration tasks. The historical
# FrankaFr3Gripper IDs above intentionally remain backed by franka_mimic.usd.
gym.register(
    id="OmniReset-FrankaResearch3-RelCartesianDiffIKJointTarget-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.research3_cfg:FrankaResearch3DiffIKJointTargetFinetuneEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaResearch3-AbsoluteJointTarget-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.research3_cfg:FrankaResearch3RGBAbsoluteJointTargetEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaResearch3-RelCartesianOSC-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.research3_cfg:FrankaResearch3RGBRelCartesianOSCEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

# Collision-only compatibility tasks. These retain the official Research 3
# articulation and visuals but use the historical Factory fingertip convex hulls.
gym.register(
    id="OmniReset-FrankaResearch3MimicFingertip-RelCartesianDiffIKJointTarget-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.research3_cfg:"
            "FrankaResearch3MimicFingertipDiffIKJointTargetFinetuneEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaResearch3MimicFingertip-RelCartesianDiffIKJointTarget-XY5T0-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.research3_cfg:"
            "FrankaResearch3MimicFingertipDiffIKJointTargetXY5T0EvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaResearch3MimicFingertip-AbsoluteJointTarget-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.research3_cfg:"
            "FrankaResearch3MimicFingertipRGBAbsoluteJointTargetEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaResearch3MimicFingertip-RelCartesianOSC-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.research3_cfg:"
            "FrankaResearch3MimicFingertipRGBRelCartesianOSCEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-FrankaResearch3MimicFingertip-RelCartesianOSC-XY5T0-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.research3_cfg:"
            "FrankaResearch3MimicFingertipRGBRelCartesianOSCXY5T0Cfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-CupCake-FrankaResearch3-PolicyGripper-DiffIK-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cupcake_policy_gripper_cfg:CupCakePolicyGripperTrainCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-CupCake-FrankaResearch3-LatchedGripper-DiffIK-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cupcake_latched_gripper_cfg:CupCakeLatchedGripperTrainCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-CupCakeHalf-FrankaResearch3-GuardedGripper-DiffIK-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.cupcake_half_guarded_gripper_cfg:CupCakeHalfGuardedTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-CupCakeHalf-FrankaResearch3-GuardedGripper-GraspRepair-DiffIK-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cupcake_grasp_repair_cfg:CupCakeGraspRepairTrainCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-CupCakeHalf-FrankaResearch3-GuardedGripper-TrajectoryAugmented-DiffIK-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cupcake_trajectory_augmented_cfg:CupCakeTrajectoryAugmentedTrainCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Register reset state recording environments
gym.register(
    id="OmniReset-FrankaPanda-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:FrankaObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-FrankaPanda-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:FrankaObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-FrankaPanda-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:FrankaObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-FrankaPanda-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_states_cfg:FrankaObjectPartiallyAssembledEEAnywhereResetStatesCfg"
    },
)

gym.register(
    id="OmniReset-FrankaPanda-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_states_cfg:FrankaObjectPartiallyAssembledEEGraspedResetStatesCfg"
    },
)

# Register SysID env (CMA-ES closed-loop replay; see sysid_cfg.py)
gym.register(
    id="OmniReset-FrankaFr3Gripper-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sysid_cfg:SysidEnvCfg"},
)
