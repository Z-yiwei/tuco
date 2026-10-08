# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene and manager-based env config for Franka FR3 system identification (CMA-ES).

Franka port of ur5e_robotiq_2f85/sysid_cfg.py. Reuses the same robot asset and
RelCartesianOSCAction as RL so the in-env OSC is identical to training — no
duplicate controller. Differences from the UR5e version:

- SYSID_SIM_DT = 1/1000: real data is collected by franka_osc_controller at
  the 1 kHz FCI rate (UR5e RTDE ran at 500 Hz).
- Gripper: plain binary action (the franka/actions.py Sysid class uses the
  grasp-guarded gripper, which requires an `insertive_object` in the scene —
  the sysid scene has no objects).
- The CMA-ES script replaces actuators `panda_arm1`/`panda_arm2` with a single
  DelayedPDActuator group; see scripts_v2/tools/sim2real/sysid_franka_osc.py.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.franka import BINARY_GRIPPER

from uwlab_tasks.manager_based.manipulation.factory_extension.factory_assets_cfg import FRANKA_PANDA_CFG

from ... import mdp as task_mdp
from .actions import FRANKA_FR3_RELATIVE_OSC_UNSCALED

# Default simulation timestep for sysid (1 kHz, matches franka_osc_controller /
# FCI control rate on the real robot)
SYSID_SIM_DT = 1.0 / 1000.0

# Franka arm joint names / EE body in the sim asset (panda naming convention).
FRANKA_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
FRANKA_EE_BODY_NAME = "panda_hand"
NUM_FRANKA_ARM_JOINTS = 7


@configclass
class FrankaSysidOSCAction:
    """Unscaled arm action (Cartesian delta) + plain binary gripper."""

    arm = FRANKA_FR3_RELATIVE_OSC_UNSCALED
    gripper = BINARY_GRIPPER


@configclass
class SysidSceneCfg(InteractiveSceneCfg):
    """Scene for system identification: robot + ground + light, no objects."""

    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


# Minimal MDP for sysid env (same action as RL; obs/rew/term minimal so env runs)
@configclass
class SysidObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SysidRewardsCfg:
    pass


@configclass
class SysidTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


@configclass
class SysidEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based env for sysid: same robot + RelCartesianOSC as RL, decimation=1."""

    scene: SysidSceneCfg = SysidSceneCfg(num_envs=512, env_spacing=2.0)
    actions: FrankaSysidOSCAction = FrankaSysidOSCAction()
    observations: SysidObservationsCfg = SysidObservationsCfg()
    rewards: SysidRewardsCfg = SysidRewardsCfg()
    terminations: SysidTerminationsCfg = SysidTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 1
        self.episode_length_s = 99999.0
        self.sim.dt = SYSID_SIM_DT
