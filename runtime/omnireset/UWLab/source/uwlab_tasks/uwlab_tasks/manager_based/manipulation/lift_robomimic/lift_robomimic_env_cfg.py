"""RoboMimic Lift environment config for IsaacLab.

Scene geometry, control timing, observations, and rewards are aligned
with robosuite's Lift task to enable sim-to-sim (MuJoCo ↔ PhysX) comparison.

RoboMimic Lift reference parameters:
  - Robot: Franka Panda, 7-DOF arm + 2-DOF parallel gripper
  - Table: 0.8 x 0.8 x 0.05 m, surface at z=0.8 (relative to world)
  - Cube: ~4.2 cm, ~75 g, red, placed at table center ±3 cm
  - Control: 20 Hz (Joint Velocity default), MuJoCo dt=0.002 s
  - Episode: 1000 control steps = 50 s
  - Success: cube lifted > 4 cm above table
  - Obs (low_dim): eef_pos(3) + eef_quat(4) + gripper(2) + object(10) = 19D
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Scene — matched to RoboMimic Lift geometry
##


@configclass
class LiftRobomimicSceneCfg(InteractiveSceneCfg):
    """Scene: Panda + table + red cube, geometry aligned with robosuite Lift.

    Coordinate convention (matching IsaacLab Lift layout):
      - Robot base at origin (0, 0, 0)
      - Table surface at z ≈ 0, centered at x = 0.5 in front of robot
      - Cube sitting on table surface
    """

    # Robot — set by subclass (Panda with JointVelocity or JointPosition)
    robot: ArticulationCfg = MISSING

    # Ground plane (below table)
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # Dome light
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Table — CuboidCfg matching RoboMimic: 0.8 x 0.8 x 0.05 m
    # Surface at z = 0.0 (center at z = -0.025)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.45, 0.4)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.025)),
    )

    # Cube — RoboMimic: ~4.2 cm (half-ext 0.021), ~75 g, red, μ=1.0
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.042, 0.042, 0.042),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.075),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        # Cube center at table_surface + half_cube = 0.0 + 0.021
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.021), rot=(1, 0, 0, 0)),
    )


##
# MDP — aligned with RoboMimic Lift
##


@configclass
class ActionsCfg:
    """Action specs — set by subclass (JointVelocity or JointPosition)."""

    arm_action: mdp.JointVelocityActionCfg | mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """RoboMimic-matched 19D low_dim observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        # eef_pos (3D)
        eef_pos = ObsTerm(func=mdp.eef_pos_in_env_frame)
        # eef_quat (4D)
        eef_quat = ObsTerm(func=mdp.eef_quat)
        # gripper_qpos (2D)
        gripper_qpos = ObsTerm(func=mdp.gripper_joint_pos)
        # object state: pos(3) + quat(4) + vel(3) = 10D
        object_state = ObsTerm(func=mdp.object_state_in_env_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events — cube position randomized ±3 cm (matching RoboMimic)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # RoboMimic: cube placed at table center ±0.03 m in x, y
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Sparse reward (RoboMimic default): 1.0 if cube lifted > 4 cm above table."""

    lift_success = RewTerm(
        func=mdp.lift_success,
        params={"minimal_height": 0.04},
        weight=1.0,
    )


@configclass
class TerminationsCfg:
    """Episode ends on timeout or if cube falls off table."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


##
# Full environment config
##


@configclass
class LiftRobomimicEnvCfg(ManagerBasedRLEnvCfg):
    """RoboMimic Lift environment, parameterized to match robosuite."""

    scene: LiftRobomimicSceneCfg = LiftRobomimicSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Timing aligned with RoboMimic: dt=0.002s, decimation=25 → 20 Hz control."""
        self.sim.dt = 0.002             # 500 Hz physics (matches MuJoCo timestep)
        self.decimation = 25            # 25 physics steps per control step → 20 Hz
        self.episode_length_s = 50.0    # 1000 control steps × 0.05s = 50s
        self.sim.render_interval = self.decimation

        # PhysX solver settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
