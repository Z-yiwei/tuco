"""Opt-in CupCake controller candidate; historical task defaults stay unchanged."""
import torch
from isaaclab.utils import configclass
from ...mdp.actions.task_space_actions import RelCartesianDiffIKJointPositionAction
from ...mdp.actions.actions_cfg import RelCartesianDiffIKJointPositionActionCfg
from .friction_compensation_math import friction_feedforward,capture_assist_scale


class CupCakeFrictionCompensatedDiffIKAction(RelCartesianDiffIKJointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self._friction_effort=torch.zeros_like(self.joint_position_targets)
        self._capture_assist_scale=torch.zeros(env.num_envs,device=env.device)
        self._friction_dt_s=env.sim.get_physics_dt()
        if cfg.friction_capture_ramp_s<0:raise ValueError('negative capture ramp')

    @property
    def friction_effort(self):
        return self._friction_effort

    def apply_actions(self):
        super().apply_actions()
        data=self._asset.data
        ids=self._joint_ids
        error=self.joint_position_targets-data.joint_pos[:,ids]
        velocity=data.joint_vel[:,ids]
        pd=data.joint_stiffness[:,ids]*error-data.joint_damping[:,ids]*velocity
        budget=(data.joint_effort_limits[:,ids]-pd.abs()).clamp(min=0)
        self._friction_effort[:]=friction_feedforward(
            error,velocity,data.joint_friction_coeff[:,ids],
            data.joint_dynamic_friction_coeff[:,ids],data.joint_viscous_friction_coeff[:,ids],
            static_gain=self.cfg.static_friction_compensation_gain,
            position_deadband=self.cfg.friction_position_deadband,
            velocity_threshold=self.cfg.friction_velocity_threshold,effort_budget=budget,
            assist_only_toward_target=self.cfg.friction_assist_only_toward_target,
            moving_gain=self.cfg.moving_friction_compensation_gain)
        if self.cfg.friction_after_capture_only:
            grip=self._env.action_manager.get_term('gripper')
            if not hasattr(grip,'_capture_command') or grip.cfg.capture_confirm_steps<=0:
                raise RuntimeError('post-capture compensation requires confirmed hysteretic guard')
            self._capture_assist_scale[:]=capture_assist_scale(
                self._capture_assist_scale,grip._capture_command,self._friction_dt_s,self.cfg.friction_capture_ramp_s)
            self._friction_effort*=self._capture_assist_scale[:,None]
        self._asset.set_joint_effort_target(self._friction_effort,joint_ids=ids)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        # Avoid leaving effort from a previous episode on newly reset rows.
        if hasattr(self,'_friction_effort'):
            index=slice(None) if env_ids is None else env_ids
            self._friction_effort[index]=0
            self._capture_assist_scale[index]=0
            self._asset.set_joint_effort_target(self._friction_effort[index],joint_ids=self._joint_ids,env_ids=env_ids)


@configclass
class CupCakeFrictionCompensatedDiffIKActionCfg(RelCartesianDiffIKJointPositionActionCfg):
    class_type:type=CupCakeFrictionCompensatedDiffIKAction
    static_friction_compensation_gain:float=.9
    moving_friction_compensation_gain:float=1.
    friction_after_capture_only:bool=False
    friction_capture_ramp_s:float=0.
    friction_position_deadband:float=1e-4
    friction_velocity_threshold:float=.001
    friction_assist_only_toward_target:bool=False
