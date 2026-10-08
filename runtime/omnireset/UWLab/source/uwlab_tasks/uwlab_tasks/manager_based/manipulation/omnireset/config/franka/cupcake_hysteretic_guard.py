"""Opt-in release hysteresis; original capture thresholds remain unchanged."""
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from .grasp_guarded_action import GraspGuardedBinaryGripperAction,GraspGuardedBinaryGripperActionCfg


class CupCakeHystereticGuard(GraspGuardedBinaryGripperAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        if not 0<=cfg.release_margin_m<=.005:raise ValueError('release margin outside tested range')
        self._capture_command=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        if not isinstance(cfg.capture_confirm_steps,int) or cfg.capture_confirm_steps<0:raise ValueError('invalid capture confirmation length')
        self._capture_count=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        self._capture_anchor=torch.zeros(env.num_envs,3,device=env.device)

    def _relative_position(self):
        data=self._asset.data
        pos=data.body_link_pos_w[:,self._ee_body_idx]
        rot=data.body_link_quat_w[:,self._ee_body_idx]
        tcp=pos+quat_apply(rot,self._tcp_offset.expand_as(pos))
        return quat_apply_inverse(rot,self._peg.data.root_pos_w-tcp)

    def compute_rule_actions(self):
        local=self._relative_position()
        extra=self.cfg.release_margin_m*self._capture_command
        closed=(local[:,:2].norm(dim=-1)<self._lateral_thresh+extra)&(local[:,2].abs()<self._vertical_thresh+extra)
        return torch.where(closed,-1.,1.)[:,None]

    def process_actions(self,actions):
        if self._hold_processed_action:return
        if self.cfg.capture_confirm_steps:
            inside=super().compute_rule_actions()[:,0]<0
            width=self._asset.data.joint_pos[:,self._joint_ids].sum(-1)
            valid=(self._raw_actions[:,0]<0)&inside&(width>.005)&(width<.075)
            local=self._relative_position()
            restart=(self._capture_count==0)|((local-self._capture_anchor).norm(dim=-1)>.01)
            self._capture_anchor[restart]=local[restart]
            self._capture_count=torch.where(valid,torch.where(restart,1,self._capture_count+1),0)
            self._capture_command|=self._capture_count>=self.cfg.capture_confirm_steps
        super().process_actions(actions)
        closing=self._raw_actions[:,0]<0
        if self.cfg.capture_confirm_steps:
            self._capture_command&=closing
            self._capture_count=torch.where(closing,self._capture_count,0)
        else:
            self._capture_command[:]=closing

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'_capture_command'):
            self._capture_command[slice(None) if env_ids is None else env_ids]=False
            self._capture_count[slice(None) if env_ids is None else env_ids]=0
            self._capture_anchor[slice(None) if env_ids is None else env_ids]=0


@configclass
class CupCakeHystereticGuardCfg(GraspGuardedBinaryGripperActionCfg):
    class_type:type=CupCakeHystereticGuard
    release_margin_m:float=.002
    capture_confirm_steps:int=0
