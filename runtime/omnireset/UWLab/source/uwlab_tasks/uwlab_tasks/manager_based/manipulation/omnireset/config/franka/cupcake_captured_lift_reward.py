"""Opt-in CupCake acquisition-referenced held-lift bonus, weight set by task."""
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from .captured_lift_tracker import CapturedLiftTracker


class CupCakeCapturedLiftBonus(ManagerTermBase):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.robot=env.scene['robot']
        self.object=env.scene['insertive_object']
        self.gripper=env.action_manager.get_term('gripper')
        self.hand=self.robot.find_bodies('fr3_hand')[0][0]
        self.fingers=self.robot.find_joints(['fr3_finger_joint1','fr3_finger_joint2'],preserve_order=True)[0]
        self.offset=torch.tensor(self.gripper.cfg.tcp_offset,device=env.device).view(1,3).expand(env.num_envs,-1)
        self.hold_steps=int(cfg.params.get('hold_steps',20))
        self.release_margin_m=float(cfg.params.get('release_margin_m',0.))
        self.tracker=CapturedLiftTracker(env.num_envs,env.device,self.hold_steps,self.release_margin_m)

    def reset(self,env_ids=None):
        self.tracker.reset(env_ids)

    def __call__(self,env,hold_steps=20,release_margin_m=0.):
        if hold_steps!=self.hold_steps or release_margin_m!=self.release_margin_m:
            raise ValueError('held-lift parameters cannot change after reward initialization')
        rot=self.robot.data.body_link_quat_w[:,self.hand]
        tcp=self.robot.data.body_link_pos_w[:,self.hand]+quat_apply(rot,self.offset)
        relative=quat_apply_inverse(rot,self.object.data.root_pos_w-tcp)
        return self.tracker.step(self.object.data.root_pos_w[:,2],tcp[:,2],
                                 self.gripper.processed_actions.sum(-1),
                                 self.robot.data.joint_pos[:,self.fingers].sum(-1),relative)
