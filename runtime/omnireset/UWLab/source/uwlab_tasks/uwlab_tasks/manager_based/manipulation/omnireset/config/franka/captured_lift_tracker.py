"""Kinematic held-lift evidence referenced to acquisition, not episode spawn z.

No contact-force certificate: reward requires both object and TCP to rise from
their first captured frame and remain mutually stable for a full hold window.
"""
import torch


class CapturedLiftTracker:
    def __init__(self,num_envs,device='cpu',hold_steps=20,release_margin_m=0.):
        assert hold_steps>0
        assert 0<=release_margin_m<=.005
        self.hold_steps=hold_steps
        self.release_margin_m=release_margin_m
        self.captured=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.given=torch.zeros_like(self.captured)
        self.ref_object_z=torch.zeros(num_envs,device=device)
        self.ref_tcp_z=torch.zeros_like(self.ref_object_z)
        self.count=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.anchor=torch.zeros(num_envs,3,device=device)

    def reset(self,ids=None):
        ids=slice(None) if ids is None else ids
        self.captured[ids]=False
        self.given[ids]=False
        self.count[ids]=0
        self.anchor[ids]=0
        self.ref_object_z[ids]=0
        self.ref_tcp_z[ids]=0

    def step(self,object_z,tcp_z,command_width,measured_width,relative):
        finite=torch.isfinite(object_z)&torch.isfinite(tcp_z)&torch.isfinite(command_width)&torch.isfinite(measured_width)&torch.isfinite(relative).all(-1)
        lateral=relative[:,:2].norm(dim=-1)
        vertical=relative[:,2].abs()
        acquisition_zone=(lateral<.02)&(vertical<.03)
        retained_zone=(lateral<.02+self.release_margin_m)&(vertical<.03+self.release_margin_m)
        captured=finite&(command_width==0)&(measured_width>.005)&(measured_width<.075)&(acquisition_zone|(self.captured&retained_zone))
        acquire=captured&~self.captured
        self.ref_object_z[acquire]=object_z[acquire]
        self.ref_tcp_z[acquire]=tcp_z[acquire]
        self.captured.copy_(captured)
        valid=captured&(object_z-self.ref_object_z>=.03)&(tcp_z-self.ref_tcp_z>=.03)
        restart=acquire|(self.count==0)|((relative-self.anchor).norm(dim=-1)>.01)
        self.anchor[restart]=relative[restart]
        self.count=torch.where(valid,torch.where(restart,1,self.count+1),0)
        fire=valid&(self.count>=self.hold_steps)&~self.given
        self.given|=fire
        return fire.float()
