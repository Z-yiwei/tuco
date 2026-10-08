"""Phase-balanced reset sampling, with equal source-trajectory mass per phase."""
import hashlib
import json
import os
from pathlib import Path
import torch
from .cupcake_success400_phase_reset import CupCakeSuccess400PhaseReset
from .cupcake_policy_gripper_cfg import RESET_TYPES
from ...mdp.events import ManagerTermBase, SuccessMonitorCfg


def tensor_tree(s):
    return {'initial_state':{'articulation':{'robot':{
        'root_pose':s[:,18:25], 'root_velocity':s[:,25:31],
        'joint_position':s[:,:9], 'joint_velocity':s[:,9:18]}},
        'rigid_object':{'insertive_object':{'root_pose':s[:,31:38],'root_velocity':s[:,38:44]},
                        'receptive_object':{'root_pose':s[:,44:51],'root_velocity':s[:,51:57]}}}}


class CupCakeB3Legacy1200Reset(CupCakeSuccess400PhaseReset):
    def __init__(self,cfg,env):
        # Same bookkeeping and RNG call as MultiResetManager.__init__, replacing
        # ONLY the expensive list[Tensor] file loader with an exact matrix cache.
        ManagerTermBase.__init__(self,cfg,env)
        root=Path(os.environ['CUPCAKE_PREGRASP1200_DATASET_ROOT'])
        manifest=json.loads((root/'manifest.json').read_text())
        assert Path(cfg.params['dataset_dir']).resolve()==root.resolve()
        assert cfg.params['reset_types']==RESET_TYPES and cfg.params['probs']==[.25]*4
        offsets=cfg.params.get('rigid_object_position_offsets') or {}
        assert isinstance(offsets,dict)
        self.rigid_object_position_offsets={}
        for asset_name,position_offset in offsets.items():
            assert asset_name in env.scene._rigid_objects
            value=torch.as_tensor(position_offset,dtype=torch.float32,device=env.device)
            assert value.shape==(3,) and torch.count_nonzero(value)==0
            self.rigid_object_position_offsets[asset_name]=value
        cache=json.loads((root/'compact_cache_contract.json').read_text())
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        assert cache['passed'] and cache['dataset_manifest_sha256']==digest(root/'manifest.json')
        assert cache['compact_sha256']==digest(root/'compact_states.pt')
        matrices=torch.load(root/'compact_states.pt',map_location='cpu',weights_only=False)
        self.datasets=[tensor_tree(matrices[name].to(env.device)) for name in RESET_TYPES]
        self.num_states=torch.tensor([len(matrices[name]) for name in RESET_TYPES],device=env.device)
        self.num_tasks=4;self.probs=torch.tensor(cfg.params['probs'],device=env.device)
        self.probs=self.probs/self.probs.sum()
        self.receptive_object_z_lift=float(os.environ.get('OMNIRESET_REAL_RECEPTIVE_OBJECT_Z_LIFT','0.0'))
        assert self.receptive_object_z_lift==0
        assert cfg.params.get('state_indices') is None and cfg.params.get('state_index_repeats',1)==1
        self._forced_state_sequence=None;self._forced_state_cursor=0
        if cfg.params.get('success') is not None:
            monitor_cfg=SuccessMonitorCfg(monitored_history_len=100,num_monitored_data=4,device=env.device)
            self.success_monitor=monitor_cfg.class_type(monitor_cfg)
        self.task_id=torch.randint(0,4,(self.num_envs,),device=self.device)
        self.state_id=torch.full((self.num_envs,),-1,dtype=torch.long,device=self.device)
        self.source_counts=torch.zeros(4,3,dtype=torch.long,device=env.device)
        self.state_weights=[]
        path=root/'sampling_weights.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sampling_weights_sha256']
        weights=torch.load(path,map_location='cpu',weights_only=False)
        mapping=torch.load(root/'source_index.pt',map_location='cpu',weights_only=False)
        for i,name in enumerate(RESET_TYPES):
            w=weights[name];keys=mapping[name]['source_keys']
            assert w.shape==(int(self.num_states[i]),) and torch.isfinite(w).all() and (w>0).all()
            source,inv=torch.unique(keys[:,0],return_inverse=True)
            mass=torch.zeros(len(source),dtype=torch.float64).scatter_add_(0,inv,w)
            assert torch.allclose(mass,torch.full_like(mass,1./len(source)),atol=1e-12,rtol=1e-9)
            self.state_weights.append(w.to(env.device))

