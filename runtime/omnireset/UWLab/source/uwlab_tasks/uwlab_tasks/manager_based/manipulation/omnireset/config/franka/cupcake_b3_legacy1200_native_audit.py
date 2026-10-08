"""Exhaustive native restore + one physical B3 tick, not a recovery-success gate."""
import json
from pathlib import Path
import torch
from ...mdp.events import sample_from_nested_dict
from .cupcake_unique_t0_recording import initial_matrix


def audit_pool_frames(env,manager,root):
    manifest=json.loads((root/'manifest.json').read_text())
    robot=env.scene['robot'];objects=[env.scene[k] for k in ('insertive_object','receptive_object')]
    def read(ids):
        parts=[robot.data.joint_pos[ids],robot.data.joint_vel[ids]]
        for asset in [robot,*objects]:
            v=asset.data.root_state_w[ids].clone();v[:,:3]-=env.scene.env_origins[ids];parts.append(v)
        return torch.cat(parts,-1)
    max_error=max_motion=0.;checked=0;counts={}
    # Use the same first 512 origins at every capacity to keep numerical restore
    # precision independent of the distant origins of large-env stress tests.
    batch=min(env.num_envs,512)
    for kind,name in enumerate(manifest['counts']):
        count=manifest['counts'][name];counts[name]=0
        for start in range(0,count,batch):
            chosen=torch.arange(start,min(start+batch,count),device=env.device)
            ids=torch.arange(len(chosen),device=env.device)
            values=sample_from_nested_dict(manager.datasets[kind],chosen)
            expected=initial_matrix(values).to(env.device)
            manager._reset_to(values['initial_state'],env_ids=ids,is_relative=True)
            robot.set_joint_position_target(expected[:,:9],env_ids=ids)
            robot.set_joint_velocity_target(torch.zeros_like(expected[:,:9]),env_ids=ids)
            robot.set_joint_effort_target(torch.zeros_like(expected[:,:9]),env_ids=ids)
            env.sim.forward();env.scene.update(env.physics_dt)
            actual=read(ids);comparable=actual.clone()
            for col in (21,34,47):
                flip=(comparable[:,col:col+4]*expected[:,col:col+4]).sum(-1)<0
                comparable[flip,col:col+4]*=-1
            error=float((comparable-expected).abs().max());assert error<1e-4,(kind,start,error)
            env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
            after=read(ids);assert torch.isfinite(after).all()
            motion=float((after[:,[31,32,33,44,45,46]]-actual[:,[31,32,33,44,45,46]]).abs().max())
            assert motion<.02,('one_tick_displacement_above_2cm',kind,start,motion)
            assert torch.count_nonzero(robot.data.joint_effort_target[ids,:7])==0
            max_error=max(max_error,error);max_motion=max(max_motion,motion);checked+=len(chosen);counts[name]+=len(chosen)
            if start%(batch*50)==0:print('[LEGACY_POOL_NATIVE]',kind,start,count,flush=True)
    assert checked==759425 and counts==manifest['counts']
    result=dict(passed=True,count=checked,counts=counts,max_readback_error=max_error,
                max_one_tick_object_displacement_m=max_motion,physical_dt_s=env.physics_dt,
                extra_effort_zero=True,scope='all candidate placement and one-tick check only; not recovery/teacher SR')
    (Path(env.cfg.log_dir)/'legacy_pool_native_reset_audit.json').write_text(json.dumps(result,indent=2))
    print('[LEGACY_POOL_NATIVE_PASS]',json.dumps(result),flush=True)
