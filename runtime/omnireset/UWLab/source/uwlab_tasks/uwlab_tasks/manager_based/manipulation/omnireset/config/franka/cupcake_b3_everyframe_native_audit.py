"""All inserted rows: native placement/readback and one fixed-B3 physical tick."""
import json
from pathlib import Path
import torch
from ...mdp.events import sample_from_nested_dict
from .cupcake_unique_t0_recording import initial_matrix


def audit_replacement_frames(env,manager,root):
    mapping=json.loads((root/'source_index.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    robot=env.scene['robot'];objects=[env.scene[k] for k in ('insertive_object','receptive_object')]
    def read(ids):
        parts=[robot.data.joint_pos[ids],robot.data.joint_vel[ids]]
        for asset in [robot,*objects]:
            v=asset.data.root_state_w[ids].clone();v[:,:3]-=env.scene.env_origins[ids];parts.append(v)
        return torch.cat(parts,-1)
    max_error=max_motion=0.;checked=0;seen=[]
    for kind,name in enumerate(manifest['counts']):
        if kind==0:continue
        row_ids=mapping[name]['replaced_old_row_ids']
        for start in range(0,len(row_ids),env.num_envs):
            chosen=torch.tensor(row_ids[start:start+env.num_envs],device=env.device)
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
            max_error=max(max_error,error);max_motion=max(max_motion,motion);checked+=len(chosen)
            seen.extend((kind,int(j)) for j in chosen.tolist())
    assert checked==manifest['all_frame_count'] and len(set(seen))==checked
    result=dict(passed=True,count=checked,all_inserted_rows_tested=True,max_readback_error=max_error,
                max_one_tick_object_displacement_m=max_motion,physical_dt_s=env.physics_dt,
                extra_effort_zero=True,scope='placement and one-tick stability only; not recovery/teacher success gate')
    (Path(env.cfg.log_dir)/'everyframe_native_reset_audit.json').write_text(json.dumps(result,indent=2))
    print('[EVERYFRAME_NATIVE_AUDIT]',json.dumps(result),flush=True)
