"""All-row native reset readback and one-step physics diagnostic, smoke only."""
import json
from pathlib import Path
import torch
from ...mdp.events import sample_from_nested_dict


def flatten(state):
    r = state['articulation']['robot']
    o, p = [state['rigid_object'][k] for k in ('insertive_object','receptive_object')]
    return torch.cat([r['joint_position'],r['joint_velocity'],r['root_pose'],r['root_velocity'],
                      o['root_pose'],o['root_velocity'],p['root_pose'],p['root_velocity']], -1)


def measured(env, ids):
    robot = env.scene['robot']
    roots = [obj.data.root_state_w[ids].clone() for obj in
             (robot,env.scene['insertive_object'],env.scene['receptive_object'])]
    for root in roots:
        root[:,:3] -= env.scene.env_origins[ids]
    return torch.cat([robot.data.joint_pos[ids],robot.data.joint_vel[ids],*roots],-1)


def audit_all_resets(env, manager, root):
    result = {'scope':'all four paths: real reset setter + native readback + one physics step; not long-term grasp stability',
              'scene_reset_before_write':True, 'dataset_root':str(root), 'path_diagnostics':[], 'counts':[], 'max_readback_error':0., 'max_object_step_displacement_m':0.}
    batch = min(env.num_envs,2048)
    for kind, dataset in enumerate(manager.datasets):
        count = int(manager.num_states[kind])
        diagnostic={'path':kind,'max_displacement_m':0.,'above_2cm':0,'above_5cm':0,'worst_rows':[]}
        for start in range(0,count,batch):
            chosen = torch.arange(start,min(start+batch,count),device=env.device)
            ids = torch.arange(len(chosen),device=env.device)
            state = sample_from_nested_dict(dataset,chosen)['initial_state']
            expected = flatten(state)
            env.scene.reset(ids)  # single-variable ablation: canonical pre-reset buffer clearing
            manager._reset_to(state,env_ids=ids,is_relative=True)
            env.sim.forward()
            env.scene.update(env.physics_dt)
            actual = measured(env,ids)
            # Quaternions q and -q encode the same restored orientation.
            for begin in (21,34,47):
                q, ref = actual[:,begin:begin+4], expected[:,begin:begin+4]
                q[(q*ref).sum(-1)<0] *= -1
            error = float((actual-expected).abs().max())
            result['max_readback_error'] = max(result['max_readback_error'],error)
            assert torch.isfinite(actual).all() and error < .0001, (kind,start,error)
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            after = measured(env,ids)
            motion = (after[:,31:34]-actual[:,31:34]).norm(dim=-1)
            result['max_object_step_displacement_m'] = max(result['max_object_step_displacement_m'],float(motion.max()))
            diagnostic['max_displacement_m']=max(diagnostic['max_displacement_m'],float(motion.max()))
            diagnostic['above_2cm']+=int((motion>.02).sum())
            diagnostic['above_5cm']+=int((motion>.05).sum())
            for j in torch.where(motion>.02)[0].tolist():
                diagnostic['worst_rows'].append({'row':int(chosen[j]),'motion_m':float(motion[j]),
                    'source_linear_speed_m_s':float(expected[j,38:41].norm()),
                    'before':actual[j,31:44].tolist(),'after':after[j,31:44].tolist()})
            assert torch.isfinite(after).all() and (after[:,31:34].abs()<10).all(), (kind,start,'nonfinite/extreme')
            print('[PHASE_RESET_AUDIT]',kind,min(start+batch,count),'/',count,'max_error',error,flush=True)
        result['counts'].append(count)
        result['path_diagnostics'].append(diagnostic)
    result.update(passed=True,all_rows_checked=sum(result['counts']))
    path=Path(env.cfg.log_dir)/'phase_native_reset_audit.json'
    with path.open('x') as stream:
        json.dump(result,stream,indent=2)
    print('[PHASE_RESET_AUDIT_COMPLETE]',json.dumps(result),flush=True)


def verify_diagnostic(env,env_ids):
    import os
    from .cupcake_success400_phase_reset import verify_phase_startup
    assert os.environ['CUPCAKE_PHASE_FULL_RESET_AUDIT']=='0'
    verify_phase_startup(env,env_ids)
    audit_all_resets(env,env.event_manager.get_term_cfg('reset_from_reset_states').func,
                     Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']).resolve())


