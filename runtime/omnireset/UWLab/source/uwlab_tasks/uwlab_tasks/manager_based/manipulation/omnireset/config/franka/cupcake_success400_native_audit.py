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
              'dataset_root':str(root), 'counts':[], 'max_readback_error':0., 'max_object_step_displacement_m':0.}
    batch = min(env.num_envs,2048)
    for kind, dataset in enumerate(manager.datasets):
        count = int(manager.num_states[kind])
        for start in range(0,count,batch):
            chosen = torch.arange(start,min(start+batch,count),device=env.device)
            ids = torch.arange(len(chosen),device=env.device)
            state = sample_from_nested_dict(dataset,chosen)['initial_state']
            expected = flatten(state)
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
            assert torch.isfinite(after).all() and (after[:,31:34].abs()<10).all(), (kind,start,'nonfinite/extreme')
            print('[PHASE_RESET_AUDIT]',kind,min(start+batch,count),'/',count,'max_error',error,flush=True)
        result['counts'].append(count)
    result.update(passed=True,all_rows_checked=sum(result['counts']))
    path=Path(env.cfg.log_dir)/'phase_native_reset_audit.json'
    with path.open('x') as stream:
        json.dump(result,stream,indent=2)
    print('[PHASE_RESET_AUDIT_COMPLETE]',json.dumps(result),flush=True)
