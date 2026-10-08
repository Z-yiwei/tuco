"""One full terminal-task-success rollout per sealed pregrasp T0 ID.

Read-only hooks: never sample RNG, alter actions, force resets, or rebalance paths.
Each rank watches its ID partition; a file lock additionally enforces global uniqueness.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch
from .cupcake_t0_trajectory import T0TrajectoryRecorder, validate_trajectory, sha


def initial_matrix(tree):
    s = tree['initial_state']; r = s['articulation']['robot']
    parts = [r['joint_position'],r['joint_velocity'],r['root_pose'],r['root_velocity']]
    for name in ('insertive_object','receptive_object'):
        parts.extend([s['rigid_object'][name]['root_pose'],s['rigid_object'][name]['root_velocity']])
    return torch.cat([torch.stack(v) if isinstance(v,list) else v for v in parts],-1).cpu()


def initial_error(actual, expected):
    actual = actual.clone()
    for start in (21,34,47):
        if (actual[start:start+4]*expected[start:start+4]).sum() < 0:
            actual[start:start+4] *= -1
    assert torch.isfinite(actual).all()
    return float((actual-expected).abs().max())


class UniqueStateQuota:
    def __init__(self, root, expected, reset_sha256):
        self.root = Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.expected = expected.cpu().clone(); self.count = len(expected)
        assert expected.shape == (self.count,57) and torch.isfinite(expected).all()
        # Reject duplicate physical poses even if different velocities/IDs disguise them.
        pose = torch.cat([expected[:,:9],expected[:,31:38],expected[:,44:51]],-1).clone()
        for start in (12,19):
            q=pose[:,start:start+4]
            # q/-q canonicalization for physical-pose uniqueness.
            for row in q:
                nz=torch.where(row.abs()>1e-8)[0]
                if len(nz) and row[nz[0]] < 0: row *= -1
        assert len(torch.unique(pose,dim=0)) == self.count, 'duplicate initial physical poses'
        self.contract = dict(version='cupcake_unique_pregrasp_t0_success_v1',target_unique_states=self.count,
            allowed_state_ids=list(range(self.count)),selection='terminal task_success only; one per reset SHA256 + state ID',
            reset_file_sha256=reset_sha256,expected_initial_states_sha256=hashlib.sha256(self.expected.numpy().tobytes()).hexdigest(),
            recorder_sha256=sha(__file__),base_recorder_sha256=sha(Path(__file__).with_name('cupcake_t0_trajectory.py')),
            initial_readback_max_abs_tolerance=1e-4,full_episode_required=True)
        with (self.root/'quota.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            p=self.root/'recording_contract.json'
            if p.exists(): assert json.loads(p.read_text()) == self.contract
            else:
                with p.open('x') as f: json.dump(self.contract,f,indent=2)
            entries=self.entries(); self.saved_ids={e['state_id'] for e in entries}
            assert len(entries)==len(self.saved_ids)
            self.status(entries)

    def entries(self):
        p=self.root/'manifest.jsonl'
        rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        assert len(list(self.root.glob('state_*.pt'))) == len(rows), 'orphan file: preserve and audit before resuming'
        assert all(0<=r['state_id']<self.count and r['task_success'] for r in rows)
        return rows

    def status(self, entries):
        ids={e['state_id'] for e in entries}
        value=dict(saved_unique_states=len(ids),target_unique_states=self.count,
            missing_state_ids=sorted(set(range(self.count))-ids),complete=len(ids)==self.count)
        temp=self.root/'coverage.tmp'
        with temp.open('w') as f:
            json.dump(value,f); f.flush(); os.fsync(f.fileno())
        os.replace(temp,self.root/'coverage.json')

    def full(self): return (self.root/'COMPLETE.json').exists()

    def save(self, payload):
        if not payload['metadata']['task_success']: return False
        validate_trajectory(payload)
        m=payload['metadata']; sid=m['state_id']
        assert 0<=sid<self.count and m['reset_file_sha256']==self.contract['reset_file_sha256']
        error=initial_error(payload['states'][0],self.expected[sid])
        assert error<1e-4, ('initial state mismatch',sid,error)
        m['initial_state_max_abs_error']=error
        m['expected_initial_state_sha256']=hashlib.sha256(self.expected[sid].numpy().tobytes()).hexdigest()
        with (self.root/'quota.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            entries=self.entries(); self.saved_ids={e['state_id'] for e in entries}
            if sid in self.saved_ids: return False
            p=self.root/f'state_{sid:04d}.pt'; temp=self.root/f'state_{sid:04d}.pt.tmp'
            assert not p.exists() and not temp.exists()
            with temp.open('xb') as f:
                torch.save(payload,f); f.flush(); os.fsync(f.fileno())
            restored=torch.load(temp,map_location='cpu',weights_only=False)
            validate_trajectory(restored)
            for key in ('states','actions','executed_actions','substep_actions','flags'):
                assert torch.equal(payload[key],restored[key]),key
            assert restored['metadata']==m
            digest=sha(temp); os.replace(temp,p)
            entry=dict(file=p.name,sha256=digest,state_id=sid,reset_file_sha256=m['reset_file_sha256'],
                episode_key=m['episode_key'],steps=m['steps'],task_success=True,held_lift_2s=m['held_lift_2s'],
                initial_state_max_abs_error=error,expected_initial_state_sha256=m['expected_initial_state_sha256'])
            index=self.root/'manifest.jsonl'
            with index.open('a') as f:
                f.write(json.dumps(entry)+'\n'); f.flush(); os.fsync(f.fileno())
            entries.append(entry); self.saved_ids.add(sid); self.status(entries)
            if len(entries)==self.count:
                with (self.root/'COMPLETE.json').open('x') as f:
                    json.dump(dict(count=self.count,unique_state_ids=sorted(self.saved_ids),manifest_sha256=sha(index)),f)
            return True


class UniqueT0Recorder(T0TrajectoryRecorder):
    def __init__(self,env,root):
        self.env=env; self.audit=env._cupcake_t0_audit; self.robot=env.scene['robot']
        self.grip=env.action_manager.get_term('gripper')
        self.buffers={}; self.pending=None; self.saved=0; self.stopped=False; self.diagnostic_all=False
        assert env.cfg.decimation==12 and abs(env.step_dt-.1)<1e-8
        dataset=Path(env.cfg.events.reset_from_reset_states.params['dataset_dir'])
        m=json.loads((dataset/'manifest.json').read_text())
        source=dataset/'Resets/CupCakeHalf__Plate/resets_ObjectAnywhereEEAnywhere.pt'
        digest=sha(source); assert digest==m['reset_sha256'][source.name]
        expected=initial_matrix(torch.load(source,map_location='cpu',weights_only=False))
        assert len(expected)==1200
        self.quota=UniqueStateQuota(root,expected,digest)
        self.rank=int(os.environ.get('RANK','0')); self.world=int(os.environ.get('WORLD_SIZE','1'))
        self.metadata=dict(version='cupcake_unique_pregrasp_t0_success_v1',seed=env.cfg.seed,rank=str(self.rank),
            run_dir=str(env.cfg.log_dir),reset_root=str(dataset),reset_sha256=sha(dataset/'manifest.json'),
            reset_file_sha256=digest,source_index=m['t0_source_index'],source_index_sha256=m['t0_source_index_sha256'],
            recorder_sha256=sha(__file__),control_dt_s=.1,physics_dt_s=env.sim.get_physics_dt(),
            states_columns={'joint_q':[0,9],'joint_qvel':[9,18],'robot_root_state':[18,31],
                            'object_root_state':[31,44],'plate_root_state':[44,57]},
            state_frame='env-local positions; wxyz quaternions; world-aligned velocities',
            actions_columns='7D actual input to env action manager, after wrapper clipping',
            executed_actions_columns='7 absolute arm joint targets + executed guard width',
            substep_actions_columns='7 arm joint targets + guard width + 7 feedforward efforts',
            flags_columns=['task_success','held_lift_2s','terminated','timeout'],
            selection='terminal task success only, one per sealed T0 ID; no held-only substitution',
            coverage='complete reset-to-terminal control states and all 12 substep commands; no RGB',
            policy_identity='changing training weights; saved actions authoritative, not fixed checkpoint rollout',
            resume_checkpoint=json.loads(Path(os.environ['CUPCAKE_UNIQUE_T0_RUN_CONTRACT']).read_text())['checkpoint'])

    def pre(self, actions):
        assert self.pending is None
        if self.stopped: return
        if self.quota.full():
            self.buffers.clear(); self.stopped=True
            print('[UNIQUE_T0] all1200 complete; buffers released, training continues',flush=True); return
        active=torch.where(self.audit.active)[0]
        keys=torch.stack([active,self.audit.episode[active],self.audit.state[active],self.audit.steps[active]],-1).cpu().tolist()
        occupied={b['state_id'] for b in self.buffers.values()}
        chosen=[]; new=[]
        for eid,ep,sid,steps in keys:
            if eid in self.buffers:
                assert self.buffers[eid]['episode']==ep
                chosen.append((eid,ep,sid)); continue
            # Do not begin in the middle of a rollout. Skip randomized initial episode lengths.
            if steps!=0 or ep<=1 or sid%self.world!=self.rank or sid in occupied or sid in self.quota.saved_ids: continue
            occupied.add(sid); chosen.append((eid,ep,sid)); new.append((eid,ep,sid))
        if not chosen: return
        if new:
            ids=torch.tensor([k[0] for k in new],device=self.env.device)
            initial=self.states(ids).detach().cpu()
            for (eid,ep,sid),start in zip(new,initial):
                error=initial_error(start,self.quota.expected[sid])
                assert error<1e-4,('reset start mismatch',sid,error)
                self.buffers[eid]=dict(episode=ep,state_id=sid,initial=start.clone(),
                    start_sim_step=int(self.env.common_step_counter),frames=[])
        ids=torch.tensor([k[0] for k in chosen],device=self.env.device)
        self.pending=dict(ids=ids,keys=chosen,action=actions[ids].detach().clone(),substeps=[])


def verify_with_unique_t0_recording(env,env_ids):
    from .cupcake_ik_pregrasp1200_cfg import verify_pregrasp1200_startup
    verify_pregrasp1200_startup(env,env_ids)
    recorder=UniqueT0Recorder(env,os.environ['CUPCAKE_UNIQUE_T0_ROOT'])
    assert not hasattr(env,'_cupcake_t0_trajectories')
    process=env.action_manager.process_action; apply=env.action_manager.apply_action; compute=env.reward_manager.compute
    def process_with_record(actions):
        recorder.pre(actions); return process(actions)
    def apply_with_record():
        result=apply(); recorder.substep(); return result
    def compute_with_record(dt):
        result=compute(dt); recorder.post(); return result
    env.action_manager.process_action=process_with_record
    env.action_manager.apply_action=apply_with_record
    env.reward_manager.compute=compute_with_record
    env._cupcake_t0_trajectories=recorder
    print('[UNIQUE_T0] enabled one terminal success per initial state; rank',recorder.rank,'of',recorder.world,flush=True)
