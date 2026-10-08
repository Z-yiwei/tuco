"""Read-only retrospective T0 trajectories; shared, process-safe 200-file cap.

States are sampled at control boundaries. Every physics-substep command is
preserved, including feedforward effort. This is not an RGB/video recorder.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrajectoryQuota:
    def __init__(self, root, diagnostic_all=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = dict(version='cupcake_t0_states_actions_v1', max_trajectories=200,
                             selection='diagnostic_all' if diagnostic_all else 'task_success_or_held_lift_2s',
                             recorder_sha256=sha(__file__))
        with (self.root/'quota.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root/'recording_contract.json'
            if path.exists():
                assert json.loads(path.read_text()) == self.contract, 'recording contract drift'
            else:
                with path.open('x') as stream:
                    json.dump(self.contract, stream, indent=2)

    def full(self):
        return (self.root/'COMPLETE.json').exists()

    def save(self, payload):
        validate_trajectory(payload)
        with (self.root/'quota.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            index = self.root/'manifest.jsonl'
            entries = [json.loads(x) for x in index.read_text().splitlines()] if index.exists() else []
            # Never silently overwrite an orphan left by an interrupted commit.
            assert len(list(self.root.glob('trajectory_*.pt'))) == len(entries), 'orphan trajectory: recover before continuing'
            if len(entries) >= 200:
                return False
            key = payload['metadata']['episode_key']
            assert not any(e['episode_key'] == key for e in entries), 'duplicate episode'
            number = len(entries) + 1
            path = self.root/f'trajectory_{number:05d}.pt'
            temp = self.root/f'trajectory_{number:05d}.pt.tmp'
            assert not path.exists() and not temp.exists(), 'preserve incomplete prior write'
            with temp.open('xb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            restored = torch.load(temp, map_location='cpu', weights_only=False)
            validate_trajectory(restored)
            for name in ('states', 'actions', 'executed_actions', 'substep_actions', 'flags'):
                assert torch.equal(payload[name], restored[name]), f'write/read mismatch: {name}'
            digest = sha(temp)
            os.replace(temp, path)
            entry = dict(file=path.name, sha256=digest, episode_key=key,
                         state_id=payload['metadata']['state_id'], steps=len(payload['actions']),
                         task_success=payload['metadata']['task_success'],
                         held_lift_2s=payload['metadata']['held_lift_2s'])
            with index.open('a') as stream:
                stream.write(json.dumps(entry, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            if number == 200:
                with (self.root/'COMPLETE.json').open('x') as stream:
                    json.dump(dict(count=200, manifest_sha256=sha(index)), stream)
            return True


def validate_trajectory(payload):
    t = len(payload['actions'])
    assert 1 <= t <= 640
    assert payload['states'].shape == (t+1, 57)
    assert payload['actions'].shape == (t, 7)
    assert payload['executed_actions'].shape == (t, 8)
    assert payload['substep_actions'].shape == (t, 12, 15)
    assert payload['flags'].shape == (t, 4) and payload['flags'].dtype == torch.bool
    for name in ('states', 'actions', 'executed_actions', 'substep_actions'):
        assert payload[name].dtype == torch.float32 and torch.isfinite(payload[name]).all(), name
    assert torch.equal(payload['executed_actions'], payload['substep_actions'][:, -1, :8])
    assert ((payload['substep_actions'][:, :, 7] == 0) | (payload['substep_actions'][:, :, 7] == .08)).all()
    assert not payload['flags'][:-1, 2:].any(), 'trajectory crossed an automatic reset'
    assert payload['flags'][-1, 2:].any(), 'missing terminal frame'
    assert payload['metadata']['task_success'] == bool(payload['flags'][-1, 0])
    assert payload['metadata']['held_lift_2s'] == bool(payload['flags'][-1, 1])
    assert payload['metadata']['steps'] == t


class T0TrajectoryRecorder:
    def __init__(self, env, root, diagnostic_all=False):
        self.env = env
        self.audit = env._cupcake_t0_audit
        self.robot = env.scene['robot']
        self.grip = env.action_manager.get_term('gripper')
        self.quota = TrajectoryQuota(root, diagnostic_all)
        self.diagnostic_all = diagnostic_all
        self.buffers = {}
        self.pending = None
        self.saved = 0
        self.stopped = False
        assert env.cfg.decimation == 12 and abs(env.step_dt-.1) < 1e-8
        dataset = Path(env.cfg.events.reset_from_reset_states.params['dataset_dir'])
        manifest = dataset/'manifest.json'
        if not manifest.exists():
            manifest = dataset/'candidate_manifest.json'
        self.metadata = dict(version='cupcake_t0_states_actions_v1', seed=env.cfg.seed,
                             rank=os.environ.get('RANK', '0'), run_dir=str(env.cfg.log_dir),
                             reset_root=str(dataset.resolve()), reset_sha256=sha(manifest),
                             recorder_sha256=sha(__file__), control_dt_s=env.step_dt,
                             physics_dt_s=env.sim.get_physics_dt(),
                             states_columns={'joint_q':[0,9], 'joint_qvel':[9,18],
                                             'robot_root_state':[18,31], 'object_root_state':[31,44],
                                             'plate_root_state':[44,57]},
                             state_frame='env-local positions; wxyz quaternions; world-aligned velocities',
                             actions_columns='7D input to env action manager (after wrapper clipping)',
                             executed_actions_columns='7 arm absolute joint targets + total guard width',
                             substep_actions_columns='7 arm joint targets + guard width + 7 feedforward effort targets',
                             flags_columns=['task_success','held_lift_2s','terminated','timeout'],
                             coverage='complete control-boundary states and every physics-substep command; no RGB/contact-force certificate',
                             policy_identity='weights change during training; saved actions are authoritative, not a fixed policy rollout')
        if (dataset/'training_contract.json').exists():
            self.metadata['training_contract_sha256'] = sha(dataset/'training_contract.json')

    def states(self, ids):
        roots = []
        for name in ('robot', 'insertive_object', 'receptive_object'):
            value = self.env.scene[name].data.root_state_w[ids].clone()
            value[:, :3] -= self.env.scene.env_origins[ids]
            roots.append(value)
        return torch.cat([self.robot.data.joint_pos[ids], self.robot.data.joint_vel[ids], *roots], -1)

    def pre(self, actions):
        assert self.pending is None
        if self.stopped:
            return
        if self.quota.full():
            self.buffers.clear()
            self.stopped = True
            print('[T0_TRAJECTORY] global quota complete; full-trajectory buffers released', flush=True)
            return
        ids = torch.where(self.audit.active)[0]
        if not len(ids):
            return
        keys = torch.stack([ids, self.audit.episode[ids], self.audit.state[ids]], -1).cpu().tolist()
        new = [i for i, (env_id, episode, _) in enumerate(keys) if env_id not in self.buffers]
        if new:
            initial = self.states(ids[new]).detach().cpu()
            for i, start in zip(new, initial):
                env_id, episode, state_id = keys[i]
                self.buffers[env_id] = dict(episode=episode, state_id=state_id, initial=start.clone(),
                                           start_sim_step=int(self.env.common_step_counter), frames=[])
        for env_id, episode, state_id in keys:
            assert self.buffers[env_id]['episode'] == episode, 'reset without terminal recording'
        self.pending = dict(ids=ids, keys=keys, action=actions[ids].detach().clone(), substeps=[])

    def substep(self):
        if self.pending is None:
            return
        ids = self.pending['ids']
        value = torch.cat([self.robot.data.joint_pos_target[ids, :7],
                           self.grip.processed_actions[ids].sum(-1, keepdim=True),
                           self.robot.data.joint_effort_target[ids, :7]], -1)
        self.pending['substeps'].append(value.detach().clone())

    def post(self):
        if self.pending is None:
            return
        p, self.pending = self.pending, None
        assert len(p['substeps']) == 12, 'missing physics-substep commands'
        ids = p['ids']
        success = self.env.reward_manager.get_term_cfg('progress_context').func.success[ids]
        held = self.audit.tracker.given[ids]
        flags = torch.stack([success, held, self.env.reset_terminated[ids], self.env.reset_time_outs[ids]], -1)
        packed = torch.cat([self.states(ids), p['action'],
                            torch.stack(p['substeps'], 1).flatten(1), flags.float()], -1).detach().cpu()
        assert packed.shape[1] == 248 and torch.isfinite(packed).all()
        done = self.env.reset_buf[ids].cpu().tolist()
        for i, ((env_id, episode, state_id), terminal) in enumerate(zip(p['keys'], done)):
            buffer = self.buffers[env_id]
            buffer['frames'].append((packed, i))
            assert len(buffer['frames']) <= 640
            if not terminal:
                continue
            del self.buffers[env_id]
            task_success, hold = bool(packed[i, -4]), bool(packed[i, -3])
            if not (task_success or hold or self.diagnostic_all) or self.quota.full():
                continue
            data = torch.stack([chunk[row] for chunk, row in buffer['frames']])
            steps = len(data)
            meta = dict(self.metadata, episode_key=f'{self.metadata["run_dir"]}/{self.metadata["rank"]}/{env_id}/{episode}',
                        env_id=env_id, episode_id=episode, state_id=state_id, steps=steps,
                        start_sim_step=buffer['start_sim_step'], end_sim_step=int(self.env.common_step_counter),
                        task_success=task_success, held_lift_2s=hold, diagnostic_all=self.diagnostic_all)
            substeps = data[:, 64:244].reshape(steps, 12, 15).clone()
            payload = dict(metadata=meta, states=torch.cat([buffer['initial'][None], data[:, :57]]),
                           actions=data[:, 57:64].clone(), executed_actions=substeps[:, -1, :8].clone(),
                           substep_actions=substeps, flags=data[:, -4:].bool())
            if self.quota.save(payload):
                self.saved += 1
                print(f'[T0_TRAJECTORY] rank={self.metadata["rank"]} saved={self.saved} state={state_id} steps={steps}', flush=True)


def install_t0_trajectories(env, root, diagnostic_all=False):
    assert not hasattr(env, '_cupcake_t0_trajectories')
    recorder = T0TrajectoryRecorder(env, root, diagnostic_all)
    process = env.action_manager.process_action
    apply = env.action_manager.apply_action
    compute = env.reward_manager.compute
    def process_with_record(action):
        recorder.pre(action)
        return process(action)
    def apply_with_record():
        result = apply()
        recorder.substep()
        return result
    def compute_with_record(dt):
        result = compute(dt)  # Includes existing T0 summary/held tracker, before automatic reset.
        recorder.post()
        return result
    env.action_manager.process_action = process_with_record
    env.action_manager.apply_action = apply_with_record
    env.reward_manager.compute = compute_with_record
    env._cupcake_t0_trajectories = recorder
    return recorder


def verify_repair_with_trajectories(env, env_ids):
    from .cupcake_grasp_repair_cfg import verify_grasp_repair_contract
    verify_grasp_repair_contract(env, env_ids)
    root = os.environ['CUPCAKE_T0_TRAJECTORY_ROOT']
    install_t0_trajectories(env, root, diagnostic_all=False)
    print('[T0_TRAJECTORY] enabled: T0 success OR held-lift, full episode, global cap200', flush=True)
