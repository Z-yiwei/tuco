"""Independent original-T0 task-success quota; no changes to policy or physics."""
import fcntl
import json
import os
from pathlib import Path

import torch

from . import cupcake_t0_trajectory as original

SELECTION = 'original_t0_terminal_task_success_full_episode_v1'
CAP = 400


def validate_success_trajectory(payload):
    original.validate_trajectory(payload)
    meta = payload['metadata']
    assert meta['task_success'] is True and not meta['diagnostic_all']
    assert meta['reset_type'] == 'ObjectAnywhereEEAnywhere'
    assert isinstance(meta['state_id'], int) and 0 <= meta['state_id'] < 2500
    assert meta['selection'] == SELECTION
    assert meta['selector_sha256'] == original.sha(__file__)
    assert meta['end_sim_step'] - meta['start_sim_step'] == meta['steps']


class OriginalT0AuditView:
    """Filter recording only; never mutate the training audit or reset sampler."""
    def __init__(self, audit):
        self.audit = audit

    @property
    def active(self):
        return self.audit.active & (self.audit.state >= 0) & (self.audit.state < 2500)

    def __getattr__(self, name):
        return getattr(self.audit, name)


class SuccessTrajectoryQuota:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = dict(version=SELECTION, selection=SELECTION, max_trajectories=CAP,
            original_state_id_range_exclusive=[0, 2500],
            recorder_sha256=original.sha(original.__file__), selector_sha256=original.sha(__file__),
            termination='full reset to first original termination/timeout; terminal task success required',
            held_lift_required=False, unique_state_ids_required=False,
            completion='release recording buffers only; continue training unchanged')
        with (self.root / 'quota.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'recording_contract.json'
            if path.exists():
                assert json.loads(path.read_text()) == self.contract, 'recording contract drift'
            else:
                with path.open('x') as stream:
                    json.dump(self.contract, stream, indent=2)

    def full(self):
        return (self.root / 'COMPLETE.json').exists()

    def save(self, payload):
        meta = payload['metadata']
        if not meta['task_success'] or not 0 <= meta['state_id'] < 2500:
            return False
        validate_success_trajectory(payload)
        with (self.root / 'quota.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            index = self.root / 'manifest.jsonl'
            entries = [json.loads(x) for x in index.read_text().splitlines()] if index.exists() else []
            assert len(list(self.root.glob('trajectory_*.pt'))) == len(entries), 'orphan file; recover explicitly'
            if len(entries) >= CAP:
                assert self.full(), 'missing completion marker; recover explicitly'
                return False
            key = meta['episode_key']
            assert not any(row['episode_key'] == key for row in entries), 'duplicate episode'
            number = len(entries) + 1
            path = self.root / f'trajectory_{number:05d}.pt'
            temp = path.with_suffix('.pt.tmp')
            assert not path.exists() and not temp.exists(), 'preserve prior incomplete write'
            with temp.open('xb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            restored = torch.load(temp, map_location='cpu', weights_only=False)
            validate_success_trajectory(restored)
            assert restored['metadata'] == meta
            for name in ('states', 'actions', 'executed_actions', 'substep_actions', 'flags'):
                assert torch.equal(payload[name], restored[name]), f'write/read mismatch: {name}'
            digest = original.sha(temp)
            os.replace(temp, path)
            entry = dict(file=path.name, sha256=digest, episode_key=key, state_id=meta['state_id'],
                steps=meta['steps'], task_success=True, held_lift_2s=meta['held_lift_2s'])
            with index.open('a') as stream:
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            if number == CAP:
                marker = self.root / 'COMPLETE.tmp'
                with marker.open('x') as stream:
                    json.dump(dict(count=CAP, manifest_sha256=original.sha(index)), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(marker, self.root / 'COMPLETE.json')
            return True


def verify_success_recording_startup(env, env_ids):
    from .cupcake_trajectory_augmented_reset import verify_augmented_startup

    sidecar_path = Path(os.environ['CUPCAKE_SUCCESS400_CONTRACT']).resolve()
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar['selection'] == SELECTION and sidecar['cap'] == CAP
    for path, digest in sidecar['code_sha256'].items():
        assert original.sha(path) == digest, path
    assert original.sha(sidecar['checkpoint']) == sidecar['checkpoint_sha256']
    dataset = Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']).resolve()
    assert original.sha(dataset / 'manifest.json') == sidecar['dataset_manifest_sha256']
    assert original.sha(dataset / 'training_contract.json') == sidecar['training_contract_sha256']
    root = Path(os.environ['CUPCAKE_SUCCESS400_ROOT']).resolve()
    assert str(root) == sidecar['trajectory_root']
    for key in ('CUPCAKE_MIXED_HELD_ROOT', 'CUPCAKE_T0_TRAJECTORY_ROOT'):
        historical = Path(os.environ[key]).resolve()
        assert root != historical and (historical / 'COMPLETE.json').exists()

    # Execute the original sealed verifier intact, then replace only the recorder's
    # selection/quota before the first reset. Its audit and physics hooks stay intact.
    verify_augmented_startup(env, env_ids)
    recorder = env._cupcake_t0_trajectories
    assert not recorder.buffers and recorder.pending is None and not recorder.stopped
    assert recorder.saved == 0 and not recorder.diagnostic_all
    recorder.quota = SuccessTrajectoryQuota(root)
    recorder.audit = OriginalT0AuditView(env._cupcake_t0_audit)
    recorder.metadata.update(selection=SELECTION, selector_sha256=original.sha(__file__),
        trajectory_root=str(root), recording_launch_sha256=original.sha(sidecar_path),
        resume_checkpoint=sidecar['checkpoint'], resume_checkpoint_sha256=sidecar['checkpoint_sha256'],
        success_definition='terminal position error <20mm and abs(roll)+abs(pitch)<3deg; no grasp requirement',
        full_episode_definition='reset to first actual terminal; inherited random initial episode ages may shorten first episodes')
    result = dict(root=str(root), cap=CAP, selection=SELECTION, original_id_range_exclusive=[0,2500],
        recording_launch_sha256=original.sha(sidecar_path), training_actions_physics_rewards_sampler_unchanged=True)
    with (Path(env.cfg.log_dir) / f'success400_recording_rank{os.environ.get("RANK", "0")}.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print('[ORIGINAL_T0_SUCCESS400]', json.dumps(result), flush=True)
