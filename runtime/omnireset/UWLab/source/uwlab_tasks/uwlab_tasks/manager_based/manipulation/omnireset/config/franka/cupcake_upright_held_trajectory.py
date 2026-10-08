"""Opt-in held-only quota; the sealed original recorder and training stay intact."""
import fcntl
import json
import os
from pathlib import Path

import torch

from . import cupcake_t0_trajectory as original


SELECTION = 'upright_t0_acquisition_referenced_held_lift_2s'


def validate_held_trajectory(payload):
    original.validate_trajectory(payload)
    meta = payload['metadata']
    assert meta['held_lift_2s'] is True and not meta['diagnostic_all']
    assert meta['reset_type'] == 'ObjectAnywhereEEAnywhere'
    assert meta['selection'] == SELECTION
    assert meta['selector_sha256'] == original.sha(__file__)
    q = payload['states'][0, 34:38]  # insertive root wxyz, not robot root
    assert q[1:3].abs().max() < 1e-5 and abs(q.norm().item() - 1) < 1e-5
    assert meta['end_sim_step'] - meta['start_sim_step'] == meta['steps']
    # The held flag is latched, but the recorder must keep the terminal suffix.
    held = payload['flags'][:, 1]
    assert held.any() and held[int(held.nonzero()[0]) :].all()


class HeldTrajectoryQuota(original.TrajectoryQuota):
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = dict(
            version='cupcake_upright_t0_held_states_actions_v1', max_trajectories=200,
            selection=SELECTION, recorder_sha256=original.sha(original.__file__),
            selector_sha256=original.sha(__file__),
            termination='original episode termination or timeout; never stop on first held flag',
            completion='stop recording and release buffers only; do not stop or modify training',
        )
        with (self.root / 'quota.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'recording_contract.json'
            if path.exists():
                assert json.loads(path.read_text()) == self.contract, 'held recording contract drift'
            else:
                with path.open('x') as stream:
                    json.dump(self.contract, stream, indent=2)

    def save(self, payload):
        # The unchanged retrospective recorder also offers task-only episodes.
        # Reject them before quota accounting; never count placement as grasp.
        if not payload['metadata']['held_lift_2s']:
            return False
        validate_held_trajectory(payload)
        return super().save(payload)


def verify_upright_with_held_recording(env, env_ids):
    from .cupcake_upright_training_contract import verify_upright_startup

    sidecar_path = Path(os.environ['CUPCAKE_HELD_RECORDING_CONTRACT']).resolve()
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar['version'] == 'cupcake_upright_held_recording_launch_v1'
    assert sidecar['selection'] == SELECTION and sidecar['max_trajectories'] == 200
    for path, digest in sidecar['code_sha256'].items():
        assert original.sha(path) == digest, path
    dataset = Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']).resolve()
    assert original.sha(dataset / 'manifest.json') == sidecar['reset_manifest_sha256']
    assert original.sha(dataset / 'training_contract.json') == sidecar['training_contract_sha256']
    assert original.sha(sidecar['checkpoint']) == sidecar['checkpoint_sha256']
    root = Path(os.environ['CUPCAKE_T0_HELD_TRAJECTORY_ROOT']).resolve()
    assert str(root) == sidecar['trajectory_root']
    assert root != Path(os.environ['CUPCAKE_T0_TRAJECTORY_ROOT']).resolve()

    # Includes historical200 integrity, all native settings and T0 hooks.
    verify_upright_startup(env, env_ids)
    recorder = env._cupcake_t0_trajectories
    assert not recorder.buffers and recorder.pending is None and not recorder.stopped
    assert recorder.saved == 0 and not recorder.diagnostic_all
    recorder.quota = HeldTrajectoryQuota(root)
    recorder.metadata.update(
        reset_type='ObjectAnywhereEEAnywhere', selection=SELECTION,
        selector_sha256=original.sha(__file__), trajectory_root=str(root),
        recording_launch_sha256=original.sha(sidecar_path),
        resume_checkpoint=sidecar['checkpoint'], resume_checkpoint_sha256=sidecar['checkpoint_sha256'],
        hold_evidence='closed command + measured width 5..75mm; object/TCP each rise >=3cm from capture; relative stable for 20 control frames',
    )
    result = dict(root=str(root), cap=200, selection=SELECTION,
                  recording_launch_sha256=original.sha(sidecar_path), training_unchanged=True)
    with (Path(env.cfg.log_dir) / f'held_recording_rank{os.environ.get("RANK", "0")}.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print('[UPRIGHT_HELD_RECORDING]', json.dumps(result), flush=True)
