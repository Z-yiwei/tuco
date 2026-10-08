"""Resolve project runtime packages strictly inside this release."""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime/omnireset'

def environment(artifacts=None):
    assets=Path(artifacts or ROOT/'artifacts').resolve()
    paths=[ROOT/'src',ROOT/'third_party/rsl_rl',ROOT/'third_party/cupid',ROOT/'third_party/cupid/third_party/trak']
    for base in [RUNTIME/'UWLab/_isaaclab/IsaacLab/source',RUNTIME/'UWLab/source']:
        paths.extend(sorted(p for p in base.iterdir() if p.is_dir()))
    env=os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(map(str,paths)), PYTHONNOUSERSITE='1',
        OMNIRESET_ROOT=str(RUNTIME),TUCO_ARTIFACTS=str(assets),TUCO_ISAAC_ASSETS=str(assets/'native/Isaac'),
        UWLAB_CLOUD_ASSETS_DIR=str(assets/'cloud'),UWLAB_LOCAL_ASSETS_DIR=str(assets/'local_assets'),
        OMNIRESET_DATASET_DIR=str(assets/'datasets/OmniReset'),
        OMNIRESET_EXACT_WRIST_VERTICAL_FLIP='0',OMNI_KIT_ACCEPT_EULA='YES')
    return env
