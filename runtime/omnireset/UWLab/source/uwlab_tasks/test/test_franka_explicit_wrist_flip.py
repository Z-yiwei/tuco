"""Run the actual camera switch block without importing Isaac Sim."""

import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
CFG = ROOT / 'UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/franka/data_collection_rgb_cfg.py'


def resolve(exact, value):
    source = CFG.read_text()
    start = source.index('_EXACT_CAMERA_INTRINSICS =')
    end = source.index('if _EXACT_CAMERA_INTRINSICS and (', start)
    env = {'OMNIRESET_EXACT_CAMERA_INTRINSICS': str(int(exact))}
    if value is not None:
        env['OMNIRESET_EXACT_WRIST_VERTICAL_FLIP'] = value
    scope = {'os': os}
    with patch.dict(os.environ, env, clear=True):
        exec(compile(source[start:end], str(CFG), 'exec'), scope)
    return scope['_EXACT_WRIST_VERTICAL_FLIP']


class TestExplicitWristFlip(unittest.TestCase):
    def test_exact_requires_explicit_choice(self):
        with self.assertRaisesRegex(ValueError, 'require explicit'):
            resolve(True, None)

    def test_no_flip_and_legacy_flip_are_preserved(self):
        self.assertFalse(resolve(True, '0'))
        self.assertTrue(resolve(True, '1'))

    def test_nonexact_never_flips(self):
        for value in (None, '0', '1'):
            self.assertFalse(resolve(False, value))

    def test_invalid_values_fail(self):
        for value in ('', 'true', 'false', '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve(True, value)

    def test_eval_wrappers_reject_missing_choice_before_isaac(self):
        env = {k: v for k, v in os.environ.items() if k != 'WRIST_VERTICAL_FLIP'}
        env.update(CKPT='/unused.ckpt', PREPROCESS='normal')
        for suffix in ('dp.sh', 'dp_paired.sh'):
            script = ROOT / f'scripts/franka_kl_distill/eval_cupcake_fr3_xy5_t0_v025_{suffix}'
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Set WRIST_VERTICAL_FLIP explicitly', result.stderr)


if __name__ == '__main__':
    unittest.main()
