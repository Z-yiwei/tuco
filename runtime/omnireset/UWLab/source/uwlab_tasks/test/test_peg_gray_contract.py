"""No-Isaac regressions for the permanent gray-only Peg appearance rule."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / 'scripts/franka_kl_distill'
spec = importlib.util.spec_from_file_location('peg_gray_contract', SCRIPTS / 'peg_gray_contract.py')
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def make_config(peg=True):
    scene = NS()
    events = NS()
    for name, asset in zip(contract.OBJECTS, ('Peg/peg.usd', 'PegHole/peg_hole_big.usd')):
        setattr(scene, name, NS(spawn=NS(usd_path='/assets/' + (asset if peg else 'Cube/cube.usd'))))
        setattr(events, f'randomize_{name}_appearance', NS(mode='reset', params={
            'asset_cfg': NS(name=name), 'mesh_names': [], 'texture_prob': 0.0,
            'colors': {c: (v, v) for c, v in zip('rgb', contract.PEG_GRAY_RGB)},
            'roughness_range': (0.25, 0.85),
        }))
    return NS(scene=scene, events=events)


class TestPegGrayContract(unittest.TestCase):
    def test_gray_passes_without_changing_roughness_or_event_mode(self):
        cfg = make_config()
        self.assertTrue(contract.assert_peg_gray_contract(cfg))
        event = cfg.events.randomize_insertive_object_appearance
        self.assertEqual(event.params['roughness_range'], (0.25, 0.85))
        self.assertEqual(event.mode, 'reset')

    def test_original_face_material_flag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'preserve_object_face_materials'):
            contract.assert_peg_gray_contract(make_config(), preserve_object_face_materials=True)

    def test_missing_event_is_rejected_for_either_object(self):
        for name in contract.OBJECTS:
            with self.subTest(name=name):
                cfg = make_config()
                setattr(cfg.events, f'randomize_{name}_appearance', None)
                with self.assertRaisesRegex(ValueError, 'must remain active'):
                    contract.assert_peg_gray_contract(cfg)

    def test_green_and_color_randomization_are_rejected(self):
        for name in contract.OBJECTS:
            for colors in ({'r': (0, 0), 'g': (1, 1), 'b': (0, 0)},
                           {'r': (0, 1), 'g': (0, 1), 'b': (0, 1)}):
                cfg = make_config()
                getattr(cfg.events, f'randomize_{name}_appearance').params['colors'] = colors
                with self.assertRaisesRegex(ValueError, 'colors'):
                    contract.assert_peg_gray_contract(cfg)

    def test_texture_and_partial_mesh_overrides_are_rejected(self):
        for key, value in (('texture_prob', 0.5), ('mesh_names', ['one_mesh'])):
            cfg = make_config()
            cfg.events.randomize_insertive_object_appearance.params[key] = value
            with self.assertRaises(ValueError):
                contract.assert_peg_gray_contract(cfg)

    def test_nonpeg_tasks_can_preserve_their_materials(self):
        self.assertFalse(contract.assert_peg_gray_contract(make_config(False), preserve_object_face_materials=True))

    def test_reset_type_detects_renamed_assets(self):
        cfg = make_config(False)
        cfg.events.reset_from_reset_states = NS(params={'reset_types': ['PegT0HomeQCurr_custom']})
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            contract.assert_peg_gray_contract(cfg, preserve_object_face_materials=True)

    def test_entrypoints_guard_final_configuration_before_gym_make(self):
        for path in (ROOT / 'scripts/eval_dp_cupid_image_random_fs.py', SCRIPTS / 'collect_vision_kl.py'):
            source = path.read_text()
            ast.parse(source)
            self.assertIn('preserve_object_face_materials=args_cli.preserve_object_face_materials', source)
            self.assertIn('if assert_peg_gray_contract(env_cfg):', source)
            self.assertLess(source.index('if assert_peg_gray_contract(env_cfg):'), source.index('env = gym.make('))

    def test_formal_eval_never_preserves_green_asset_materials(self):
        self.assertNotIn('--preserve_object_face_materials', (SCRIPTS / 'eval_peg_model4800_xy5cm_deltaq.sh').read_text())

    def test_future_source_contract_includes_the_guard(self):
        self.assertIn('scripts/franka_kl_distill/peg_gray_contract.py', (SCRIPTS / 'prepare_peg_model4800_xy5cm_fast84_run.sh').read_text())


if __name__ == '__main__':
    unittest.main()
