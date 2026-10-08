import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[4] / "scripts/franka_kl_distill/build_cupcake_t0_upright_fourpath.py"
SPEC = importlib.util.spec_from_file_location("upright_builder", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class UprightBuilderTest(unittest.TestCase):
    def test_selection_and_grasped_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pair = root / "source/Resets/CupCake__Plate"
            pair.mkdir(parents=True)
            poses = [torch.tensor([0., 0., 0., math.cos(a / 2), math.sin(a / 2), 0., 0.])
                     for a in [0., math.pi / 2, math.pi, math.radians(2)]]
            original = {"initial_state": {"rigid_object": {"insertive_object": {"root_pose": poses}},
                                          "articulation": {"robot": {"joint_position": [torch.ones(9) * i for i in range(4)]}}}}
            for name in builder.RESET_TYPES:
                torch.save(original, pair / f"resets_{name}.pt")
            manifest = builder.build(root / "source", root / "output")
            self.assertEqual(manifest["source_indices"], [0, 3])
            for name in builder.RESET_TYPES[1:]:
                file = f"resets_{name}.pt"
                self.assertEqual((pair / file).read_bytes(), (root / "output/Resets/CupCake__Plate" / file).read_bytes())
            filtered = torch.load(root / "output/Resets/CupCake__Plate/resets_ObjectAnywhereEEAnywhere.pt", weights_only=False)
            self.assertTrue(torch.equal(filtered["initial_state"]["articulation"]["robot"]["joint_position"][1], torch.ones(9) * 3))
            with self.assertRaises(FileExistsError):
                builder.build(root / "source", root / "output")


if __name__ == "__main__":
    unittest.main()
