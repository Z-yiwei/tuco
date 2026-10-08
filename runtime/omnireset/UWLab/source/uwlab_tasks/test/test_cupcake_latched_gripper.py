"""CPU state-machine tests; the Isaac action parent is exercised by sim audits."""

import ast
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/franka/latched_gripper_action.py"


class BinaryParent:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = 4, "cpu"
        self._open_command, self._close_command = torch.full((2,), 0.04), torch.zeros(2)
        self._raw_actions, self._processed_actions = torch.zeros(4, 1), torch.zeros(4, 2)

    def reset(self, env_ids=None):
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self):
        pass


tree = ast.parse(SOURCE.read_text())
selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in (
    "LatchedBinaryGripperAction", "previous_actions_with_gripper_latch",
)]
scope = {"torch": torch, "Sequence": Sequence, "BinaryJointPositionAction": BinaryParent}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), scope)
Action = scope["LatchedBinaryGripperAction"]


class TestLatchedGripper(unittest.TestCase):
    def make_action(self, grasped=False, names=None):
        names = names or ["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"]
        cfg = SimpleNamespace(
            clip=None, close_grasped_resets=grasped, reset_event_name="reset_from_reset_states",
            grasped_reset_types=tuple(names[1:]), open_reset_types=(names[0],),
        )
        self.reset_term = SimpleNamespace(params={"reset_types": names}, func=SimpleNamespace(task_id=torch.arange(4)))
        env = SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=lambda _: self.reset_term))
        action = Action(cfg, env)
        action.reset()
        return action

    def test_open_until_first_negative_then_no_reopen(self):
        action = self.make_action()
        action.process_actions(torch.zeros(4, 1))
        self.assertTrue(torch.all(action._processed_actions == 0.04))
        action.process_actions(torch.tensor([[-0.1], [0.0], [1.0], [-3.0]]))
        self.assertEqual(action.latched_closed[:, 0].tolist(), [True, False, False, True])
        action.process_actions(torch.full((4, 1), 999.0))
        self.assertEqual(action.latched_closed[:, 0].tolist(), [True, False, False, True])
        self.assertTrue(torch.all(action._raw_actions == 999.0))
        self.assertTrue(torch.all(action._processed_actions[[0, 3]] == 0.0))

    def test_bool_false_closes_matching_isaac_parent(self):
        action = self.make_action()
        action.process_actions(torch.tensor([[False], [True], [False], [True]]))
        self.assertEqual(action.latched_closed[:, 0].tolist(), [True, False, True, False])

    def test_partial_reset_does_not_unlock_other_envs(self):
        action = self.make_action()
        action.process_actions(-torch.ones(4, 1))
        action.reset(torch.tensor([1, 3]))
        self.assertEqual(action.latched_closed[:, 0].tolist(), [True, False, True, False])
        self.assertTrue(torch.all(action._processed_actions[[1, 3]] == 0.04))

    def test_grasped_path_is_closed_before_first_positive_action(self):
        action = self.make_action(grasped=True)
        self.assertEqual(action.latched_closed[:, 0].tolist(), [False, True, True, True])
        action.process_actions(torch.ones(4, 1))
        self.assertEqual(action.latched_closed[:, 0].tolist(), [False, True, True, True])

    def test_reset_uses_new_path_not_old_latch(self):
        action = self.make_action(grasped=True)
        self.reset_term.func.task_id[1] = 0
        self.reset_term.func.task_id[0] = 2
        action.reset(torch.tensor([0, 1]))
        self.assertEqual(action.latched_closed[:, 0].tolist(), [True, False, True, True])

    def test_unknown_path_is_rejected(self):
        action = self.make_action(grasped=True)
        self.reset_term.params["reset_types"] = ["unknown"] * 4
        with self.assertRaises(ValueError):
            action.reset()

    def test_last_applied_targets_survive_reset(self):
        action = self.make_action()
        action.process_actions(-torch.ones(4, 1))
        action.apply_actions()
        action.reset()
        self.assertTrue(torch.all(action._processed_actions == 0.04))
        self.assertTrue(torch.all(action.last_applied_actions == 0.0))

    def test_observation_exposes_latch_without_mutating_ppo_action(self):
        action = self.make_action(grasped=True)
        raw = torch.arange(28, dtype=torch.float).reshape(4, 7)
        env = SimpleNamespace(action_manager=SimpleNamespace(action=raw, get_term=lambda _: action))
        obs = scope["previous_actions_with_gripper_latch"](env)
        self.assertEqual(obs[:, 6].tolist(), [1.0, -1.0, -1.0, -1.0])
        self.assertTrue(torch.equal(obs[:, :6], raw[:, :6]))
        self.assertTrue(torch.equal(raw, torch.arange(28, dtype=torch.float).reshape(4, 7)))


if __name__ == "__main__":
    unittest.main()
