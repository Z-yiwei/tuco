# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract tests for the continuous StackCube XY5 t0 RL task."""

import ast
import math
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FRANKA_DIR = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/config/franka"
)
RL_CFG = FRANKA_DIR / "rl_state_cfg.py"
REGISTRY = FRANKA_DIR / "__init__.py"
EVENTS = FRANKA_DIR.parents[1] / "mdp/events.py"
EVAL_RL = REPO_ROOT / "scripts/eval_rl_sr.py"
BASE_TRAIN = REPO_ROOT / "scripts/franka_kl_distill/train_stackcube_diffik_jointtarget.sh"
BASE_EVAL = REPO_ROOT / "scripts/franka_kl_distill/eval_stackcube_diffik_jointtarget.sh"
XY5_TRAIN = REPO_ROOT / "scripts/franka_kl_distill/train_stackcube_xy5_t0_diffik_jointtarget.sh"
XY5_EVAL = REPO_ROOT / "scripts/franka_kl_distill/eval_stackcube_xy5_t0_diffik_jointtarget.sh"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assignment(module: ast.Module, name: str) -> ast.AST:
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    raise AssertionError(f"global assignment {name} is missing")


def _evaluate_constant(module: ast.Module, name: str):
    expression = ast.Expression(body=_assignment(module, name))
    return eval(compile(expression, str(RL_CFG), "eval"), {"math": math})


class TestStackCubeXY5T0OnlineResetContract(unittest.TestCase):
    def test_distribution_is_continuous_and_matches_frozen_bounds(self):
        module = _parse(RL_CFG)
        insertive = _evaluate_constant(module, "STACKCUBE_XY5_T0_INSERTIVE_POSE_RANGES")
        receptive = _evaluate_constant(module, "STACKCUBE_XY5_T0_RECEPTIVE_POSE_RANGES")
        team_home = _evaluate_constant(module, "STACKCUBE_XY5_T0_TEAM_HOME")

        self.assertEqual(insertive["x"], (0.343470, 0.393394))
        self.assertEqual(insertive["y"], (-0.153059, -0.103080))
        self.assertEqual(insertive["z"], (0.020, 0.020))
        self.assertEqual(receptive["x"], (0.440422, 0.490394))
        self.assertEqual(receptive["y"], (-0.082498, -0.032517))
        self.assertEqual(receptive["z"], (0.020, 0.020))
        self.assertEqual(insertive["yaw"], (-math.pi, math.pi))
        self.assertEqual(receptive["yaw"], (-math.pi, math.pi))
        self.assertEqual(len(team_home), 7)

        event_source = ast.unparse(_class(module, "JointB3XY5T0EventCfg"))
        self.assertIn("task_mdp.StackCubeXY5T0OnlineReset", event_source)
        self.assertIn("STACKCUBE_XY5_T0_HOME_JOINT_JITTER_RAD", event_source)
        self.assertNotIn("MultiResetManager", event_source)
        self.assertNotIn("reset_types", event_source)
        self.assertNotIn("dataset_dir", event_source)

    def test_online_reset_is_single_task_and_does_not_modify_multi_reset(self):
        module = _parse(EVENTS)
        online_source = ast.unparse(_class(module, "StackCubeXY5T0OnlineReset"))
        multi_source = ast.unparse(_class(module, "MultiResetManager"))
        self.assertIn("math_utils.sample_uniform", online_source)
        self.assertIn("self.num_tasks = 1", online_source)
        self.assertIn("self.state_id", online_source)
        self.assertIn("self.state_id[env_ids] = -1", online_source)
        self.assertIn("soft_joint_pos_limits", online_source)
        self.assertNotIn("articulation_joint_position_offset_ranges", multi_source)

    def test_task_ids_and_launchers_freeze_v020_h20_fixed_b3_contract(self):
        registry = REGISTRY.read_text()
        for task_id in (
            "OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-XY5T0-State-v0",
            "OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-XY5T0-State-Play-v0",
        ):
            self.assertIn(f'id="{task_id}"', registry)

        train = XY5_TRAIN.read_text()
        evaluation = XY5_EVAL.read_text()
        self.assertIn("STAGE=fixed_b3_resume", train)
        self.assertIn("JOINT_MAX_VELOCITY=0.20", train)
        self.assertIn("EPISODE_LENGTH_S=20.0", train)
        self.assertIn("INJECT_RECEPTIVE_OBJECT_Z_LIFT=0", train)
        self.assertIn("PHYSICS_MODE=fixed_b3", evaluation)
        self.assertIn("JOINT_MAX_VELOCITY=0.20", evaluation)
        self.assertIn("EPISODE_LENGTH_S=20.0", evaluation)
        self.assertIn("ONE_EPISODE_PER_ENV=1", evaluation)
        self.assertIn("does not accept STATE_IDS", evaluation)
        self.assertNotIn("candidate1440", train + evaluation)

    def test_base_launchers_can_skip_legacy_lift_and_eval_rejects_state_ids(self):
        for source in (BASE_TRAIN.read_text(), BASE_EVAL.read_text()):
            self.assertIn("INJECT_RECEPTIVE_OBJECT_Z_LIFT", source)
            self.assertIn("RESET_OVERRIDES", source)
        eval_source = EVAL_RL.read_text()
        self.assertIn('uses_discrete_reset_states = "reset_types" in reset_params', eval_source)
        self.assertIn("not valid for an online continuous reset task", eval_source)
        self.assertIn("reset_sampling=", eval_source)
        self.assertIn("--stochastic_actions", eval_source)
        self.assertIn("--stochastic_std_scale", eval_source)
        self.assertIn("log_std.add_(math.log(args_cli.stochastic_std_scale))", eval_source)
        self.assertIn("policy_action_mode=", eval_source)
        self.assertIn("stochastic_std_scale=", eval_source)
        self.assertIn("STOCHASTIC_ACTIONS", BASE_EVAL.read_text())


if __name__ == "__main__":
    unittest.main()
