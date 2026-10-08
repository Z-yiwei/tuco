# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static Franka joint-ADR contract tests that do not import or launch Isaac Sim."""

import ast
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
FRANKA_CFG = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/config/franka/rl_state_cfg.py"
)
FRANKA_REGISTRY = FRANKA_CFG.with_name("__init__.py")
OMNIRESET_EVENTS = FRANKA_CFG.parents[2] / "mdp/events.py"
TRAIN_SCRIPT = REPO_ROOT / "UWLab/scripts/reinforcement_learning/rsl_rl/train.py"
B3_METADATA = REPO_ROOT / "data/sysid/metadata_franka_b3.yaml"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assignment(class_node: ast.ClassDef, name: str) -> ast.AST:
    for node in class_node.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    raise AssertionError(f"{class_node.name}.{name} is not assigned")


def _keyword(call: ast.Call, name: str) -> ast.AST:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _dict_value(mapping: ast.Dict, name: str) -> ast.AST:
    for key, value in zip(mapping.keys, mapping.values):
        if isinstance(key, ast.Constant) and key.value == name:
            return value
    raise AssertionError(f"dictionary key is missing: {name}")


def _event_params(module: ast.Module, class_name: str) -> tuple[str, ast.Dict]:
    event_call = _assignment(_class(module, class_name), "randomize_arm_sysid")
    assert isinstance(event_call, ast.Call)
    func = _keyword(event_call, "func")
    assert isinstance(func, ast.Attribute)
    params = _keyword(event_call, "params")
    assert isinstance(params, ast.Dict)
    return func.attr, params


def _literal(mapping: ast.Dict, name: str) -> Any:
    return ast.literal_eval(_dict_value(mapping, name))


class TestFrankaJointAdrConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _parse(FRANKA_CFG)

    def test_b3_metadata_has_seven_identified_joint_entries(self):
        sysid = yaml.safe_load(B3_METADATA.read_text())["sysid"]
        for key in ("armature", "static_friction", "dynamic_ratio", "viscous_friction"):
            self.assertEqual(len(sysid[key]), 7, key)

    def test_stage1_train_and_play_are_exact_b3_nominal(self):
        for class_name in ("JointB3TrainEventCfg", "JointB3EvalEventCfg"):
            func, params = _event_params(self.module, class_name)
            self.assertEqual(func, "randomize_arm_from_sysid_fixed")
            self.assertEqual(_literal(params, "scale_range"), (1.0, 1.0))
            self.assertEqual(_literal(params, "delay_range"), (0, 0))
            metadata = _dict_value(params, "sysid_metadata_path")
            self.assertIsInstance(metadata, ast.Name)
            self.assertEqual(metadata.id, "FRANKA_B3_SYSID_METADATA_PATH")

    def test_stage2_is_nominal_origin_arm_only_adr(self):
        func, params = _event_params(self.module, "JointFinetuneEventCfg")
        self.assertEqual(func, "randomize_arm_from_sysid")
        self.assertEqual(_literal(params, "scale_range"), (0.8, 1.2))
        self.assertEqual(_literal(params, "delay_range"), (0, 0))
        self.assertEqual(_literal(params, "initial_scale_progress"), 0.0)
        self.assertTrue(_literal(params, "interpolate_from_sysid_nominal"))

        curriculum = _class(self.module, "JointFinetuneCurriculumsCfg")
        self.assertRaises(AssertionError, _assignment, curriculum, "action_scale")
        adr_call = _assignment(curriculum, "adr_sysid")
        self.assertIsInstance(adr_call, ast.Call)
        adr_params = _keyword(adr_call, "params")
        self.assertIsInstance(adr_params, ast.Dict)
        self.assertEqual(_literal(adr_params, "event_term_names"), ["randomize_arm_sysid"])
        self.assertEqual(_literal(adr_params, "success_threshold_up"), 0.92)
        self.assertEqual(_literal(adr_params, "warmup_success_threshold"), 0.92)

    def test_stage2_play_is_fixed_at_full_range(self):
        func, params = _event_params(self.module, "JointFinetuneEvalEventCfg")
        self.assertEqual(func, "randomize_arm_from_sysid_fixed")
        self.assertEqual(_literal(params, "scale_range"), (0.8, 1.2))
        self.assertEqual(_literal(params, "delay_range"), (0, 0))

    def test_concrete_joint_tasks_use_the_b3_event_contracts(self):
        expected = {
            "FrankaFr3GripperRelativeJointTargetTrainCfg": "JointB3TrainEventCfg",
            "FrankaFr3GripperRelativeJointTargetEvalCfg": "JointB3EvalEventCfg",
            "FrankaFr3GripperRelativeJointTargetFinetuneCfg": "JointFinetuneEventCfg",
            "FrankaFr3GripperRelativeJointTargetFinetuneEvalCfg": "JointFinetuneEvalEventCfg",
        }
        for class_name, event_class_name in expected.items():
            event_call = _assignment(_class(self.module, class_name), "events")
            self.assertIsInstance(event_call, ast.Call)
            self.assertIsInstance(event_call.func, ast.Name)
            self.assertEqual(event_call.func.id, event_class_name)

    def test_sysid_event_implements_nominal_origin_interpolation(self):
        event_module = _parse(OMNIRESET_EVENTS)
        event_class = _class(event_module, "randomize_arm_from_sysid")
        source = ast.unparse(event_class)
        self.assertIn("cfg.params.get('interpolate_from_sysid_nominal', False)", source)
        self.assertIn("if self.interpolate_from_sysid_nominal", source)
        self.assertIn("base_arm = torch.as_tensor(self.armature", source)

    def test_joint_stage2_gym_ids_are_registered(self):
        registry = _parse(FRANKA_REGISTRY)
        registrations: dict[str, str] = {}
        for node in ast.walk(registry):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "register":
                continue
            task_id = ast.literal_eval(_keyword(node, "id"))
            kwargs = _keyword(node, "kwargs")
            assert isinstance(kwargs, ast.Dict)
            entry_point = ast.unparse(_dict_value(kwargs, "env_cfg_entry_point"))
            registrations[task_id] = entry_point

        self.assertIn("OmniReset-FrankaFr3Gripper-RelJointTarget-State-Finetune-v0", registrations)
        self.assertIn("RelativeJointTargetFinetuneCfg", registrations[
            "OmniReset-FrankaFr3Gripper-RelJointTarget-State-Finetune-v0"
        ])
        self.assertIn("OmniReset-FrankaFr3Gripper-RelJointTarget-State-Finetune-Play-v0", registrations)
        self.assertIn("RelativeJointTargetFinetuneEvalCfg", registrations[
            "OmniReset-FrankaFr3Gripper-RelJointTarget-State-Finetune-Play-v0"
        ])


class TestCurriculumCheckpointCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = _parse(TRAIN_SCRIPT)
        selected = [
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_file_md5", "apply_stage2_curriculum_state"}
        ]
        namespace = {"Any": Any, "hashlib": hashlib, "os": os}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(TRAIN_SCRIPT), "exec"), namespace)
        cls.apply_state = staticmethod(namespace["apply_stage2_curriculum_state"])

    def setUp(self):
        handle = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        handle.write(b"test-b3-metadata")
        handle.close()
        self.addCleanup(lambda: os.unlink(handle.name))
        self.metadata_path = handle.name
        self.metadata_md5 = hashlib.md5(b"test-b3-metadata").hexdigest()

    def _env_cfg(self, event_names: list[str], with_action_scale: bool) -> Any:
        arm = SimpleNamespace(params={
            "sysid_metadata_path": self.metadata_path,
            "initial_scale_progress": 0.0,
        })
        events = SimpleNamespace(randomize_arm_sysid=arm)
        if "randomize_osc_gains" in event_names:
            events.randomize_osc_gains = SimpleNamespace(params={"initial_scale_progress": 0.0})
        curriculum = SimpleNamespace(adr_sysid=SimpleNamespace(params={
            "event_term_names": event_names,
            "initial_scale_progress": 0.0,
            "initial_warmed_up": None,
        }))
        if with_action_scale:
            curriculum.action_scale = SimpleNamespace(params={"initial_progress": 0.0})
        return SimpleNamespace(events=events, curriculum=curriculum)

    def _state(self, event_names: list[str], with_action_scale: bool) -> dict[str, Any]:
        state = {
            "version": 1,
            "adr_sysid_scale_progress": 0.37,
            "adr_sysid_warmed_up": True,
            "adr_event_term_names": event_names,
            "sysid_metadata_md5": self.metadata_md5,
        }
        if with_action_scale:
            state["action_scale_progress"] = 0.61
        return state

    def test_restores_joint_only_adr_without_action_scale(self):
        env_cfg = self._env_cfg(["randomize_arm_sysid"], with_action_scale=False)
        restored = self.apply_state(env_cfg, self._state(["randomize_arm_sysid"], False))
        self.assertTrue(restored)
        self.assertEqual(env_cfg.events.randomize_arm_sysid.params["initial_scale_progress"], 0.37)
        self.assertEqual(env_cfg.curriculum.adr_sysid.params["initial_scale_progress"], 0.37)
        self.assertTrue(env_cfg.curriculum.adr_sysid.params["initial_warmed_up"])

    def test_keeps_legacy_cartesian_action_scale_restore(self):
        event_names = ["randomize_osc_gains", "randomize_arm_sysid"]
        env_cfg = self._env_cfg(event_names, with_action_scale=True)
        restored = self.apply_state(env_cfg, self._state(event_names, True))
        self.assertTrue(restored)
        self.assertEqual(env_cfg.events.randomize_osc_gains.params["initial_scale_progress"], 0.37)
        self.assertEqual(env_cfg.curriculum.action_scale.params["initial_progress"], 0.61)

    def test_rejects_cross_controller_curriculum_shape(self):
        env_cfg = self._env_cfg(["randomize_arm_sysid"], with_action_scale=False)
        with self.assertRaisesRegex(ValueError, "different Stage-2 curriculum shape"):
            self.apply_state(env_cfg, self._state(["randomize_arm_sysid"], True))


if __name__ == "__main__":
    unittest.main()
