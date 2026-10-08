# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static CupCake/StackCube DiffIK joint-target contract tests; no Isaac Sim launch."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FRANKA_DIR = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/config/franka"
)
ACTIONS = FRANKA_DIR / "actions.py"
RL_CFG = FRANKA_DIR / "rl_state_cfg.py"
REGISTRY = FRANKA_DIR / "__init__.py"
ACTION_TERM = FRANKA_DIR.parents[1] / "mdp/actions/task_space_actions.py"
TRAIN = REPO_ROOT / "UWLab/scripts/reinforcement_learning/rsl_rl/train.py"
BASE_RL_CFG = FRANKA_DIR.parent / "ur5e_robotiq_2f85/rl_state_cfg.py"
STACKCUBE_TRAIN = REPO_ROOT / "scripts/franka_kl_distill/train_stackcube_diffik_jointtarget.sh"
STACKCUBE_EVAL = REPO_ROOT / "scripts/franka_kl_distill/eval_stackcube_diffik_jointtarget.sh"
STACKCUBE_COLLECT = REPO_ROOT / "scripts/franka_kl_distill/collect_stackcube_diffik_jointtarget_demos.sh"
CUPCAKE_TRAIN = REPO_ROOT / "scripts/franka_kl_distill/train_cupcake_diffik_jointtarget.sh"
CUPCAKE_EVAL = REPO_ROOT / "scripts/franka_kl_distill/eval_cupcake_diffik_jointtarget.sh"
CUPCAKE_COLLECT = REPO_ROOT / "scripts/franka_kl_distill/collect_cupcake_diffik_jointtarget_demos.sh"
STACKCUBE_300X10 = REPO_ROOT / (
    "scripts/franka_kl_distill/collect_stackcube_diffik_jointtarget_teamhome_300x10.sh"
)
STACKCUBE_WILD_300X20 = REPO_ROOT / (
    "scripts/franka_kl_distill/collect_stackcube_diffik_jointtarget_teamhome_wild_300x20.sh"
)
PEG_TRAIN = REPO_ROOT / "scripts/franka_kl_distill/train_peg_diffik_jointtarget.sh"
PEG_EVAL = REPO_ROOT / "scripts/franka_kl_distill/eval_peg_diffik_jointtarget.sh"


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


class TestDiffIKActionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actions = _parse(ACTIONS)

    def test_policy_scale_and_ik_plant_match_frozen_contract(self):
        assignment = next(
            node for node in self.actions.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "FRANKA_FR3_RELATIVE_DIFF_IK_JOINT_TARGET"
                for target in node.targets
            )
        )
        call = assignment.value
        self.assertIsInstance(call, ast.Call)
        self.assertEqual(ast.literal_eval(_keyword(call, "scale_xyz_axisangle")),
                         (0.01, 0.01, 0.002, 0.02, 0.02, 0.2))
        self.assertEqual(ast.literal_eval(_keyword(call, "simulation_jacobian_point")), "link_origin")
        self.assertEqual(ast.literal_eval(_keyword(call, "ik_damping")), 0.05)
        self.assertEqual(ast.literal_eval(_keyword(call, "ik_step_scale")), 1.0)
        self.assertEqual(ast.literal_eval(_keyword(call, "joint_limit_margin")), 0.01)
        self.assertEqual(ast.unparse(_keyword(call, "max_joint_velocity")), "(1.5,) * 7")

    def test_action_keeps_guarded_gripper_and_six_dimensional_arm(self):
        cfg = _class(self.actions, "FrankaFr3GripperRelativeDiffIKJointTargetAction")
        self.assertEqual(ast.unparse(_assignment(cfg, "arm")), "FRANKA_FR3_RELATIVE_DIFF_IK_JOINT_TARGET")
        self.assertEqual(ast.unparse(_assignment(cfg, "gripper")), "GRASP_GUARDED_GRIPPER")

    def test_runtime_action_exposes_exact_applied_targets(self):
        source = ast.unparse(_class(_parse(ACTION_TERM), "RelCartesianDiffIKJointPositionAction"))
        for name in (
            "last_ik_joint_position_targets",
            "last_joint_position_targets",
            "last_applied_joint_position_targets",
            "last_ik_valid",
        ):
            self.assertIn(f"def {name}", source)
        self.assertIn("set_joint_position_target(self._joint_position_targets", source)


class TestDiffIKTaskAndTransferContract(unittest.TestCase):
    def test_stage1_and_stage2_use_joint_b3_events(self):
        module = _parse(RL_CFG)
        expected = {
            "FrankaFr3GripperRelCartesianDiffIKJointTargetTrainCfg": "JointB3TrainEventCfg",
            "FrankaFr3GripperRelCartesianDiffIKJointTargetEvalCfg": "JointB3EvalEventCfg",
            "FrankaFr3GripperRelCartesianDiffIKJointTargetFinetuneCfg": "JointFinetuneEventCfg",
            "FrankaFr3GripperRelCartesianDiffIKJointTargetFinetuneEvalCfg": "JointFinetuneEvalEventCfg",
        }
        for class_name, event_name in expected.items():
            self.assertEqual(ast.unparse(_assignment(_class(module, class_name), "events")), f"{event_name}()")

        state_source = ast.unparse(_class(module, "FrankaFr3GripperRelCartesianDiffIKJointTargetStateCfg"))
        self.assertIn("stiffness = 80.0", state_source)
        self.assertIn("damping = 4.0", state_source)
        self.assertIn("K5000/D50/E400", state_source)

    def test_all_four_diffik_task_ids_are_registered(self):
        source = REGISTRY.read_text()
        task_root = "OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-State"
        for suffix in ("-v0", "-Play-v0", "-Finetune-v0", "-Finetune-Play-v0"):
            self.assertIn(f'id="{task_root}{suffix}"', source)

    def test_all_four_peg_diffik_task_ids_are_registered(self):
        source = REGISTRY.read_text()
        task_root = "OmniReset-FrankaFr3Gripper-PegRelCartesianDiffIKJointTarget-State"
        for suffix in ("-v0", "-Play-v0", "-Finetune-v0", "-Finetune-Play-v0"):
            self.assertIn(f'id="{task_root}{suffix}"', source)

    def test_init_path_is_weight_only_and_resets_iteration(self):
        source = TRAIN.read_text()
        self.assertIn('"--init_path"', source)
        self.assertIn("runner.load(init_path, load_optimizer=False)", source)
        self.assertIn("runner.current_learning_iteration = 0", source)
        self.assertIn("--init_path and --resume_path are mutually exclusive", source)

    def test_resume_can_preserve_iteration_with_fresh_optimizer(self):
        source = TRAIN.read_text()
        self.assertIn('"--reset_optimizer_on_resume"', source)
        self.assertIn("--reset_optimizer_on_resume requires --resume_path", source)
        self.assertIn("runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer_on_resume)", source)
        self.assertIn("preserved checkpoint iteration", source)
        self.assertNotIn("runner.current_learning_iteration = 900", source)

    def test_resume_can_explicitly_drop_stage2_curriculum_for_fixed_task(self):
        source = TRAIN.read_text()
        self.assertIn('"--drop_checkpoint_curriculum_on_resume"', source)
        self.assertIn("--drop_checkpoint_curriculum_on_resume requires --resume_path", source)
        self.assertIn("requires a target task without ADR curriculum", source)

        launcher = STACKCUBE_TRAIN.read_text()
        self.assertIn("fixed_b3_resume)", launcher)
        self.assertIn("--drop_checkpoint_curriculum_on_resume", launcher)
        self.assertIn("agent.algorithm.schedule=\"${SCHEDULE}\"", launcher)

    def test_stackcube_launcher_uses_compatible_best_osc_expert(self):
        source = STACKCUBE_TRAIN.read_text()
        self.assertIn("model_31000.pt", source)
        self.assertIn("--init_path", source)
        self.assertIn("env.scene.insertive_object=cube", source)
        self.assertIn("env.scene.receptive_object=cube", source)
        self.assertIn('JOINT_MAX_VELOCITY="${JOINT_MAX_VELOCITY:-0.60}"', source)
        self.assertIn(
            "rigid_object_position_offsets.receptive_object=[0.0,0.0,${RECEPTIVE_OBJECT_Z_LIFT}]",
            source,
        )

    def test_stackcube_eval_matches_v06_lifted_training_contract(self):
        source = STACKCUBE_EVAL.read_text()
        self.assertIn('JOINT_MAX_VELOCITY="${JOINT_MAX_VELOCITY:-0.60}"', source)
        self.assertIn(
            "rigid_object_position_offsets.receptive_object=[0.0,0.0,${RECEPTIVE_OBJECT_Z_LIFT}]",
            source,
        )

    def test_stackcube_collection_records_post_limit_joint_targets(self):
        source = STACKCUBE_COLLECT.read_text()
        self.assertIn("--joint_target_bridge", source)
        self.assertIn("--joint_simulation_jacobian_point", source)
        self.assertIn("link_origin", source)
        self.assertIn("--joint_max_velocity", source)
        self.assertIn('JOINT_MAX_VELOCITY="${JOINT_MAX_VELOCITY:-0.60}"', source)
        self.assertIn(
            "rigid_object_position_offsets.receptive_object=[0.0,0.0,${RECEPTIVE_OBJECT_Z_LIFT}]",
            source,
        )

    def test_stackcube_300x10_collection_is_balanced_fixed_physics_and_team_home(self):
        source = STACKCUBE_300X10.read_text()
        self.assertIn("stackcube_jointtarget_factoryhome_300x10_state_ids.json", source)
        self.assertIn("--successful_repeats_per_state", source)
        self.assertIn("--stackcube_fixed_scene", source)
        self.assertIn("--preserve_object_face_materials", source)
        self.assertIn("stackcube_flip90_bc_scale_v2_train1440_teamhome_20260820", source)
        self.assertIn("OMNIRESET_REAL_OBJECT_TABLE_Z_OFFSET=0", source)
        self.assertIn("OMNIRESET_REAL_TABLE_COVER_COLLISION=0", source)
        self.assertIn("OMNIRESET_REAL_RECEPTIVE_OBJECT_Z_LIFT=0", source)
        self.assertIn("--post_reset_warmup_steps 1", source)
        self.assertIn("--image_size \"${IMAGE_SIZE}\"", source)

    def test_stackcube_wild_collection_changes_only_camera_dr_and_repeat_count(self):
        source = STACKCUBE_WILD_300X20.read_text()
        self.assertIn('SUCCESSFUL_REPEATS_PER_STATE="${SUCCESSFUL_REPEATS_PER_STATE:-20}"', source)
        self.assertIn('CAMERA_POSITION_JITTER_M="${CAMERA_POSITION_JITTER_M:-0.03}"', source)
        self.assertIn('CAMERA_ROTATION_JITTER_DEG="${CAMERA_ROTATION_JITTER_DEG:-15.0}"', source)
        self.assertIn("collect_stackcube_diffik_jointtarget_teamhome_300x10.sh", source)

    def test_shared_rl_table_collision_top_is_world_zero(self):
        source = BASE_RL_CFG.read_text()
        self.assertIn("pos=(0.4, 0.0, -0.868)", source)
        self.assertIn("collision top is at local z=0.868 m", source)

    def test_peg_launcher_uses_aligned_table0_osc_expert(self):
        source = PEG_TRAIN.read_text()
        self.assertIn("model_11350.pt", source)
        self.assertIn("--init_path", source)
        self.assertIn("env.scene.insertive_object=peg", source)
        self.assertIn("env.scene.receptive_object=peghole_big", source)
        self.assertIn("rigid_object_position_offsets.insertive_object=[0.0,0.0,0.013]", source)
        self.assertIn("rigid_object_position_offsets.receptive_object=[0.0,0.0,0.013]", source)
        self.assertIn("panda_hand.stiffness=1000.0", source)
        self.assertIn('JOINT_MAX_VELOCITY="${JOINT_MAX_VELOCITY:-0.60}"', source)
        self.assertIn("env.actions.arm.max_joint_velocity=${JOINT_MAX_VELOCITY_7}", source)
        self.assertIn('SCHEDULE="${SCHEDULE:-fixed}"', source)
        self.assertIn('SAVE_INTERVAL="${SAVE_INTERVAL:-25}"', source)

    def test_peg_diffik_config_preserves_non_arm_expert_contract(self):
        source = RL_CFG.read_text()
        peg_cfg = ast.unparse(
            _class(_parse(RL_CFG), "FrankaFr3GripperPegRelCartesianDiffIKJointTargetStateCfg")
        )
        self.assertIn("hand.stiffness = 1000.0", peg_cfg)
        self.assertIn("hand.damping = 14.0", peg_cfg)
        self.assertIn("hand.effort_limit_sim = 60.0", peg_cfg)
        self.assertIn("_B3_NOMINAL_FIXED_EVENTS", peg_cfg)
        self.assertIn("self.scene.table.init_state.pos = (0.4, 0.0, -0.868)", peg_cfg)
        self.assertIn("'insertive_object': (0.0, 0.0, 0.013)", peg_cfg)
        self.assertIn("'receptive_object': (0.0, 0.0, 0.013)", peg_cfg)
        self.assertIn("_B3_NOMINAL_FIXED_EVENTS = {", source)

    def test_peg_eval_matches_train_scene_contract(self):
        source = PEG_EVAL.read_text()
        self.assertIn("PegRelCartesianDiffIKJointTarget-State-Play-v0", source)
        self.assertIn("env.scene.receptive_object=peghole_big", source)
        self.assertIn("rigid_object_position_offsets.insertive_object=[0.0,0.0,0.013]", source)
        self.assertIn("rigid_object_position_offsets.receptive_object=[0.0,0.0,0.013]", source)
        self.assertIn("panda_hand.stiffness=1000.0", source)
        self.assertIn('JOINT_MAX_VELOCITY="${JOINT_MAX_VELOCITY:-0.60}"', source)
        self.assertIn("env.actions.arm.max_joint_velocity=${JOINT_MAX_VELOCITY_7}", source)

    def test_cupcake_train_eval_and_collection_match_v06_table0_contract(self):
        train = CUPCAKE_TRAIN.read_text()
        eval_source = CUPCAKE_EVAL.read_text()
        collect = CUPCAKE_COLLECT.read_text()
        for source in (train, eval_source, collect):
            self.assertIn('JOINT_MAX_VELOCITY="${JOINT_MAX_VELOCITY:-0.60}"', source)
            self.assertIn("env.scene.table.init_state.pos=[0.4,0.0,-0.868]", source)
        self.assertIn("env.actions.arm.max_joint_velocity=${JOINT_MAX_VELOCITY_7}", train)
        self.assertIn("env.actions.arm.max_joint_velocity=${JOINT_MAX_VELOCITY_7}", eval_source)
        self.assertIn('TASK_OVERRIDE="${TASK_OVERRIDE:-}"', eval_source)
        self.assertIn('STATE_IDS="${STATE_IDS:-}"', eval_source)
        self.assertIn("--state_indices_file \"${STATE_IDS}\"", eval_source)
        self.assertIn("--allow_critic_shape_mismatch", eval_source)
        self.assertIn('--joint_max_velocity "${JOINT_MAX_VELOCITY}"', collect)


if __name__ == "__main__":
    unittest.main()
