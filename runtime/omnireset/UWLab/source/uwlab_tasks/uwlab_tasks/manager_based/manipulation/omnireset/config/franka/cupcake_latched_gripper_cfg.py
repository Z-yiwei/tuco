"""Independent CupCake PPO contract with one policy-triggered gripper closure."""

import json
import hashlib
import math
import os
from pathlib import Path

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .cupcake_policy_gripper_cfg import (
    CupCakePolicyGripperEvents,
    CupCakePolicyGripperTrainCfg,
    PLATE_SCALE,
    RESET_TYPES,
)
from .latched_gripper_action import (
    LatchedBinaryGripperAction,
    LatchedBinaryGripperActionCfg,
    previous_actions_with_gripper_latch,
)


def verify_latched_gripper_contract(env, env_ids):
    command = env.command_manager.get_term("task_command")
    command.success_position_threshold = 0.020
    command.success_orientation_threshold = math.radians(3.0)
    grip = env.action_manager.get_term("gripper")
    arm = env.action_manager.get_term("arm")
    assert type(grip) is LatchedBinaryGripperAction
    assert grip.cfg.close_grasped_resets
    assert torch.equal(grip._close_command, torch.zeros_like(grip._close_command))
    assert torch.allclose(grip._open_command, torch.full_like(grip._open_command, 0.04))
    assert env.action_manager.total_action_dim == 7
    assert tuple(arm.cfg.max_joint_velocity) == (0.25,) * 7
    assert math.isclose(env.step_dt, 0.1) and env.max_episode_length == 640
    assert not env.curriculum_manager.active_terms
    assert tuple(env.scene["receptive_object"].cfg.spawn.scale) == (PLATE_SCALE,) * 3
    expected_insertive_scale = float(os.environ.get("CUPCAKE_EXPECTED_INSERTIVE_SCALE", "1.0"))
    assert tuple(env.scene["insertive_object"].cfg.spawn.scale) == (expected_insertive_scale,) * 3
    assert env.cfg.scene.table.init_state.pos[2] == -0.868
    assert "FrankaFR3/fr3.usd" in env.scene["robot"].cfg.spawn.usd_path
    assert env.observation_manager.group_obs_dim["policy"] == (200,)
    assert env.observation_manager.group_obs_dim["critic"] == (168,)
    assert env.cfg.observations.policy.prev_actions.func is previous_actions_with_gripper_latch
    assert env.cfg.observations.critic.prev_actions.func is previous_actions_with_gripper_latch
    reset = env.cfg.events.reset_from_reset_states.params
    assert reset["reset_types"] == RESET_TYPES and reset["probs"] == [0.25] * 4
    assert tuple(env.cfg.events.randomize_arm_sysid.params["scale_range"]) == (1.0, 1.0)

    saved = {name: getattr(grip, name).clone() for name in (
        "_latched_closed", "_raw_actions", "_processed_actions",
    )}
    try:
        grip._latched_closed.zero_()
        for sign, target in ((1.0, 0.04), (-1.0, 0.0), (1.0, 0.0)):
            grip.process_actions(torch.full_like(grip.raw_actions, sign))
            assert torch.allclose(grip.processed_actions, torch.full_like(grip.processed_actions, target))
    finally:
        for name, value in saved.items():
            getattr(grip, name)[:] = value
    contract = {
        "version": "cupcake_fr3_latched_gripper_fourpath_v025h64_20260906",
        "num_envs": env.num_envs, "action_dim": 7, "actor_obs_dim": 200, "critic_obs_dim": 168,
        "gripper_action_class": type(grip).__name__, "gripper_guard": False,
        "T0": "initial open; first negative policy sample closes until reset",
        "T1_T2_T3": "closed from reset, cannot reopen within episode",
        "previous_action_gripper_slot": "executed latch sign, not raw Gaussian sample",
        "raw_PPO_action_modified": False, "latch_transition_test_passed": True,
        "joint_velocity_cap_rad_s": 0.25, "dt_s": env.step_dt, "episode_steps": 640,
        "plate_scale": PLATE_SCALE, "reset_types": reset["reset_types"],
        "cupcake_scale": expected_insertive_scale,
        "reset_probabilities": reset["probs"], "adr": False,
        "physics_and_rewards": "inherited unchanged CupCakePolicyGripperTrainCfg",
        "success": "20mm / abs(roll)+abs(pitch)<3deg; no release requirement added",
    }
    upright_root = os.environ.get("CUPCAKE_T0_UPRIGHT_DATASET_ROOT")
    reset_root = os.environ.get("CUPCAKE_RESET_DATASET_ROOT")
    assert not (upright_root and reset_root), "select exactly one custom CupCake reset root"
    if reset_root:
        assert Path(reset["dataset_dir"]).resolve() == Path(reset_root).resolve()
        manifest = json.loads((Path(reset_root) / "manifest.json").read_text())
        assert manifest["version"] == "cupcake_half_omnireset_v1"
        assert manifest["cupcake_uniform_scale"] == expected_insertive_scale == 0.5
        pair = manifest["pair"]
        assert pair == "CupCakeHalf__Plate"
        for name, expected_hash in manifest["reset_sha256"].items():
            path = Path(reset_root) / "Resets" / pair / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        contract["reset_dataset"] = {
            "root": reset_root,
            "pair": pair,
            "manifest_version": manifest["version"],
            "hashes_verified": True,
        }
    if upright_root:
        assert Path(reset["dataset_dir"]).resolve() == Path(upright_root).resolve()
        manifest = json.loads((Path(upright_root) / "manifest.json").read_text())
        assert manifest["version"] == "cupcake_t0_upright_fourpath_20260909"
        for name, hashes in manifest["files"].items():
            path = Path(upright_root) / "Resets/CupCake__Plate" / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes["output_sha256"]
            if name != "resets_ObjectAnywhereEEAnywhere.pt":
                assert hashes["source_sha256"] == hashes["output_sha256"]
        manager = env.event_manager.get_term_cfg("reset_from_reset_states").func
        q = manager.datasets[0]["initial_state"]["rigid_object"]["insertive_object"]["root_pose"][:, 3:7]
        assert len(q) == manifest["selected_count"]
        assert torch.all(1 - 2 * (q[:, 1].square() + q[:, 2].square()) >= math.cos(math.radians(3.0)))
        contract["t0_upright_reset"] = {
            "dataset_root": upright_root, "count": len(q), "max_tilt_deg": 3.0,
            "T1_T2_T3_byte_identical": True, "whole_state_filter_only": True,
        }
    print("[LATCHED_GRIPPER_CONTRACT] " + json.dumps(contract, sort_keys=True), flush=True)
    if env.cfg.log_dir:
        path = Path(env.cfg.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"latched_gripper_contract_rank{os.environ.get('RANK', '0')}.json").write_text(
            json.dumps(contract, indent=2) + "\n"
        )


@configclass
class CupCakeLatchedGripperEvents(CupCakePolicyGripperEvents):
    policy_gripper_contract = EventTerm(func=verify_latched_gripper_contract, mode="startup")


@configclass
class CupCakeLatchedGripperTrainCfg(CupCakePolicyGripperTrainCfg):
    events: CupCakeLatchedGripperEvents = CupCakeLatchedGripperEvents()

    def __post_init__(self):
        super().__post_init__()
        self.actions.gripper = LatchedBinaryGripperActionCfg(
            asset_name="robot", joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
            close_grasped_resets=True,
        )
        self.observations.policy.prev_actions.func = previous_actions_with_gripper_latch
        self.observations.critic.prev_actions.func = previous_actions_with_gripper_latch
