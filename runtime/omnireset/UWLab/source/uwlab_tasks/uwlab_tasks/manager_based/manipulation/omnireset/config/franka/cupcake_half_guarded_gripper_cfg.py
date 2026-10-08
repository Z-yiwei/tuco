"""Half-size CupCake four-path PPO with a deterministic grasp guard."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .research3_actions import FR3_GUARDED_GRIPPER
from .cupcake_policy_gripper_cfg import (
    CupCakePolicyGripperEvents,
    CupCakePolicyGripperTrainCfg,
    PLATE_SCALE,
    RESET_TYPES,
)
from .grasp_guarded_action import GraspGuardedBinaryGripperAction


CUPCAKE_SCALE = 0.5
JOINT_VELOCITY_CAP_RAD_S = 0.2


def verify_half_guarded_contract(env, env_ids):
    command = env.command_manager.get_term("task_command")
    command.success_position_threshold = 0.020
    command.success_orientation_threshold = math.radians(3.0)
    grip = env.action_manager.get_term("gripper")
    arm = env.action_manager.get_term("arm")
    assert type(grip) is GraspGuardedBinaryGripperAction
    assert tuple(grip.cfg.tcp_offset) == (0.0, 0.0, 0.1034)
    assert grip.cfg.lateral_thresh_xy == 0.020
    assert grip.cfg.vertical_thresh_z == 0.030
    assert tuple(env.scene["insertive_object"].cfg.spawn.scale) == (CUPCAKE_SCALE,) * 3
    assert tuple(env.scene["receptive_object"].cfg.spawn.scale) == (PLATE_SCALE,) * 3
    assert tuple(arm.cfg.max_joint_velocity) == (JOINT_VELOCITY_CAP_RAD_S,) * 7
    assert math.isclose(env.step_dt, 0.1) and env.max_episode_length == 640
    assert env.action_manager.total_action_dim == 7
    assert env.observation_manager.group_obs_dim["policy"] == (200,)
    assert env.observation_manager.group_obs_dim["critic"] == (168,)
    assert not env.curriculum_manager.active_terms
    reset = env.cfg.events.reset_from_reset_states.params
    contract_mode = os.environ.get("CUPCAKE_HALF_CONTRACT_MODE", "train")
    if contract_mode == "train":
        expected_reset_types = RESET_TYPES
        expected_reset_probabilities = [0.25] * 4
    elif contract_mode == "pregrasp100_train":
        expected_reset_types = ["ObjectRestingEEGrasped", "ObjectPartiallyAssembledEEGrasped"]
        expected_reset_probabilities = [0.5, 0.5]
    elif contract_mode == "t0_eval":
        expected_reset_types = ["ObjectAnywhereEEAnywhere"]
        expected_reset_probabilities = [1.0]
    elif contract_mode == "single_path_eval":
        single_reset_type = os.environ.get("CUPCAKE_HALF_SINGLE_RESET_TYPE")
        if single_reset_type not in RESET_TYPES:
            raise ValueError(
                "CUPCAKE_HALF_SINGLE_RESET_TYPE must name one canonical reset path, "
                f"got {single_reset_type!r}"
            )
        expected_reset_types = [single_reset_type]
        expected_reset_probabilities = [1.0]
    else:
        raise ValueError(f"unsupported CUPCAKE_HALF_CONTRACT_MODE={contract_mode!r}")
    assert reset["reset_types"] == expected_reset_types
    assert reset["probs"] == expected_reset_probabilities
    dataset_root = os.environ.get("CUPCAKE_RESET_DATASET_ROOT")
    assert dataset_root, "half-size guarded training requires an explicit new reset root"
    assert Path(reset["dataset_dir"]).resolve() == Path(dataset_root).resolve()
    manifest = json.loads((Path(dataset_root) / "manifest.json").read_text())
    manifest_version = os.environ.get("CUPCAKE_HALF_RESET_MANIFEST_VERSION", "cupcake_half_omnireset_v1")
    pregrasp_version = "cupcake_half_papercup_pregrasp_augmented_v1"
    hundred_version = "cupcake_half_pregrasp100_v1"
    halfmix_version = "cupcake_half_pregrasp_halfmix_v1"
    assert manifest_version in ("cupcake_half_omnireset_v1", "cupcake_half_papercup_omnireset_v1", pregrasp_version, hundred_version, halfmix_version)
    assert manifest["version"] == manifest_version
    assert (contract_mode == "pregrasp100_train") == (manifest_version == hundred_version)
    if manifest_version in ("cupcake_half_papercup_omnireset_v1", pregrasp_version, hundred_version, halfmix_version):
        expected_counts = {name: 2500 for name in RESET_TYPES}
        if manifest_version == halfmix_version:
            assert contract_mode == "train"
            mixed_types = RESET_TYPES[1:]
            expected_counts.update({name: 5000 for name in mixed_types})
            assert set(manifest["pregrasp_additions"]) == set(mixed_types)
            selection_path = Path(dataset_root) / "pregrasp_selection.json"
            assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == manifest["pregrasp_selection_sha256"]
            selection = json.loads(selection_path.read_text())
            assert set(selection) == set(mixed_types)
            for name in mixed_types:
                extra = manifest["pregrasp_additions"][name]
                assert extra["count"] == 2500 and extra["unique_count"] == 50
                assert extra["start_index"] == 2500 and extra["end_index_exclusive"] == 5000
                assert len(selection[name]) == len(set(selection[name])) == 50
                for field, digest in (("pool_path", "addition_sha256"), ("evidence_path", "evidence_sha256")):
                    assert hashlib.sha256((Path(dataset_root) / extra[field]).read_bytes()).hexdigest() == extra[digest]
        if manifest_version == hundred_version:
            expected_counts.update({name: 50 for name in expected_reset_types})
            assert manifest["unique_active_states"] == 100
            assert manifest["active_reset_types"] == expected_reset_types
            assert manifest["active_reset_probabilities"] == expected_reset_probabilities
            selection_path = Path(dataset_root) / "pregrasp_selection.json"
            assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == manifest["pregrasp_selection_sha256"]
            selection = json.loads(selection_path.read_text())
            assert len(selection) == 100
            assert len({(r["kind"], r["addition_index"]) for r in selection}) == 100
        if manifest_version == pregrasp_version:
            augmented_types = ("ObjectRestingEEGrasped", "ObjectPartiallyAssembledEEGrasped")
            expected_counts.update({name: 3000 for name in augmented_types})
            assert set(manifest["pregrasp_additions"]) == set(augmented_types)
            for name in augmented_types:
                extra = manifest["pregrasp_additions"][name]
                assert extra["count"] == 500 and extra["start_index"] == 2500 and extra["end_index_exclusive"] == 3000
        assert manifest["counts"] == expected_counts
        assert math.isclose(manifest["plate_uniform_scale"], PLATE_SCALE)
        root = Path(dataset_root)
        for artifact in ("grasp", "partial_assemblies"):
            assert hashlib.sha256((root / manifest[artifact + "_path"]).read_bytes()).hexdigest() == manifest[artifact + "_sha256"]
        metadata_path = root / "Grasps/CupCakeHalf/papercup_grasp_metadata.json"
        assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == manifest["grasp_metadata_sha256"]
        metadata = json.loads(metadata_path.read_text())
        assert len(metadata["grasps"]) == 10
        assert all(10 <= z <= 20 for grasp in metadata["grasps"] for z in grasp["pad_z_mm"])
        assert hashlib.sha256((root / "source_contract_precollection.json").read_bytes()).hexdigest() == manifest["source_contract_sha256"]
        assert set(manifest["reset_sha256"]) == {f"resets_{name}.pt" for name in RESET_TYPES}
    assert manifest["cupcake_uniform_scale"] == CUPCAKE_SCALE
    assert manifest["pair"] == "CupCakeHalf__Plate"
    for name, expected in manifest["reset_sha256"].items():
        path = Path(dataset_root) / "Resets" / manifest["pair"] / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    contract = {
        "version": "cupcake_half_guarded_v020h64_20260913",
        "contract_mode": contract_mode,
        "cupcake_scale": CUPCAKE_SCALE,
        "plate_scale": PLATE_SCALE,
        "gripper": type(grip).__name__,
        "policy_gripper_output": "ignored_by_guard",
        "guard_tcp_offset_m": list(grip.cfg.tcp_offset),
        "guard_lateral_threshold_m": grip.cfg.lateral_thresh_xy,
        "guard_vertical_threshold_m": grip.cfg.vertical_thresh_z,
        "joint_velocity_cap_rad_s": JOINT_VELOCITY_CAP_RAD_S,
        "episode_steps": 640,
        "reset_root": dataset_root,
        "reset_manifest_version": manifest_version,
        "reset_manifest_sha256": hashlib.sha256((Path(dataset_root) / "manifest.json").read_bytes()).hexdigest(),
        "reset_hashes_verified": True,
        "reset_types": expected_reset_types,
        "reset_probabilities": expected_reset_probabilities,
        "reset_counts": manifest.get("counts"),
        "pregrasp_additions": manifest.get("pregrasp_additions", {}),
        "success": "20mm / abs(roll)+abs(pitch)<3deg",
    }
    print("[CUPCAKE_HALF_GUARDED_CONTRACT] " + json.dumps(contract, sort_keys=True), flush=True)
    if env.cfg.log_dir:
        output = Path(env.cfg.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        (output / f"cupcake_half_guarded_contract_rank{rank}.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )


@configclass
class CupCakeHalfGuardedEvents(CupCakePolicyGripperEvents):
    policy_gripper_contract = EventTerm(func=verify_half_guarded_contract, mode="startup")


@configclass
class CupCakeHalfGuardedTrainCfg(CupCakePolicyGripperTrainCfg):
    events: CupCakeHalfGuardedEvents = CupCakeHalfGuardedEvents()

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm.max_joint_velocity = (JOINT_VELOCITY_CAP_RAD_S,) * 7
        self.scene.insertive_object = copy.deepcopy(
            self.variants["scene.insertive_object"]["cupcake_half"]
        )
        self.actions.gripper = FR3_GUARDED_GRIPPER
        if os.environ.get("CUPCAKE_HALF_CONTRACT_MODE") == "pregrasp100_train":
            self.events.reset_from_reset_states.params.update(
                reset_types=["ObjectRestingEEGrasped", "ObjectPartiallyAssembledEEGrasped"],
                probs=[0.5, 0.5],
            )
