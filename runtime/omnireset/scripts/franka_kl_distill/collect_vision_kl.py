"""Vision demo collect for KL distillation — records 3-camera RGB + privileged state +
teacher (mean, std), so an IMAGE student can be distilled from the STATE teacher.

Key trick: the state teacher (`model_6800`) needs its native ~200-dim history-concatenated
state, but the RGB env's `policy` group is image-based. So we REASSIGN the RGB env's policy
obs group to the State env's PolicyCfg in-script — the teacher then reads pure state via
`policy(obs)`, while the raw uint8 images come from the `data_collection` group
(`obs_buf["data_collection"]["{front,side,wrist}_rgb"]`, process_image=False → HWC uint8).
The 200-dim policy state is also recorded as the aux-reconstruction target for the vision model.

Saves only SUCCESSFUL episodes. Legacy OSC output uses 7-D EE/gripper
actions and proprio. With ``--joint_target_bridge``, both become 8-D:
``[q1..q7, gripper_width]`` and the stored action is the exact post-safety
target passed to joint-position control.

Output zarr:
  data/{front_rgb, side_rgb, wrist_rgb}  uint8 (N,224,224,3)
  data/{action(=teacher mean), action_std, state(200-dim privileged)}  float32
  data/proprio  float32 (N,7 or 8) = deployable robot-only state
      (The 200-dim `state` is privileged — it contains insertive/receptive object poses — so it
       is kept only as the DP aux target, NOT fed to a vision policy.)
  meta/episode_ends

Example (cameras forced on; RGB rendering is heavy → fewer envs):
    CUDA_VISIBLE_DEVICES=1 python scripts/franka_kl_distill/collect_vision_kl.py \\
        --teacher_ckpt logs/.../2026-06-12_11-01-50/model_6800.pt \\
        --camera_setup real --num_envs 32 --num_demos 200 \\
        --output datasets/franka_vision_kl.zarr --headless \\
        env.scene.robot.actuators.panda_hand.stiffness=1000.0 \\
        env.scene.robot.actuators.panda_hand.damping=14.0 \\
        env.scene.robot.actuators.panda_hand.effort_limit_sim=60.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

from isaaclab.app import AppLauncher


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

parser = argparse.ArgumentParser(description="Vision demo collect (RGB + teacher labels).")
parser.add_argument("--task", default="OmniReset-FrankaFr3Gripper-RelCartesianOSC-RGB-DataCollection-v0")
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_demos", type=int, default=200)
parser.add_argument("--output", required=True)
parser.add_argument(
    "--receptive_object_uniform_scale",
    type=float,
    default=1.0,
    help=(
        "Uniform USD spawn scale for the receptive object. Applied after Hydra task/object "
        "variant resolution so object-pair defaults cannot silently overwrite it."
    ),
)
parser.add_argument(
    "--image_size",
    type=int,
    default=224,
    help="Stored square RGB size. Use 84 for the joint-target DP pipeline.",
)
parser.add_argument(
    "--rotate_180_camera",
    action="append",
    choices=("front_rgb", "side_rgb", "wrist_rgb"),
    default=[],
    help="Rotate this RGB observation by 180 degrees before writing it to Zarr. Repeat as needed.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--camera_setup", choices=["historical", "real"], default=None,
                    help="camera base extrinsics for RGB collection. Default real uses the 2026-07 "
                         "L515/D415/D405 calibrated setup; use historical only for old DP artifacts.")
parser.add_argument("--four_path", action="store_true", help="faithful 4-path reset mix (vs task_0 only)")
parser.add_argument("--reset_types", nargs="+", default=None,
                    help="override single-path reset_types, e.g. "
                         "ObjectAnywhereEEAnywhere_upright_yaw_3cm_fixedhome. Default None -> "
                         "ObjectAnywhereEEAnywhere (wide). Ignored when --four_path is set. "
                         "NOTE: a CLI hydra override of reset_from_reset_states is clobbered by "
                         "this script's in-code reset selection, so use THIS flag, not hydra.")
parser.add_argument("--keep_appearance", action="store_true",
                    help="with --no_visual_dr: keep appearance (texture) randomization so the table/"
                         "curtains keep randomizing; only camera/light jitter is zeroed. By default, "
                         "peg/hole stay fixed #8E9089 while other appearance events are removed.")
parser.add_argument("--no_visual_dr", action="store_true",
                    help="legacy/debug only: disable visual DR (texture/HDRI/camera jitter). Default is DR ON.")
parser.add_argument("--no_dynamics_dr", action="store_true",
                    help="debug ablation: freeze/remove non-visual DR (mass/material/gripper/OSC/sysid).")
parser.add_argument(
    "--stackcube_fixed_scene",
    action="store_true",
    help="apply the validated deterministic StackCube B3 mass/friction profile; "
         "also fixes gripper/OSC/sysid events.",
)
parser.add_argument(
    "--b3_nominal_fixed_scene",
    action="store_true",
    help="Apply the same deterministic midpoint B3 mass/friction profile under a "
         "task-neutral name. The insertive-object mass is fixed to 0.11 kg.",
)
parser.add_argument(
    "--fixed_insertive_mass_kg",
    type=float,
    default=None,
    help="Keep only the insertive-object startup mass event and pin it to this absolute mass. "
         "Other non-visual randomization is disabled as usual.",
)
parser.add_argument(
    "--preserve_object_face_materials",
    action="store_true",
    help="disable insertive/receptive visual overrides so asset-authored face colors remain visible; "
         "camera, curtain, and HDRI DR stay active.",
)
parser.add_argument("--no_all_dr", action="store_true",
                    help="debug ablation: disable all visual/non-visual DR, including HDRI swaps.")
parser.add_argument("--target_frames", type=int, default=0,
                    help="stop after this many stored frames (vs --num_demos episodes). "
                         "Use with --four_path: episode-count stopping degenerates there since "
                         "near-success resets finish in 1 frame.")
parser.add_argument(
    "--no_store_images",
    action="store_true",
    help="Trajectory-selection mode: still run the RGB task, but omit camera arrays from zarr. "
         "Useful for choosing successful physical reset states before visual re-rendering.",
)
parser.add_argument(
    "--zarr_compressor",
    choices=("default", "zstd"),
    default="default",
    help="Zarr array compressor. 'default' preserves historical LZ4 output; "
         "'zstd' uses lossless Blosc Zstd level 5 with byte shuffle for large datasets.",
)
parser.add_argument(
    "--target_attempts",
    type=int,
    default=0,
    help="stop after this many completed episodes and save only successes; used for unbiased success-rate gates.",
)
parser.add_argument(
    "--post_reset_warmup_steps",
    type=int,
    default=0,
    help="Execute this many unrecorded zero-arm steps after every reset so camera/material updates settle.",
)
parser.add_argument(
    "--save_all_episodes",
    action="store_true",
    help="Diagnostic only: save failed and successful episodes instead of the canonical "
         "success-only BC filter. Episode outcome is stored in data/episode_success and metadata.",
)
parser.add_argument(
    "--outcomes_only",
    action="store_true",
    help="Qualification-only mode: execute every forced state and write a compact JSON "
         "outcome audit instead of buffering trajectory arrays.",
)
parser.add_argument(
    "--record_placement_metrics",
    action="store_true",
    help="Store per-frame placement distance/orientation diagnostics in the zarr.",
)
parser.add_argument(
    "--save_fixed_gate_episodes",
    action="store_true",
    help="Also save episodes that ever satisfy --diagnostic_position_m/"
         "--diagnostic_orientation_deg, even if the env success termination does not fire.",
)
parser.add_argument(
    "--truncate_to_fixed_gate",
    action="store_true",
    help="With --save_fixed_gate_episodes, truncate saved episodes at the first fixed-gate frame.",
)
parser.add_argument(
    "--raw_teacher_gripper",
    action="store_true",
    help="Historical reproduction only: retain the RL teacher's ignored 7th output. "
         "Default replaces it with the grasp guard's actual binary command so vision policies "
         "can learn gripper control.",
)
parser.add_argument(
    "--native_osc_policy_gripper",
    action="store_true",
    help="Execute the 7-D RL teacher through native Cartesian OSC and the standard "
         "policy-controlled binary gripper. Store the teacher action actually sent to both "
         "terms; no grasp guard or fixed-open rewrite.",
)
parser.add_argument(
    "--native_osc_guard_labels",
    action="store_true",
    help="Execute the arm teacher through native Cartesian OSC while the task's grasp guard "
         "controls the gripper. Store the guard's executed -1/+1 command as the BC label and "
         "preserve the ignored raw teacher output separately. Student eval must remove the guard.",
)
parser.add_argument(
    "--osc_max_linear_velocity",
    type=float,
    default=None,
    help="In a native OSC collection mode, limit the moving Cartesian reference in m/s.",
)
parser.add_argument(
    "--osc_max_angular_velocity",
    type=float,
    default=None,
    help="In a native OSC collection mode, limit the moving orientation reference in rad/s.",
)
parser.add_argument(
    "--joint_target_bridge",
    action="store_true",
    help="execute the legacy 6-D Cartesian teacher through differential IK and "
         "absolute joint-position control; store the actual 7-D q target plus gripper width target.",
)
parser.add_argument(
    "--stochastic_teacher_actions",
    action="store_true",
    help="Execute samples from the PPO action distribution. The default executes the "
         "deterministic actor mean for historical dataset reproduction.",
)
parser.add_argument(
    "--online_xy5_t0_reset",
    action="store_true",
    help="Use the task's continuous XY5 object and team-home +/-0.2 rad joint reset event "
         "instead of wrapping a MultiResetManager state pool.",
)
parser.add_argument(
    "--fixed_open_gripper",
    action="store_true",
    help="With --joint_target_bridge, replace the grasp guard with a fixed-open binary "
         "gripper. This is a controlled gripper ablation and stores a 0.08 m width target.",
)
parser.add_argument(
    "--osc_joint_trace",
    action="store_true",
    help="Execute the teacher with its native Cartesian OSC controller while storing "
         "the achieved arm joint trajectory as "
         "candidate absolute joint-target BC labels. Candidate labels require replay validation.",
)
parser.add_argument(
    "--osc_joint_trace_gripper_mode",
    choices=("fixed_open", "policy_sign", "grasp_guard"),
    default="fixed_open",
    help=(
        "Gripper command used during --osc_joint_trace. fixed_open preserves the "
        "Cupcake pilot behavior; policy_sign executes the teacher's binary gripper "
        "sign; grasp_guard executes the task rule-based guard."
    ),
)
parser.add_argument(
    "--osc_joint_trace_target",
    choices=("achieved_q", "torque_match"),
    default="achieved_q",
    help="Label construction for --osc_joint_trace. achieved_q stores the arm joint "
         "position reached by the native OSC step. torque_match stores the absolute "
         "joint-position target whose first-substep PD torque matches the native OSC "
         "torque under --joint_position_stiffness/--joint_position_damping.",
)
parser.add_argument(
    "--joint_teacher",
    action="store_true",
    help="Execute an 8-D joint-control RL teacher directly: seven normalized joint deltas "
         "plus policy-controlled gripper. Store the exact absolute joint target and width "
         "target applied to the simulator; no IK and no gripper guard.",
)
parser.add_argument("--joint_ik_damping", type=float, default=0.05)
parser.add_argument(
    "--joint_ik_step_scale",
    type=float,
    default=1.0,
    help="Fraction of each DLS joint increment retained before rate limiting. One preserves "
         "the existing bridge; lower values test response matching to the native OSC plant.",
)
parser.add_argument(
    "--joint_simulation_jacobian_point",
    choices=("link_origin", "physx_com"),
    default="link_origin",
    help="Simulation-Jacobian point for --joint_target_bridge. link_origin preserves the "
         "existing bridge; physx_com matches the historical native OSC teacher config.",
)
parser.add_argument(
    "--joint_action_reference_blend",
    type=float,
    default=None,
    help="Override the bridge action-reference blend in [0, 1]. Default inherits the "
         "native OSC teacher config (currently 0, measured-EE anchoring).",
)
parser.add_argument(
    "--joint_max_velocity",
    type=float,
    default=None,
    help="Required with --joint_target_bridge. Per-joint target rate limit in rad/s. "
         "The published 300x10 dataset used 0.30 as an experiment setting; it is not a "
         "validated real-robot streaming limit.",
)
parser.add_argument("--joint_position_stiffness", type=float, default=80.0)
parser.add_argument("--joint_position_damping", type=float, default=4.0)
parser.add_argument(
    "--joint_home_arm",
    type=float,
    nargs=7,
    default=None,
    metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
    help="Expected seven-axis arm pose at the first recorded frame. Default preserves "
         "the published Peg team-home setting; StackCube fixed-home collection must pass "
         "its factory-home pose explicitly.",
)
parser.add_argument(
    "--joint_episode_length_s",
    type=float,
    default=64.0,
    help="joint-target episodes are slower than OSC; default allows 640 policy steps at 10 Hz.",
)
parser.add_argument(
    "--action_hold_steps",
    type=int,
    default=1,
    help=(
        "Diagnostic stop-go execution for --joint_target_bridge. Query the RL teacher and "
        "compute a new DLS q target once every N environment steps, holding that exact q "
        "target and gripper command between decisions. Default 1 preserves collection."
    ),
)
parser.add_argument(
    "--reset_state_indices_file",
    default=None,
    help="Optional JSON list or comma/whitespace-separated reset-state indices. "
         "Indices are consumed deterministically instead of sampled randomly.",
)
parser.add_argument(
    "--reset_state_repeats",
    type=int,
    default=1,
    help="Repeat each forced reset-state index this many times, for controlled visual variants.",
)
parser.add_argument(
    "--complete_forced_state_set",
    action="store_true",
    help="Attempt every unique index from --reset_state_indices_file exactly once. "
         "Extra in-flight repeats after the forced sequence wraps are discarded.",
)
parser.add_argument(
    "--successful_repeats_per_state",
    type=int,
    default=0,
    help="For forced reset states, collect exactly this many successful episodes per state. "
         "Failed episodes are retried and successful excess episodes are discarded.",
)
parser.add_argument("--home_tolerance_rad", type=float, default=0.005)
parser.add_argument(
    "--skip_joint_home_check",
    action="store_true",
    help=(
        "Allow joint-label collection from reset sets whose first arm pose is not a "
        "single fixed home. Use this for OSC-to-joint trace conversion gates on "
        "the original EE-anywhere reset distribution."
    ),
)
parser.add_argument(
    "--diagnostic_done_metrics",
    action="store_true",
    help="Print endpoint placement-error statistics without changing which episodes are saved.",
)
parser.add_argument(
    "--disable_corrupted_camera_termination",
    action="store_true",
    help="Diagnostic control-only option: keep collecting when a camera frame has low variance. "
         "Do not use this for canonical vision datasets.",
)
parser.add_argument("--diagnostic_position_m", type=float, default=0.020)
parser.add_argument("--diagnostic_orientation_deg", type=float, default=3.0)
parser.add_argument(
    "--success_position_override_m",
    type=float,
    default=None,
    help="Override the asset-metadata 3D success threshold for controlled collection ablations.",
)
parser.add_argument(
    "--success_orientation_override_deg",
    type=float,
    default=None,
    help="Override the asset-metadata roll/pitch success threshold in degrees.",
)
parser.add_argument("--max_steps", type=int, default=200_000)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "UWLab", "scripts", "reinforcement_learning", "rsl_rl"))
import cli_args as _rsl_cli_args  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not math.isfinite(args_cli.receptive_object_uniform_scale) or args_cli.receptive_object_uniform_scale <= 0.0:
    parser.error("--receptive_object_uniform_scale must be finite and positive")
if args_cli.no_all_dr:
    args_cli.no_visual_dr = True
    args_cli.no_dynamics_dr = True
    args_cli.keep_appearance = False
if args_cli.native_osc_policy_gripper:
    if args_cli.raw_teacher_gripper:
        parser.error("--native_osc_policy_gripper cannot use the historical ignored-gripper mode")
    if args_cli.joint_target_bridge or args_cli.osc_joint_trace or args_cli.joint_teacher:
        parser.error("--native_osc_policy_gripper cannot be combined with a joint-label mode")
if args_cli.native_osc_guard_labels:
    if args_cli.raw_teacher_gripper:
        parser.error("--native_osc_guard_labels cannot use the historical ignored-gripper mode")
    if (
        args_cli.native_osc_policy_gripper
        or args_cli.joint_target_bridge
        or args_cli.osc_joint_trace
        or args_cli.joint_teacher
    ):
        parser.error("--native_osc_guard_labels cannot be combined with another action-label mode")
if (args_cli.osc_max_linear_velocity is None) != (args_cli.osc_max_angular_velocity is None):
    parser.error("OSC linear and angular reference limits must be set together")
if args_cli.osc_max_linear_velocity is not None:
    if not (
        args_cli.native_osc_policy_gripper
        or args_cli.native_osc_guard_labels
        or args_cli.osc_joint_trace
    ):
        parser.error("OSC reference limits require a native OSC collection mode")
    if args_cli.osc_max_linear_velocity <= 0.0 or args_cli.osc_max_angular_velocity <= 0.0:
        parser.error("OSC reference velocity limits must be positive")
if args_cli.joint_target_bridge:
    args_cli.no_dynamics_dr = True
    if args_cli.joint_max_velocity is None:
        parser.error("--joint_target_bridge requires an explicit --joint_max_velocity")
    if args_cli.four_path:
        parser.error("--joint_target_bridge requires a single home reset and cannot be combined with --four_path")
if args_cli.online_xy5_t0_reset:
    if not args_cli.joint_target_bridge:
        parser.error("--online_xy5_t0_reset currently requires --joint_target_bridge")
    if args_cli.reset_types or args_cli.reset_state_indices_file:
        parser.error("--online_xy5_t0_reset cannot be combined with reset pools or state indices")
    if args_cli.successful_repeats_per_state or args_cli.complete_forced_state_set:
        parser.error("--online_xy5_t0_reset cannot use per-state quotas")
if args_cli.fixed_open_gripper:
    if not args_cli.joint_target_bridge:
        parser.error("--fixed_open_gripper requires --joint_target_bridge")
    if args_cli.raw_teacher_gripper:
        parser.error("--fixed_open_gripper cannot be combined with --raw_teacher_gripper")
if args_cli.joint_action_reference_blend is not None and not (
    0.0 <= args_cli.joint_action_reference_blend <= 1.0
):
    parser.error("--joint_action_reference_blend must be in [0, 1]")
if not 0.0 < args_cli.joint_ik_step_scale <= 1.0:
    parser.error("--joint_ik_step_scale must be in (0, 1]")
if args_cli.osc_joint_trace:
    if args_cli.joint_target_bridge:
        parser.error("--osc_joint_trace cannot be combined with --joint_target_bridge")
    if args_cli.raw_teacher_gripper:
        parser.error("--osc_joint_trace controls the gripper via --osc_joint_trace_gripper_mode")
    if args_cli.four_path:
        parser.error("--osc_joint_trace requires a single fixed-home reset and cannot use --four_path")
    if args_cli.osc_joint_trace_target == "torque_match":
        if args_cli.joint_max_velocity is None or args_cli.joint_max_velocity <= 0.0:
            parser.error("--osc_joint_trace_target torque_match requires a positive --joint_max_velocity")
        if args_cli.joint_position_stiffness <= 0.0:
            parser.error("--osc_joint_trace_target torque_match requires positive --joint_position_stiffness")
        if args_cli.joint_position_damping < 0.0:
            parser.error("--osc_joint_trace_target torque_match requires non-negative --joint_position_damping")
elif args_cli.osc_joint_trace_target != "achieved_q":
    parser.error("--osc_joint_trace_target requires --osc_joint_trace")
if args_cli.joint_teacher:
    if args_cli.joint_target_bridge or args_cli.osc_joint_trace:
        parser.error("--joint_teacher cannot be combined with another joint-label mode")
    if args_cli.raw_teacher_gripper:
        parser.error("--joint_teacher always uses the policy-controlled gripper")
    if args_cli.four_path:
        parser.error("--joint_teacher collection requires a single fixed-home reset")
    if args_cli.joint_max_velocity is None or args_cli.joint_max_velocity <= 0.0:
        parser.error("--joint_teacher requires an explicit positive --joint_max_velocity")
if args_cli.reset_state_repeats < 1:
    parser.error("--reset_state_repeats must be positive")
if (args_cli.success_position_override_m is None) != (
    args_cli.success_orientation_override_deg is None
):
    parser.error("success position and orientation overrides must be provided together")
if args_cli.success_position_override_m is not None:
    if args_cli.success_position_override_m <= 0.0:
        parser.error("--success_position_override_m must be positive")
    if args_cli.success_orientation_override_deg <= 0.0:
        parser.error("--success_orientation_override_deg must be positive")
if args_cli.image_size < 32:
    parser.error("--image_size must be at least 32")
if args_cli.post_reset_warmup_steps < 0:
    parser.error("--post_reset_warmup_steps cannot be negative")
if args_cli.action_hold_steps < 1:
    parser.error("--action_hold_steps must be positive")
if args_cli.action_hold_steps != 1 and not args_cli.joint_target_bridge:
    parser.error("--action_hold_steps > 1 currently requires --joint_target_bridge")
if args_cli.diagnostic_position_m <= 0.0 or args_cli.diagnostic_orientation_deg <= 0.0:
    parser.error("diagnostic placement thresholds must be positive")
if args_cli.truncate_to_fixed_gate and not args_cli.save_fixed_gate_episodes:
    parser.error("--truncate_to_fixed_gate requires --save_fixed_gate_episodes")
if args_cli.fixed_insertive_mass_kg is not None:
    if args_cli.fixed_insertive_mass_kg <= 0.0:
        parser.error("--fixed_insertive_mass_kg must be positive")
    if args_cli.stackcube_fixed_scene:
        parser.error("--fixed_insertive_mass_kg cannot be combined with --stackcube_fixed_scene")
if args_cli.b3_nominal_fixed_scene and args_cli.stackcube_fixed_scene:
    parser.error("use only one of --b3_nominal_fixed_scene and --stackcube_fixed_scene")
if args_cli.b3_nominal_fixed_scene and args_cli.fixed_insertive_mass_kg is not None:
    parser.error("--b3_nominal_fixed_scene already fixes the insertive-object mass")
if args_cli.b3_nominal_fixed_scene:
    # Reuse the validated implementation while preserving a task-neutral metadata label.
    args_cli.stackcube_fixed_scene = True
if args_cli.successful_repeats_per_state < 0:
    parser.error("--successful_repeats_per_state cannot be negative")
if args_cli.save_all_episodes and args_cli.successful_repeats_per_state:
    parser.error("--save_all_episodes cannot be combined with a successful-repeat quota")
if args_cli.outcomes_only and not args_cli.complete_forced_state_set:
    parser.error("--outcomes_only requires --complete_forced_state_set")
if args_cli.outcomes_only and not args_cli.no_store_images:
    parser.error("--outcomes_only requires --no_store_images")
if args_cli.outcomes_only and (
    args_cli.save_all_episodes
    or args_cli.successful_repeats_per_state
    or args_cli.record_placement_metrics
    or args_cli.save_fixed_gate_episodes
):
    parser.error("--outcomes_only cannot be combined with trajectory-saving modes")
if args_cli.reset_state_indices_file and not (
    args_cli.joint_target_bridge
    or args_cli.osc_joint_trace
    or args_cli.joint_teacher
    or args_cli.native_osc_policy_gripper
    or args_cli.native_osc_guard_labels
):
    parser.error("--reset_state_indices_file requires an indexed collection mode")
if args_cli.complete_forced_state_set and not args_cli.reset_state_indices_file:
    parser.error("--complete_forced_state_set requires --reset_state_indices_file")
if args_cli.complete_forced_state_set and args_cli.reset_state_repeats != 1:
    parser.error("--complete_forced_state_set requires --reset_state_repeats 1")
if args_cli.successful_repeats_per_state:
    if not (
            args_cli.joint_target_bridge or args_cli.osc_joint_trace
            or args_cli.joint_teacher or args_cli.native_osc_policy_gripper
            or args_cli.native_osc_guard_labels
        ) or not args_cli.reset_state_indices_file:
            parser.error(
                "--successful_repeats_per_state requires an indexed collection mode and "
                "--reset_state_indices_file"
            )
    if args_cli.complete_forced_state_set:
        parser.error("--successful_repeats_per_state cannot be combined with --complete_forced_state_set")
    if args_cli.reset_state_repeats != 1:
        parser.error("--successful_repeats_per_state requires --reset_state_repeats 1")
    if args_cli.target_attempts:
        parser.error("--successful_repeats_per_state cannot be combined with --target_attempts")
if not os.path.isfile(args_cli.teacher_ckpt):
    parser.error(f"teacher checkpoint is missing: {args_cli.teacher_ckpt}")
_TEACHER_CHECKPOINT_SHA256 = _file_sha256(args_cli.teacher_ckpt)
_EXPECTED_TEACHER_SHA256 = os.environ.get("TEACHER_SHA256")
if (
    _EXPECTED_TEACHER_SHA256 is not None
    and _EXPECTED_TEACHER_SHA256 != _TEACHER_CHECKPOINT_SHA256
):
    parser.error(
        "TEACHER_SHA256 does not match --teacher_ckpt: "
        f"{_EXPECTED_TEACHER_SHA256} != {_TEACHER_CHECKPOINT_SHA256}"
    )


_OBJECT_APPEARANCE_EVENTS = {
    "randomize_insertive_object_appearance",
    "randomize_receptive_object_appearance",
}
_OBJECT_GREY = (142.0 / 255.0, 144.0 / 255.0, 137.0 / 255.0)  # #8E9089
_TEAM_HOME_RESET_TYPE = "ObjectAnywhereEEAnywhere_upright_yaw_3cm_frontcenter_teamhome_20260729"
_TEAM_HOME_ARM = (
    -0.057557623295429294,
    0.00018949155714934572,
    -0.010052990086534465,
    -1.5410414293429622,
    -0.03143259380432439,
    1.5459539128296862,
    -2.462186588172044,
)
_STACKCUBE_B3_FIXED_SCENE = {
    "robot_material": {
        "static_friction_range": (0.75, 0.75),
        "dynamic_friction_range": (0.60, 0.60),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "insertive_object_material": {
        "static_friction_range": (1.50, 1.50),
        "dynamic_friction_range": (1.40, 1.40),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "receptive_object_material": {
        "static_friction_range": (0.40, 0.40),
        "dynamic_friction_range": (0.325, 0.325),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "table_material": {
        "static_friction_range": (0.45, 0.45),
        "dynamic_friction_range": (0.35, 0.35),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "randomize_robot_mass": {"mass_distribution_params": (1.0, 1.0)},
    "randomize_insertive_object_mass": {"mass_distribution_params": (0.11, 0.11)},
    "randomize_receptive_object_mass": {"mass_distribution_params": (1.0, 1.0)},
    "randomize_table_mass": {"mass_distribution_params": (1.0, 1.0)},
}


def _load_reset_state_indices(path: str | None) -> list[int] | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Reset-state index file is empty: {path}")
    if text.startswith("["):
        values = json.loads(text)
    else:
        values = text.replace(",", " ").split()
    try:
        indices = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Reset-state index file must contain integers: {path}") from exc
    if not indices:
        raise ValueError(f"Reset-state index file contains no indices: {path}")
    return indices


def _freeze_grey_object_appearance(event):
    """Keep the visibility material, but remove every stochastic degree of freedom."""
    params = event.params
    event.mode = "startup"
    params["texture_prob"] = 0.0
    params.pop("texture_paths", None)
    params.pop("texture_config_path", None)
    params.pop("diffuse_tint_range", None)
    params["colors"] = {
        "r": (_OBJECT_GREY[0], _OBJECT_GREY[0]),
        "g": (_OBJECT_GREY[1], _OBJECT_GREY[1]),
        "b": (_OBJECT_GREY[2], _OBJECT_GREY[2]),
    }
    params["texture_scale_range"] = (1.0, 1.0)
    params["roughness_range"] = (0.55, 0.55)
    params["metallic_range"] = (0.10, 0.10)
    params["specular_range"] = (0.30, 0.30)


def _apply_camera_setup(camera_setup: str | None) -> str:
    if camera_setup is None:
        env_setup = os.environ.get("OMNIRESET_CAMERA_SETUP", "").strip().lower()
        if env_setup in {"historical", "real"}:
            camera_setup = env_setup
        elif os.environ.get("OMNIRESET_REAL_BASE_CAMERAS", "0") == "1":
            camera_setup = "real"
        else:
            camera_setup = "real"
    os.environ["OMNIRESET_CAMERA_SETUP"] = camera_setup
    os.environ["OMNIRESET_REAL_BASE_CAMERAS"] = "1" if camera_setup == "real" else "0"
    if camera_setup == "real":
        os.environ.setdefault("OMNIRESET_REAL_OBJECT_TABLE_Z_OFFSET", "0.0")
    print(f"[camera-setup] {camera_setup} (OMNIRESET_REAL_BASE_CAMERAS={os.environ['OMNIRESET_REAL_BASE_CAMERAS']})")
    return camera_setup


args_cli.camera_setup = _apply_camera_setup(args_cli.camera_setup)
args_cli.enable_cameras = True  # cameras REQUIRED for RGB collection
# Use the rendering experience for headless RGB collection on A100 GPUs.
if args_cli.headless and not args_cli.experience:
    args_cli.experience = "isaaclab.python.rendering.kit"

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
import numcodecs  # noqa: E402

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402
# the State env's PolicyCfg — the EXACT obs the teacher was trained on
from uwlab_tasks.manager_based.manipulation.omnireset.config.franka.rl_state_cfg import (  # noqa: E402
    ObservationsCfg as StateObsCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.franka.actions import (  # noqa: E402
    FrankaFr3GripperRelativeJointTargetAction,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.franka.research3_cfg import (  # noqa: E402
    remap_research3_names,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.franka.robot_contract import (  # noqa: E402
    resolve_franka_robot_contract,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.actions.actions_cfg import (  # noqa: E402
    RelCartesianDiffIKJointPositionActionCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import (  # noqa: E402
    MultiResetManager,
)

_COLLECT_RESET_EVENT_CFG = None
_COLLECT_RESET_MANAGER_CACHE = {}


def collect_reset_from_reset_states(
    env,
    env_ids,
    dataset_dir: str,
    reset_types: list[str],
    probs: list[float],
    success: str | None = None,
    state_indices: list[int] | None = None,
    state_index_repeats: int = 1,
    rigid_object_position_offsets: dict[str, list[float]] | None = None,
) -> None:
    if _COLLECT_RESET_EVENT_CFG is None:
        raise RuntimeError("collect reset wrapper was called before reset cfg was installed")
    _COLLECT_RESET_EVENT_CFG.params.update(
        {
            "dataset_dir": dataset_dir,
            "reset_types": reset_types,
            "probs": probs,
            "success": success,
            "state_indices": state_indices,
            "state_index_repeats": state_index_repeats,
            "rigid_object_position_offsets": rigid_object_position_offsets,
        }
    )
    manager = _COLLECT_RESET_MANAGER_CACHE.get(id(env))
    if manager is None:
        manager = MultiResetManager(cfg=_COLLECT_RESET_EVENT_CFG, env=env)
        _COLLECT_RESET_MANAGER_CACHE[id(env)] = manager
    manager(
        env,
        env_ids,
        dataset_dir=dataset_dir,
        reset_types=reset_types,
        probs=probs,
        success=success,
        state_indices=state_indices,
        state_index_repeats=state_index_repeats,
        rigid_object_position_offsets=rigid_object_position_offsets,
    )


def get_collect_reset_manager(env):
    manager = _COLLECT_RESET_MANAGER_CACHE.get(id(env))
    if manager is None:
        if _COLLECT_RESET_EVENT_CFG is None:
            raise RuntimeError("collect reset cfg has not been installed")
        manager = MultiResetManager(cfg=_COLLECT_RESET_EVENT_CFG, env=env)
        _COLLECT_RESET_MANAGER_CACHE[id(env)] = manager
    return manager


def _hide_original_table_visual(num_envs: int, label: str = "vis-collect") -> None:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    hidden = 0
    for env_id in range(num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Table")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[{label}] original OmniReset Table visual hidden: {hidden}/{num_envs}")


def _flat_policy(o):
    if isinstance(o, tuple):
        o = o[0]
    if hasattr(o, "keys"):
        try:
            o = o["policy"]
        except (KeyError, TypeError):
            o = next(iter(o.values()))
    if hasattr(o, "keys"):
        o = torch.cat([v.reshape(v.shape[0], -1) for v in o.values()], dim=-1)
    return o


def neutralize_nonvisual_dr(
    env_cfg,
    *,
    stackcube_fixed_scene: bool = False,
    fixed_insertive_mass_kg: float | None = None,
):
    """Freeze non-visual DR while keeping reset-state sampling active."""
    disabled, fixed = [], []
    scene_event_names = set(_STACKCUBE_B3_FIXED_SCENE)
    seen_scene_events = set()
    for name in list(vars(env_cfg.events)):
        if name.startswith("_") or name == "reset_from_reset_states":
            continue
        ev = getattr(env_cfg.events, name)
        if ev is None:
            continue
        params = getattr(ev, "params", None)
        if name in scene_event_names:
            seen_scene_events.add(name)
            if stackcube_fixed_scene:
                if getattr(ev, "mode", None) != "startup":
                    raise ValueError(
                        f"StackCube fixed-scene event {name} must run at startup, "
                        f"got mode={getattr(ev, 'mode', None)!r}"
                    )
                params.update(_STACKCUBE_B3_FIXED_SCENE[name])
                fixed.append(name)
            elif name == "randomize_insertive_object_mass" and fixed_insertive_mass_kg is not None:
                if getattr(ev, "mode", None) != "startup":
                    raise ValueError(
                        "fixed insertive-object mass requires a startup mass event, "
                        f"got mode={getattr(ev, 'mode', None)!r}"
                    )
                params["mass_distribution_params"] = (
                    fixed_insertive_mass_kg,
                    fixed_insertive_mass_kg,
                )
                fixed.append(name)
            else:
                setattr(env_cfg.events, name, None)
                disabled.append(name)
        elif "gripper_actuator" in name and params is not None:
            for key in ("stiffness_distribution_params", "damping_distribution_params"):
                if key in params:
                    params[key] = (1.0, 1.0)
            fixed.append(name)
        elif ("osc_gains" in name or "arm_sysid" in name) and params is not None:
            if "scale_range" in params:
                params["scale_range"] = (1.0, 1.0)
            if "delay_range" in params:
                params["delay_range"] = (0, 0)
            fixed.append(name)
    if stackcube_fixed_scene:
        missing = sorted(scene_event_names - seen_scene_events)
        if missing:
            raise ValueError(f"B3 nominal fixed-scene events missing from task config: {missing}")
        env_cfg.sim.physx.enable_enhanced_determinism = True
        env_cfg.sim.physx.gpu_max_num_partitions = 1
        env_cfg.sim.physx.bounce_threshold_velocity = 0.5
        env_cfg.sim.physx.friction_correlation_distance = 0.025
        env_cfg.sim.physx.max_position_iteration_count = 32
    if stackcube_fixed_scene:
        label = "B3 nominal fixed scene"
    elif fixed_insertive_mass_kg is not None:
        label = f"non-visual DR OFF, insertive mass fixed at {fixed_insertive_mass_kg:g} kg"
    else:
        label = "non-visual DR OFF"
    print(f"[vis-collect] {label}: disabled {disabled}; fixed {fixed}")
    fixed_scene_profile = {}
    if stackcube_fixed_scene:
        fixed_scene_profile = _plain_value(_STACKCUBE_B3_FIXED_SCENE)
    elif fixed_insertive_mass_kg is not None:
        fixed_scene_profile = {
            "randomize_insertive_object_mass": {
                "mass_distribution_params": [fixed_insertive_mass_kg, fixed_insertive_mass_kg]
            }
        }
    return {
        "disabled_events": disabled,
        "fixed_events": fixed,
        "fixed_scene_dynamics": stackcube_fixed_scene,
        "fixed_scene_profile": fixed_scene_profile,
    }


def _plain_value(value):
    """Convert config values to JSON-compatible zarr attributes."""
    if isinstance(value, dict):
        return {str(k): _plain_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _event_param(env_cfg, event_name, param_name):
    event = getattr(env_cfg.events, event_name, None)
    if event is None:
        return None
    return _plain_value(getattr(event, "params", {}).get(param_name))


def _camera_image_processing_config(env_cfg):
    group = env_cfg.observations.data_collection
    result = {}
    for observation_name in ("front_rgb", "side_rgb", "wrist_rgb"):
        term = getattr(group, observation_name, None)
        if term is None:
            result[observation_name] = None
            continue
        params = getattr(term, "params", {})
        camera_cfg = getattr(env_cfg.scene, observation_name.replace("_rgb", "_camera"))
        result[observation_name] = _plain_value(
            {
                "native_resolution_wh": [camera_cfg.width, camera_cfg.height],
                "output_size": params.get("output_size"),
                "target_intrinsics": params.get("target_intrinsics"),
                "vertical_flip": params.get("vertical_flip", False),
            }
        )
    return result


def _zarr_compressor(name: str):
    if name == "default":
        return numcodecs.Blosc(
            cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE
        )
    if name == "zstd":
        return numcodecs.Blosc(
            cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE
        )
    raise ValueError(f"unsupported zarr compressor: {name}")


class StreamingZarrWriter:
    """Append complete successful episodes so long quota runs remain recoverable."""

    def __init__(self, output_path, cam_keys, numeric_keys, collection_config):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.root = zarr.open(output_path, mode="w")
        self.root.attrs["collection_config"] = collection_config
        self.compressor = _zarr_compressor(collection_config["zarr_compressor"])
        self.data = self.root.create_group("data")
        self.meta = self.root.create_group("meta")
        self.episode_ends = self.meta.create_dataset(
            "episode_ends",
            shape=(0,),
            chunks=(1024,),
            dtype=np.int64,
            compressor=self.compressor,
        )
        self.reset_state_ids = (
            self.meta.create_dataset(
                "reset_state_ids",
                shape=(0,),
                chunks=(1024,),
                dtype=np.int64,
                compressor=self.compressor,
            )
            if "reset_state_id" in numeric_keys
            else None
        )
        self.cam_keys = tuple(cam_keys)
        self.numeric_keys = tuple(numeric_keys)
        self.collection_config = collection_config
        self.total_frames = 0
        self.num_episodes = 0

    def append(self, episode):
        length = len(episode["action"])
        if length < 2:
            raise ValueError("cannot append an episode shorter than two steps")
        for key in self.cam_keys + self.numeric_keys:
            dtype = np.uint8 if key in self.cam_keys else (
                np.int64 if key == "reset_state_id" else np.float32
            )
            array = np.asarray(episode[key], dtype=dtype)
            if key not in self.data:
                chunk_len = min(64 if key in self.cam_keys else 1024, length)
                self.data.create_dataset(
                    key,
                    shape=(0,) + array.shape[1:],
                    chunks=(chunk_len,) + array.shape[1:],
                    dtype=array.dtype,
                    compressor=self.compressor,
                )
            dataset = self.data[key]
            old_size = dataset.shape[0]
            dataset.resize((old_size + length,) + dataset.shape[1:])
            dataset[old_size : old_size + length] = array

        if self.collection_config["gripper_supervision"] in {
            "grasp_guard_binary",
            "grasp_guard_binary_verified",
        }:
            gripper = np.asarray(episode["action"])[:, -1]
            if not np.all(np.isin(gripper, (-1.0, 1.0))):
                raise ValueError("episode contains an invalid binary grasp-guard label")

        if self.collection_config["gripper_supervision"] in {
            "grasp_guard_joint_width_target",
            "fixed_open_joint_width_target",
            "policy_binary_joint_width_target",
        }:
            gripper = np.asarray(episode["action"])[:, -1]
            valid = np.isclose(gripper, 0.08, atol=1e-6)
            if self.collection_config["gripper_supervision"] in {
                "grasp_guard_joint_width_target",
                "policy_binary_joint_width_target",
            }:
                valid |= np.isclose(gripper, 0.0, atol=1e-6)
            if not np.all(valid):
                raise ValueError("episode contains an invalid gripper width target")

        self.total_frames += length
        self.num_episodes += 1
        self.episode_ends.resize((self.num_episodes,))
        self.episode_ends[-1] = self.total_frames
        if self.reset_state_ids is not None:
            state_ids = np.asarray(episode["reset_state_id"], dtype=np.int64).reshape(-1)
            if not np.all(state_ids == state_ids[0]):
                raise ValueError("reset_state_id changed within an episode")
            self.reset_state_ids.resize((self.num_episodes,))
            self.reset_state_ids[-1] = int(state_ids[0])

    def close(self):
        for key in self.data.keys():
            if self.data[key].shape[0] != self.total_frames:
                raise ValueError(
                    f"streamed array {key} has {self.data[key].shape[0]} frames, "
                    f"expected {self.total_frames}"
                )
        print(
            f"[vis-collect] incrementally saved {self.num_episodes} demos, "
            f"{self.total_frames} frames -> {self.root.store.path}"
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    global _COLLECT_RESET_EVENT_CFG
    from peg_gray_contract import assert_peg_gray_contract

    assert_peg_gray_contract(
        env_cfg, preserve_object_face_materials=args_cli.preserve_object_face_materials
    )
    agent_cfg = _rsl_cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # CRUX: make the policy group the pure-state teacher obs (so policy(obs) feeds the teacher);
    # images stay in the data_collection group.
    env_cfg.observations.policy = StateObsCfg.PolicyCfg()
    if "FrankaResearch3" in args_cli.task:
        remap_research3_names(env_cfg)
    print("[vis-collect] reassigned env.observations.policy -> State PolicyCfg (pure state for teacher)")

    if args_cli.disable_corrupted_camera_termination:
        env_cfg.terminations.corrupted_camera = None
        print(
            "[vis-collect] WARNING: corrupted-camera termination disabled for a controlled "
            "diagnostic; this run is not eligible for canonical vision data"
        )

    joint_label_mode = args_cli.joint_target_bridge or args_cli.osc_joint_trace or args_cli.joint_teacher
    indexed_reset_mode = not args_cli.online_xy5_t0_reset and (
        joint_label_mode
        or args_cli.native_osc_policy_gripper
        or args_cli.native_osc_guard_labels
    )

    if args_cli.native_osc_guard_labels:
        env_cfg.actions.arm.max_linear_velocity = args_cli.osc_max_linear_velocity
        env_cfg.actions.arm.max_angular_velocity = args_cli.osc_max_angular_velocity
        limit_description = (
            "disabled"
            if args_cli.osc_max_linear_velocity is None
            else (
                f"linear={args_cli.osc_max_linear_velocity:g}m/s, "
                f"angular={args_cli.osc_max_angular_velocity:g}rad/s"
            )
        )
        print(
            "[vis-collect] native OSC guard-label mode ON: guard controls the demonstration "
            "gripper and its executed command becomes the BC label; student eval must use a "
            f"policy-controlled gripper; reference limits={limit_description}"
        )

    if args_cli.native_osc_policy_gripper:
        env_cfg.actions.gripper = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        env_cfg.actions.arm.max_linear_velocity = args_cli.osc_max_linear_velocity
        env_cfg.actions.arm.max_angular_velocity = args_cli.osc_max_angular_velocity
        limit_description = (
            "disabled"
            if args_cli.osc_max_linear_velocity is None
            else (
                f"linear={args_cli.osc_max_linear_velocity:g}m/s, "
                f"angular={args_cli.osc_max_angular_velocity:g}rad/s"
            )
        )
        print(
            "[vis-collect] native OSC policy-gripper ON: teacher arm action and gripper sign are "
            f"executed without guard/fixed-open rewriting; reference limits={limit_description}"
        )

    if args_cli.joint_teacher:
        env_cfg.actions = FrankaFr3GripperRelativeJointTargetAction()
        env_cfg.actions.arm.max_joint_velocity = (args_cli.joint_max_velocity,) * 7
        for actuator_name in ("panda_arm1", "panda_arm2"):
            actuator = env_cfg.scene.robot.actuators[actuator_name]
            actuator.stiffness = args_cli.joint_position_stiffness
            actuator.damping = args_cli.joint_position_damping
        hand = env_cfg.scene.robot.actuators["panda_hand"]
        hand.stiffness = 1000.0
        hand.damping = 14.0
        hand.effort_limit_sim = 60.0
        env_cfg.episode_length_s = 16.0
        if getattr(env_cfg.events, "randomize_osc_gains", None) is not None:
            env_cfg.events.randomize_osc_gains = None
        print(
            "[vis-collect] joint teacher ON: normalized joint deltas execute directly through "
            f"joint PD, max_velocity={args_cli.joint_max_velocity:.3f}rad/s, "
            f"arm Kp/Kd={args_cli.joint_position_stiffness:g}/{args_cli.joint_position_damping:g}; "
            "gripper is policy-controlled"
        )

    if args_cli.osc_joint_trace:
        env_cfg.actions.arm.max_linear_velocity = args_cli.osc_max_linear_velocity
        env_cfg.actions.arm.max_angular_velocity = args_cli.osc_max_angular_velocity
        if args_cli.osc_joint_trace_gripper_mode != "grasp_guard":
            env_cfg.actions.gripper = BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": 0.0},
            )
        print(
            "[vis-collect] OSC joint-trace ON: native Cartesian OSC execution, "
            f"gripper_mode={args_cli.osc_joint_trace_gripper_mode}, "
            f"{args_cli.osc_joint_trace_target} joint labels stored as "
            "replay-gated labels; "
            f"reference limits={args_cli.osc_max_linear_velocity}/{args_cli.osc_max_angular_velocity}, "
            f"joint label Kp/Kd={args_cli.joint_position_stiffness:g}/{args_cli.joint_position_damping:g}, "
            f"label max_velocity={args_cli.joint_max_velocity}"
        )

    if args_cli.joint_target_bridge:
        if args_cli.joint_max_velocity <= 0.0:
            raise ValueError(f"--joint_max_velocity must be positive, got {args_cli.joint_max_velocity}")
        legacy_arm = env_cfg.actions.arm
        reference_blend = (
            legacy_arm.action_reference_blend
            if args_cli.joint_action_reference_blend is None
            else args_cli.joint_action_reference_blend
        )
        env_cfg.actions.arm = RelCartesianDiffIKJointPositionActionCfg(
            asset_name=legacy_arm.asset_name,
            joint_names=list(legacy_arm.joint_names),
            body_name=legacy_arm.body_name,
            jacobian_source=legacy_arm.jacobian_source,
            simulation_jacobian_point=args_cli.joint_simulation_jacobian_point,
            scale_xyz_axisangle=tuple(legacy_arm.scale_xyz_axisangle),
            input_clip=legacy_arm.input_clip,
            action_reference_blend=reference_blend,
            motion_stiffness=tuple(legacy_arm.motion_stiffness),
            motion_damping_ratio=tuple(legacy_arm.motion_damping_ratio),
            torque_limit=tuple(legacy_arm.torque_limit),
            nullspace_stiffness=legacy_arm.nullspace_stiffness,
            nullspace_damping_ratio=legacy_arm.nullspace_damping_ratio,
            nullspace_default_pos=legacy_arm.nullspace_default_pos,
            ik_damping=args_cli.joint_ik_damping,
            ik_step_scale=args_cli.joint_ik_step_scale,
            max_joint_velocity=(args_cli.joint_max_velocity,) * 7,
            joint_limit_margin=0.01,
        )
        if args_cli.fixed_open_gripper:
            env_cfg.actions.gripper = BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": 0.0},
            )
        for actuator_name in ("panda_arm1", "panda_arm2"):
            actuator = env_cfg.scene.robot.actuators[actuator_name]
            actuator.stiffness = args_cli.joint_position_stiffness
            actuator.damping = args_cli.joint_position_damping
        env_cfg.episode_length_s = args_cli.joint_episode_length_s
        if getattr(env_cfg.events, "randomize_osc_gains", None) is not None:
            env_cfg.events.randomize_osc_gains = None
        print(
            "[vis-collect] joint-target bridge ON: legacy Cartesian teacher -> DLS IK -> "
            f"absolute q target; max_velocity={args_cli.joint_max_velocity:.3f}rad/s, "
            f"arm Kp/Kd={args_cli.joint_position_stiffness:g}/{args_cli.joint_position_damping:g}, "
            f"ik_damping={args_cli.joint_ik_damping:g}, "
            f"ik_step_scale={args_cli.joint_ik_step_scale:g}, "
            f"jacobian_point={args_cli.joint_simulation_jacobian_point}, "
            f"reference_blend={reference_blend:g}, episode={args_cli.joint_episode_length_s:g}s"
        )

    forced_reset_state_indices = _load_reset_state_indices(args_cli.reset_state_indices_file)
    forced_unique_state_ids = (
        list(dict.fromkeys(forced_reset_state_indices))
        if forced_reset_state_indices is not None
        else []
    )
    joint_home_arm = (
        tuple(float(value) for value in args_cli.joint_home_arm)
        if args_cli.joint_home_arm is not None
        else _TEAM_HOME_ARM
    )

    if args_cli.online_xy5_t0_reset:
        reset_event = env_cfg.events.reset_from_reset_states
        if reset_event.func is not task_mdp.StackCubeXY5T0OnlineReset:
            raise TypeError(
                "--online_xy5_t0_reset requires a task configured with "
                "StackCubeXY5T0OnlineReset"
            )
        reset_types = ["stackcube_xy5_t0_online_continuous"]
        print(
            "[vis-collect] reset: continuous XY5 t0 objects + team-home arm-joint jitter; "
            "no discrete state pool"
        )
    elif indexed_reset_mode:
        reset_types = args_cli.reset_types if args_cli.reset_types else [_TEAM_HOME_RESET_TYPE]
        if len(reset_types) != 1:
            raise ValueError(
                "indexed collection requires exactly one reset type; "
                f"got {reset_types!r}"
            )
        env_cfg.events.reset_from_reset_states.params["reset_types"] = reset_types
        env_cfg.events.reset_from_reset_states.params["probs"] = [1.0]
        if forced_reset_state_indices is not None:
            env_cfg.events.reset_from_reset_states.params["state_indices"] = forced_reset_state_indices
            env_cfg.events.reset_from_reset_states.params["state_index_repeats"] = args_cli.reset_state_repeats
            print(
                f"[vis-collect] forced reset states: {len(forced_reset_state_indices)} indices x "
                f"{args_cli.reset_state_repeats} repeats"
            )
        if joint_label_mode:
            print(
                f"[vis-collect] reset_types = {reset_types}; "
                f"required_home_arm={list(joint_home_arm)}"
            )
        else:
            print(f"[vis-collect] reset_types = {reset_types}; indexed native-OSC collection")
    elif args_cli.four_path:  # faithful to original DataCollectionRGBEventCfg (4-path 0.25 each)
        reset_types = [
            "ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"]
        env_cfg.events.reset_from_reset_states.params["reset_types"] = reset_types
        env_cfg.events.reset_from_reset_states.params["probs"] = [0.25, 0.25, 0.25, 0.25]
        print("[vis-collect] reset: 4-path mix (faithful)")
    else:
        reset_types = args_cli.reset_types if args_cli.reset_types else ["ObjectAnywhereEEAnywhere"]
        env_cfg.events.reset_from_reset_states.params["reset_types"] = reset_types
        env_cfg.events.reset_from_reset_states.params["probs"] = [1.0 / len(reset_types)] * len(reset_types)
        print(f"[vis-collect] reset_types = {reset_types} (probs uniform)")
    if not args_cli.online_xy5_t0_reset:
        _COLLECT_RESET_EVENT_CFG = env_cfg.events.reset_from_reset_states
        _COLLECT_RESET_EVENT_CFG.func = collect_reset_from_reset_states
        print("[vis-collect] reset_from_reset_states wrapped for reset-mode ManagerTerm compatibility")

    if args_cli.preserve_object_face_materials:
        removed = []
        for name in sorted(_OBJECT_APPEARANCE_EVENTS):
            if getattr(env_cfg.events, name, None) is not None:
                setattr(env_cfg.events, name, None)
                removed.append(name)
        print(f"[vis-collect] object asset face materials preserved: disabled {removed}")

    sample_hdri_config = os.environ.get("OMNIRESET_SAMPLE_HDRI_CONFIG")
    if sample_hdri_config:
        # Render-only fallback used by the frozen Peg collection bundle.
        # Curtains draw from the production color branch and the HDRI is
        # restricted to the supplied production manifest.
        for name in list(vars(env_cfg.events)):
            event = getattr(env_cfg.events, name)
            if event is None:
                continue
            params = getattr(event, "params", None)
            if not params:
                continue
            if "curtain" in name and "appearance" in name:
                params["texture_prob"] = 0.0
                params.pop("texture_config_path", None)
                params.pop("texture_paths", None)
            elif "sky" in name or "hdri" in name:
                params["hdri_config_path"] = sample_hdri_config
        print(
            "[vis-collect] render-sample visual assets: color-only curtain "
            f"realization + manifest HDRI {sample_hdri_config}"
        )

    if args_cli.no_visual_dr:
        # NEUTRALIZE (don't delete) visual DR -> deterministic images that STILL RENDER.
        # CRITICAL: `randomize_tiled_cameras` both *configures* the cameras (base_position/
        # rotation) and jitters them. Deleting it leaves the cameras unconfigured, so the first
        # data_collection read hangs forever. So we keep camera/light events and only zero their
        # random ranges. Peg/hole retain a deterministic visibility material; other appearance
        # events are removed.
        _killed, _fixed = [], []
        for n in list(vars(env_cfg.events)):
            ev = getattr(env_cfg.events, n)
            if ev is None:
                continue
            params = getattr(ev, "params", None)
            if n in _OBJECT_APPEARANCE_EVENTS and not args_cli.keep_appearance:
                _freeze_grey_object_appearance(ev)
                _fixed.append(f"{n}:grey-startup")
            elif "appearance" in n:                     # texture/color
                if not args_cli.keep_appearance:
                    setattr(env_cfg.events, n, None)
                    _killed.append(n)
                # else: leave appearance randomizing so the scene stays visible
            elif "focal" in n and params and "focal_length_range" in params:
                lo, hi = params["focal_length_range"]
                params["focal_length_range"] = ((lo + hi) / 2.0, (lo + hi) / 2.0)
                _fixed.append(n)
            elif "camera" in n and params:              # randomize_tiled_cameras: keep base pose, zero jitter
                for dk in ("position_deltas", "euler_deltas"):
                    if dk in params and isinstance(params[dk], dict):
                        params[dk] = {k: (0.0, 0.0) for k in params[dk]}
                _fixed.append(n)
            elif ("sky" in n or "hdri" in n) and params:
                if args_cli.no_all_dr:
                    # The scene already owns a fixed DomeLight. Removing this
                    # interval term prevents random HDRI texture/orientation swaps.
                    setattr(env_cfg.events, n, None)
                    _killed.append(n)
                else:
                    ev.mode = "startup"
                    if "intensity_range" in params:
                        lo, hi = params["intensity_range"]
                        params["intensity_range"] = ((lo + hi) / 2.0, (lo + hi) / 2.0)
                    if "rotation_range" in params:
                        params["rotation_range"] = (0.0, 0.0)
                    # The leading NVIDIA entries are 4K HDRIs and can stall RTX
                    # initialization. Index 40 is the first cached 1K HDRI.
                    params["fixed_hdri_index"] = 40
                    _fixed.append(f"{n}:fixed-startup")
        print(f"[vis-collect] visual DR OFF (deterministic): dropped appearance {_killed}; "
              f"zeroed-jitter (kept setup) {_fixed}")
        if getattr(env_cfg.events, "randomize_sky_light", None) is not None:
            env_cfg.events.randomize_sky_light = None
            print("[vis-collect] sky-light startup randomization disabled for no_visual_dr collection")
    else:
        randomized_visuals = (
            "camera/curtain randomize per episode"
            if args_cli.preserve_object_face_materials
            else "camera/object/curtain randomize per episode"
        )
        print(
            f"[vis-collect] visual DR ON: {randomized_visuals}; "
            "global HDRI randomizes once at process startup"
        )

    freeze_nonvisual = (
        args_cli.no_dynamics_dr
        or args_cli.stackcube_fixed_scene
        or args_cli.fixed_insertive_mass_kg is not None
    )
    if freeze_nonvisual:
        nonvisual_profile = neutralize_nonvisual_dr(
            env_cfg,
            stackcube_fixed_scene=args_cli.stackcube_fixed_scene,
            fixed_insertive_mass_kg=args_cli.fixed_insertive_mass_kg,
        )
    else:
        nonvisual_profile = {
            "disabled_events": [],
            "fixed_events": [],
            "fixed_scene_dynamics": False,
            "fixed_scene_profile": {},
        }
        print("[vis-collect] non-visual DR ON (default): dynamics/material/mass randomization active")

    hand_actuator = env_cfg.scene.robot.actuators.get("panda_hand")
    arm_action = env_cfg.actions.arm
    if args_cli.joint_target_bridge:
        gripper_supervision = (
            "fixed_open_joint_width_target"
            if args_cli.fixed_open_gripper
            else "grasp_guard_joint_width_target"
        )
    elif args_cli.osc_joint_trace:
        gripper_supervision = {
            "fixed_open": "fixed_open_joint_width_target",
            "policy_sign": "policy_teacher_sign_joint_width_target",
            "grasp_guard": "grasp_guard_joint_width_target",
        }[args_cli.osc_joint_trace_gripper_mode]
    elif args_cli.joint_teacher:
        gripper_supervision = "policy_binary_joint_width_target"
    elif args_cli.native_osc_guard_labels:
        gripper_supervision = "grasp_guard_binary_verified"
    elif args_cli.native_osc_policy_gripper:
        gripper_supervision = "policy_teacher_sign_executed"
    else:
        gripper_supervision = "raw_teacher_ignored" if args_cli.raw_teacher_gripper else "grasp_guard_binary"
    if args_cli.native_osc_guard_labels:
        schema_version = 10
    elif args_cli.native_osc_policy_gripper:
        schema_version = 9
    elif args_cli.joint_teacher:
        schema_version = 8
    elif args_cli.osc_joint_trace:
        schema_version = 7
    elif args_cli.joint_target_bridge:
        schema_version = 4
    else:
        schema_version = 3
    for camera_key in ("front_rgb", "side_rgb", "wrist_rgb"):
        camera_term = getattr(env_cfg.observations.data_collection, camera_key)
        camera_term.params["output_size"] = (args_cli.image_size, args_cli.image_size)

    receptive_object_scale = float(args_cli.receptive_object_uniform_scale)
    env_cfg.scene.receptive_object.spawn.scale = (receptive_object_scale,) * 3
    resolved_receptive_scale = tuple(
        float(value) for value in env_cfg.scene.receptive_object.spawn.scale
    )
    if resolved_receptive_scale != (receptive_object_scale,) * 3:
        raise RuntimeError(
            "receptive-object scale contract was overwritten after explicit resolution: "
            f"requested={receptive_object_scale}, resolved={resolved_receptive_scale}"
        )
    print(
        "[vis-collect] receptive-object uniform spawn scale explicitly resolved after "
        f"task/object variants: {resolved_receptive_scale}"
    )
    collection_config = {
        "schema_version": schema_version,
        "source_commit": os.environ.get("SOURCE_COMMIT"),
        "object_base_rgb": list(_OBJECT_GREY),
        "zarr_compressor": args_cli.zarr_compressor,
        "task": args_cli.task,
        "robot_asset_contract": (
            "research3_mimic_fingertip_convex_hull"
            if "FrankaResearch3MimicFingertip" in args_cli.task
            else ("research3_official" if "FrankaResearch3" in args_cli.task else "franka_mimic")
        ),
        "robot_visual_profile": (
            os.environ.get("OMNIRESET_FR3_VISUAL_PROFILE", "official")
            if "FrankaResearch3" in args_cli.task
            else "franka_mimic_official"
        ),
        "teacher_checkpoint": args_cli.teacher_ckpt,
        "teacher_checkpoint_sha256": _TEACHER_CHECKPOINT_SHA256,
        "teacher_policy_action_mode": (
            "stochastic_sample" if args_cli.stochastic_teacher_actions else "deterministic_mean"
        ),
        "control_mode": "absolute_joint_position_target" if joint_label_mode else "relative_cartesian_osc",
        "demonstration_execution_control_mode": (
            "relative_joint_target"
            if args_cli.joint_teacher
            else (
                "relative_cartesian_diffik_absolute_joint_target"
                if args_cli.joint_target_bridge
                else (
                    "relative_cartesian_osc"
                    if (
                        args_cli.osc_joint_trace
                        or args_cli.native_osc_policy_gripper
                        or args_cli.native_osc_guard_labels
                    )
                    else None
                )
            )
        ),
        "action_layout": (
            ["q_target_1_rad", "q_target_2_rad", "q_target_3_rad", "q_target_4_rad",
             "q_target_5_rad", "q_target_6_rad", "q_target_7_rad", "gripper_width_target_m"]
            if joint_label_mode
            else ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_binary"]
        ),
        "proprio_layout": (
            ["q_1_rad", "q_2_rad", "q_3_rad", "q_4_rad", "q_5_rad", "q_6_rad", "q_7_rad",
             "gripper_width_m"]
            if joint_label_mode
            else ["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz", "gripper_width_m"]
        ),
        "stored_image_transforms": {
            camera_key: {
                "rotation_deg": 180,
                "operation": "rotate_180_before_zarr_write",
            }
            for camera_key in args_cli.rotate_180_camera
        },
        "home_arm_joint_position_rad": list(joint_home_arm) if joint_label_mode else None,
        "joint_home_check_enabled": bool(joint_label_mode and not args_cli.skip_joint_home_check),
        "joint_target_source": (
            "dls_ik_post_limit_actual_command"
            if args_cli.joint_target_bridge
            else (
                "policy_joint_delta_post_limit_actual_command"
                if args_cli.joint_teacher
                else (
                    (
                        "osc_torque_match_first_substep_candidate"
                        if args_cli.osc_joint_trace_target == "torque_match"
                        else "osc_achieved_joint_position_last_physics_substep_candidate"
                    )
                    if args_cli.osc_joint_trace
                    else None
                )
            )
        ),
        "osc_joint_trace_target": args_cli.osc_joint_trace_target if args_cli.osc_joint_trace else None,
        "osc_joint_trace_gripper_mode": (
            args_cli.osc_joint_trace_gripper_mode if args_cli.osc_joint_trace else None
        ),
        "candidate_joint_targets_require_replay_validation": bool(args_cli.osc_joint_trace),
        "replay_initial_state_schema": (
            "jointtarget_full_dynamics_v1" if args_cli.joint_target_bridge else None
        ),
        "joint_ik_damping": args_cli.joint_ik_damping if args_cli.joint_target_bridge else None,
        "joint_ik_step_scale": args_cli.joint_ik_step_scale if args_cli.joint_target_bridge else None,
        "joint_simulation_jacobian_point": (
            args_cli.joint_simulation_jacobian_point if args_cli.joint_target_bridge else None
        ),
        "joint_action_reference_blend": (
            _plain_value(getattr(arm_action, "action_reference_blend", None))
            if args_cli.joint_target_bridge
            else None
        ),
        "joint_max_velocity_rad_s": args_cli.joint_max_velocity if (args_cli.joint_target_bridge or args_cli.joint_teacher or args_cli.osc_joint_trace) else None,
        "joint_position_stiffness": args_cli.joint_position_stiffness if (args_cli.joint_target_bridge or args_cli.joint_teacher or args_cli.osc_joint_trace) else None,
        "joint_position_damping": args_cli.joint_position_damping if (args_cli.joint_target_bridge or args_cli.joint_teacher or args_cli.osc_joint_trace) else None,
        "osc_reference_max_linear_velocity_m_s": (
            args_cli.osc_max_linear_velocity
            if (
                args_cli.native_osc_policy_gripper
                or args_cli.native_osc_guard_labels
                or args_cli.osc_joint_trace
            )
            else None
        ),
        "osc_reference_max_angular_velocity_rad_s": (
            args_cli.osc_max_angular_velocity
            if (
                args_cli.native_osc_policy_gripper
                or args_cli.native_osc_guard_labels
                or args_cli.osc_joint_trace
            )
            else None
        ),
        "control_dt_s": float(env_cfg.decimation * env_cfg.sim.dt),
        "teacher_action_hold_steps": int(args_cli.action_hold_steps),
        "teacher_decision_dt_s": float(
            args_cli.action_hold_steps * env_cfg.decimation * env_cfg.sim.dt
        ),
        "episode_length_s": args_cli.joint_episode_length_s if args_cli.joint_target_bridge else env_cfg.episode_length_s,
        "camera_setup": args_cli.camera_setup,
        "real_camera_profile": os.environ.get("OMNIRESET_REAL_CAMERA_PROFILE", "vision_dp_20260715"),
        "camera_render_profile": os.environ.get("OMNIRESET_CAMERA_RENDER_PROFILE", "native"),
        "exact_camera_intrinsics": os.environ.get("OMNIRESET_EXACT_CAMERA_INTRINSICS", "0") == "1",
        "camera_image_processing": _camera_image_processing_config(env_cfg),
        "camera_pose_randomization": {
            camera_name: {
                "position_deltas_m": _event_param(
                    env_cfg, f"randomize_{camera_name}_camera", "position_deltas"
                ),
                "euler_deltas_deg": _event_param(
                    env_cfg, f"randomize_{camera_name}_camera", "euler_deltas"
                ),
                "focal_length_range": _event_param(
                    env_cfg, f"randomize_{camera_name}_camera_focal_length", "focal_length_range"
                ),
            }
            for camera_name in ("front", "side", "wrist")
        },
        "reset_types": list(reset_types),
        "reset_artifact_sha256": os.environ.get("RESET_ARTIFACT_SHA256"),
        "reset_mode": "online_continuous" if args_cli.online_xy5_t0_reset else "state_pool",
        "reset_rigid_object_position_offsets": _event_param(
            env_cfg, "reset_from_reset_states", "rigid_object_position_offsets"
        ),
        "forced_reset_state_indices": forced_reset_state_indices,
        "reset_state_repeats": args_cli.reset_state_repeats if forced_reset_state_indices is not None else None,
        "complete_forced_state_set": bool(args_cli.complete_forced_state_set),
        "seed": int(args_cli.seed),
        "visual_dr_enabled": not args_cli.no_visual_dr,
        "visual_dr_schedule": (
            (
                "camera_curtain_per_episode_hdri_per_process"
                if args_cli.preserve_object_face_materials
                else "camera_object_curtain_per_episode_hdri_per_process"
            )
            if not args_cli.no_visual_dr
            else "deterministic"
        ),
        "render_asset_fallback": (
            {
                "curtain_branch": "color_only",
                "hdri_config_path": sample_hdri_config,
                "hdri_config_sha256": _file_sha256(sample_hdri_config),
            }
            if sample_hdri_config
            else None
        ),
        "dynamics_dr_enabled": not freeze_nonvisual,
        "fixed_insertive_mass_kg": args_cli.fixed_insertive_mass_kg,
        "preserve_object_face_materials": bool(args_cli.preserve_object_face_materials),
        "full_render": os.environ.get("VIS_FULL_RENDER", "0") == "1",
        "images_stored": not args_cli.no_store_images,
        "image_size": int(args_cli.image_size),
        "post_reset_warmup_steps": int(args_cli.post_reset_warmup_steps),
        "successful_repeats_per_state": int(args_cli.successful_repeats_per_state),
        "successful_episodes_only": not args_cli.save_all_episodes,
        "outcomes_only": bool(args_cli.outcomes_only),
        "record_placement_metrics": bool(
            args_cli.record_placement_metrics or args_cli.save_fixed_gate_episodes
        ),
        "save_fixed_gate_episodes": bool(args_cli.save_fixed_gate_episodes),
        "truncate_to_fixed_gate": bool(args_cli.truncate_to_fixed_gate),
        "fixed_gate_position_m": float(args_cli.diagnostic_position_m),
        "fixed_gate_orientation_deg": float(args_cli.diagnostic_orientation_deg),
        "corrupted_camera_termination_enabled": (
            not args_cli.disable_corrupted_camera_termination and not args_cli.no_store_images
        ),
        "success_position_override_m": args_cli.success_position_override_m,
        "success_orientation_override_deg": args_cli.success_orientation_override_deg,
        "object_table_z_offset_m": float(os.environ.get("OMNIRESET_REAL_OBJECT_TABLE_Z_OFFSET", "0.0")),
        "object_table_cover_size_m": _plain_value(
            getattr(
                getattr(getattr(env_cfg.scene, "table_cover", None), "spawn", None),
                "size",
                None,
            )
        ),
        "table_cover_collision_enabled": os.environ.get("OMNIRESET_REAL_TABLE_COVER_COLLISION", "0") == "1",
        "receptive_object_z_lift_m": float(
            os.environ.get("OMNIRESET_REAL_RECEPTIVE_OBJECT_Z_LIFT", "0.0")
        ),
        "insertive_object_spawn_scale_xyz": _plain_value(
            env_cfg.scene.insertive_object.spawn.scale
        ),
        "insertive_object_uniform_scale": float(
            env_cfg.scene.insertive_object.spawn.scale[0]
        ),
        "receptive_object_spawn_scale_xyz": _plain_value(
            env_cfg.scene.receptive_object.spawn.scale
        ),
        "receptive_object_uniform_scale": float(
            env_cfg.scene.receptive_object.spawn.scale[0]
        ),
        "physics_profile": (
            "b3_nominal_fixed"
            if args_cli.b3_nominal_fixed_scene
            else (
                "stackcube_b3_fixed"
                if args_cli.stackcube_fixed_scene
                else (
                    "b3_nominal_fixed_insertive_mass"
                    if args_cli.fixed_insertive_mass_kg is not None
                    else ("b3_nominal_fixed" if args_cli.no_dynamics_dr else "b3_domain_randomized")
                )
            )
        ),
        "panda_hand_stiffness": _plain_value(getattr(hand_actuator, "stiffness", None)),
        "panda_hand_damping": _plain_value(getattr(hand_actuator, "damping", None)),
        "panda_hand_effort_limit_sim": _plain_value(getattr(hand_actuator, "effort_limit_sim", None)),
        "osc_motion_stiffness": _plain_value(getattr(arm_action, "motion_stiffness", None)),
        "osc_motion_damping_ratio": _plain_value(getattr(arm_action, "motion_damping_ratio", None)),
        "gripper_stiffness_scale_range": _event_param(
            env_cfg, "randomize_gripper_actuator_parameters", "stiffness_distribution_params"
        ),
        "gripper_damping_scale_range": _event_param(
            env_cfg, "randomize_gripper_actuator_parameters", "damping_distribution_params"
        ),
        "osc_gain_scale_range": _event_param(env_cfg, "randomize_osc_gains", "scale_range"),
        "arm_sysid_scale_range": _event_param(env_cfg, "randomize_arm_sysid", "scale_range"),
        "arm_sysid_delay_range": _event_param(env_cfg, "randomize_arm_sysid", "delay_range"),
        "gripper_supervision": gripper_supervision,
        "gripper_action_close": 0.0 if joint_label_mode else -1.0,
        "gripper_action_open": 0.08 if joint_label_mode else 1.0,
        "gripper_action_std": (
            "teacher" if args_cli.raw_teacher_gripper else 0.0
        ),
        "raw_teacher_gripper_preserved": bool(
            args_cli.native_osc_policy_gripper or args_cli.native_osc_guard_labels
        ),
        "guard_used_during_demonstration": gripper_supervision in {
            "grasp_guard_binary",
            "grasp_guard_binary_verified",
            "grasp_guard_joint_width_target",
        },
        "guard_required_at_student_eval": (
            False
            if gripper_supervision
            in {
                "grasp_guard_binary",
                "grasp_guard_binary_verified",
                "grasp_guard_joint_width_target",
            }
            else None
        ),
        "guard_execution_verified_on_nonterminal_frames": bool(
            args_cli.native_osc_guard_labels
            or (
                args_cli.osc_joint_trace
                and args_cli.osc_joint_trace_gripper_mode == "grasp_guard"
            )
            or (args_cli.joint_target_bridge and not args_cli.fixed_open_gripper)
        ),
        "policy_gripper_eval_required": not (
            args_cli.raw_teacher_gripper or args_cli.native_osc_policy_gripper
        ),
        **nonvisual_profile,
    }
    print(f"[vis-collect] collection_config={collection_config}")
    if args_cli.rotate_180_camera:
        print(
            "[vis-collect] stored RGB rotation: "
            f"180deg for {sorted(args_cli.rotate_180_camera)} before Zarr write"
        )

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.no_store_images:
        for camera_key in ("front_rgb", "side_rgb", "wrist_rgb"):
            setattr(env_cfg.observations.data_collection, camera_key, None)
        for sensor_name in ("front_camera", "side_camera", "wrist_camera"):
            setattr(env_cfg.scene, sensor_name, None)
        for event_name in (
            "randomize_front_camera",
            "randomize_front_camera_focal_length",
            "randomize_side_camera",
            "randomize_side_camera_focal_length",
            "randomize_wrist_camera",
            "randomize_wrist_camera_focal_length",
        ):
            setattr(env_cfg.events, event_name, None)
        env_cfg.terminations.corrupted_camera = None
        print("[vis-collect] trajectory-selection fast path: camera sensors/events removed")

    # Fast render: the RGB cfg forces DLSS-RR denoiser + DLAA + raytracing, which HANGS on
    # A100 (Isaac warns DLSS-RR is buggy on this GPU). Disable for pipeline validation; set
    # VIS_FULL_RENDER=1 to keep the photorealistic path for final sim2real-quality collection.
    if os.environ.get("VIS_FULL_RENDER", "0") != "1":
        env_cfg.sim.render.enable_dl_denoiser = False
        env_cfg.sim.render.enable_reflections = False
        env_cfg.sim.render.enable_ambient_occlusion = False
        env_cfg.sim.render.antialiasing_mode = "Off"
        print("[vis-collect] fast render (no DLSS-RR/DLAA/reflections) — set VIS_FULL_RENDER=1 to disable")

    if assert_peg_gray_contract(env_cfg):
        print("[peg-gray-contract] PASS: Peg/PegHole #8E9089; active all-mesh gray appearance events")
    env = gym.make(args_cli.task, cfg=env_cfg)
    if args_cli.camera_setup == "real":
        _hide_original_table_visual(args_cli.num_envs)
    env = RslRlVecEnvWrapper(env)
    if args_cli.success_position_override_m is not None:
        task_command = env.unwrapped.command_manager.get_term("task_command")
        original_position_m = float(task_command.success_position_threshold)
        original_orientation_deg = float(np.rad2deg(task_command.success_orientation_threshold))
        task_command.success_position_threshold = args_cli.success_position_override_m
        task_command.success_orientation_threshold = float(
            np.deg2rad(args_cli.success_orientation_override_deg)
        )
        print(
            "[vis-collect] success threshold override: "
            f"position {original_position_m * 1000:.2f} -> "
            f"{args_cli.success_position_override_m * 1000:.2f}mm, "
            f"orientation {original_orientation_deg:.2f} -> "
            f"{args_cli.success_orientation_override_deg:.2f}deg"
        )
    device = env.unwrapped.device
    runtime_object_masses = {}
    for asset_name in ("insertive_object", "receptive_object"):
        masses = (
            env.unwrapped.scene[asset_name]
            .root_physx_view.get_masses()
            .detach()
            .cpu()
            .numpy()
        )
        runtime_object_masses[asset_name] = {
            "min": float(masses.min()),
            "max": float(masses.max()),
        }
    collection_config["runtime_object_mass_kg"] = runtime_object_masses
    print(f"[vis-collect] runtime object masses (kg): {runtime_object_masses}")
    expected_insertive_mass = (
        0.11 if args_cli.stackcube_fixed_scene else args_cli.fixed_insertive_mass_kg
    )
    if expected_insertive_mass is not None:
        insertive_mass = runtime_object_masses["insertive_object"]
        if not (
            abs(insertive_mass["min"] - expected_insertive_mass) <= 1.0e-6
            and abs(insertive_mass["max"] - expected_insertive_mass) <= 1.0e-6
        ):
            raise RuntimeError(
                f"fixed scene requires runtime insertive-object mass {expected_insertive_mass:g} kg; "
                f"got {insertive_mass}"
            )
    arm_term = env.unwrapped.action_manager.get_term("arm")
    gripper_term = env.unwrapped.action_manager.get_term("gripper")
    reset_term = None if args_cli.online_xy5_t0_reset else get_collect_reset_manager(env.unwrapped)
    if args_cli.joint_target_bridge:
        required = (
            "last_ik_joint_position_targets",
            "last_joint_position_targets",
            "last_applied_joint_position_targets",
            "last_ik_valid",
        )
        missing = [name for name in required if not hasattr(arm_term, name)]
        if missing:
            raise TypeError(f"joint-target bridge action term is missing diagnostics: {missing}")
        if args_cli.action_hold_steps > 1:
            if not hasattr(arm_term, "set_hold_joint_position_target"):
                raise TypeError("joint-target bridge cannot freeze its processed q target")
            if not hasattr(gripper_term, "set_hold_processed_action"):
                raise TypeError("grasp guard cannot freeze its processed gripper command")
            print(
                "[vis-collect] stop-go execution: teacher/DLS update every "
                f"{args_cli.action_hold_steps} env steps; exact q/gripper targets held between updates"
            )
        if not args_cli.online_xy5_t0_reset and not hasattr(reset_term, "state_id"):
            raise TypeError("reset manager does not expose reset state_id")
        print("[vis-collect] joint-target logging: q_target is read back from the action term after env.step")
    elif args_cli.joint_teacher:
        missing = [
            name for name in ("requested_actions", "last_applied_actions")
            if not hasattr(arm_term, name)
        ]
        if missing:
            raise TypeError(f"joint-teacher action term is missing diagnostics: {missing}")
        if not args_cli.online_xy5_t0_reset and not hasattr(reset_term, "state_id"):
            raise TypeError("reset manager does not expose reset state_id")
        if hasattr(gripper_term, "compute_rule_actions"):
            raise TypeError("joint teacher must use a policy-controlled gripper, not a grasp guard")
        print("[vis-collect] joint-teacher logging: exact post-limit q target is read after env.step")
    elif args_cli.osc_joint_trace:
        missing = [
            name
            for name in ("joint_pos_substeps", "joint_vel_substeps", "joint_torque_substeps")
            if not hasattr(arm_term, name)
        ]
        if missing:
            raise TypeError(f"OSC joint-trace action term is missing diagnostics: {missing}")
        if not args_cli.online_xy5_t0_reset and not hasattr(reset_term, "state_id"):
            raise TypeError("reset manager does not expose reset state_id")
        uses_guard = hasattr(gripper_term, "compute_rule_actions")
        if args_cli.osc_joint_trace_gripper_mode == "grasp_guard":
            if not uses_guard:
                raise TypeError("OSC joint-trace grasp_guard mode requires compute_rule_actions()")
        elif uses_guard:
            raise TypeError(
                "OSC joint-trace fixed_open/policy_sign modes must use a policy-controlled gripper"
            )
        print(
            "[vis-collect] OSC joint-trace logging: native OSC substep q/dq/tau diagnostics "
            f"available; target={args_cli.osc_joint_trace_target}"
        )
    elif args_cli.native_osc_policy_gripper:
        missing = [
            name
            for name in ("ee_velocity_substeps", "ee_reference_velocity_substeps")
            if not hasattr(arm_term, name)
        ]
        if missing:
            raise TypeError(f"native OSC action term is missing velocity diagnostics: {missing}")
        if not args_cli.online_xy5_t0_reset and not hasattr(reset_term, "state_id"):
            raise TypeError("reset manager does not expose reset state_id")
        if hasattr(gripper_term, "compute_rule_actions"):
            raise TypeError("native OSC policy-gripper mode must not use a grasp guard")
        print("[vis-collect] native OSC logging: teacher gripper command is verified after env.step")
    if (
        args_cli.osc_joint_trace
        and args_cli.osc_joint_trace_gripper_mode == "grasp_guard"
        and not hasattr(gripper_term, "compute_rule_actions")
    ):
        raise TypeError("OSC joint-trace grasp_guard mode requires a gripper term with compute_rule_actions()")
    if (
        not args_cli.raw_teacher_gripper
        and not args_cli.osc_joint_trace
        and not args_cli.joint_teacher
        and not args_cli.native_osc_policy_gripper
        and not args_cli.fixed_open_gripper
        and not hasattr(gripper_term, "compute_rule_actions")
    ):
        raise TypeError(
            "rule-supervised gripper collection requires a gripper action term with "
            "compute_rule_actions(); use --raw_teacher_gripper only for historical reproduction"
        )
    if args_cli.raw_teacher_gripper:
        print("[vis-collect] gripper labels: raw ignored RL teacher output (historical mode)")
    elif args_cli.osc_joint_trace:
        gripper_label_note = {
            "fixed_open": "fixed-open width target (0.08m); no grasp guard",
            "policy_sign": "sign of RL teacher output, executed as +/-1; no grasp guard",
            "grasp_guard": "grasp-guard width target (0 close, 0.08m open)",
        }[args_cli.osc_joint_trace_gripper_mode]
        print(f"[vis-collect] gripper labels: {gripper_label_note}")
    elif args_cli.joint_teacher:
        print("[vis-collect] gripper labels: policy-controlled width target (0 or 0.08m); no grasp guard")
    elif args_cli.native_osc_policy_gripper:
        print(
            "[vis-collect] gripper labels: sign of RL teacher output, executed as +/-1; "
            "raw teacher output is preserved separately; no grasp guard"
        )
    elif args_cli.fixed_open_gripper:
        print("[vis-collect] gripper labels: fixed-open width target (0.08m); no grasp guard")
    elif args_cli.joint_target_bridge:
        print("[vis-collect] gripper labels: actual guarded width target (0 close, 0.08m open); stored std=0")
    else:
        print("[vis-collect] gripper labels: grasp-guard command (-1 close, +1 open); stored std=0")

    print(f"[vis-collect] teacher: {args_cli.teacher_ckpt}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # The RGB env has no `critic` obs group, so the runner's critic defaults to the 200-dim
    # policy obs, but the teacher's critic was trained on the 171-dim State CriticCfg → strict
    # load mismatches. Inference uses only the actor, so load actor-only (skip critic.*).
    _loaded = torch.load(args_cli.teacher_ckpt, map_location=device, weights_only=False)
    _actor_only = {k: v for k, v in _loaded["model_state_dict"].items() if not k.startswith("critic")}
    # rsl_rl ActorCritic.load_state_dict returns a bool (not the std NamedTuple); strict=False
    # loads the actor + std + actor_obs_normalizer and ignores the missing critic.* keys.
    runner.alg.policy.load_state_dict(_actor_only, strict=False)
    print(f"[vis-collect] loaded actor-only ({len(_actor_only)} keys, critic skipped)")
    runner.alg.policy.eval()
    runner.get_inference_policy(device=device)
    actor = runner.alg.policy
    print(f"[vis-collect] teacher noise_std_type = {getattr(actor, 'noise_std_type', '?')}")
    print(
        "[vis-collect] teacher policy action mode = "
        f"{'stochastic_sample' if args_cli.stochastic_teacher_actions else 'deterministic_mean'}"
    )

    num_envs = env.unwrapped.num_envs
    cam_keys = [] if args_cli.no_store_images else ["front_rgb", "side_rgb", "wrist_rgb"]
    target_demo_count = (
        len(forced_unique_state_ids) * args_cli.successful_repeats_per_state
        if args_cli.successful_repeats_per_state
        else args_cli.num_demos
    )
    print(
        f"[vis-collect] num_envs={num_envs} target_demos={target_demo_count} "
        f"image_size={args_cli.image_size} cams={cam_keys}"
    )
    if args_cli.no_store_images:
        print("[vis-collect] trajectory-selection mode: RGB arrays will not be stored")

    def progress_term():
        return env.unwrapped.reward_manager.get_term_cfg("progress_context").func

    def dc_group():
        return env.unwrapped.obs_buf["data_collection"]

    # Clean, deployable robot-only state. Joint-target datasets use measured
    # arm joints directly; legacy OSC datasets retain the historical EE pose.
    robot = env.unwrapped.scene["robot"]
    robot_contract = resolve_franka_robot_contract(robot)
    _hand_idx = robot.body_names.index(robot_contract.hand_body_name)
    _arm_ids, _arm_names = robot.find_joints(
        list(robot_contract.arm_joint_names), preserve_order=True
    )
    _finger_ids, _ = robot.find_joints(list(robot_contract.finger_joint_names))
    if joint_label_mode:
        print(f"[vis-collect] proprio: measured joints {_arm_names} + gripper width -> 8-dim")
    else:
        print(
            f"[vis-collect] proprio: ee({robot_contract.hand_body_name},body#{_hand_idx}) "
            f"+ gripper(joints {_finger_ids}) -> 7-dim"
        )

    def compute_proprio():
        grip = robot.data.joint_pos[:, _finger_ids].sum(dim=-1, keepdim=True)  # total opening ~0..0.08
        if joint_label_mode:
            joint_pos = robot.data.joint_pos[:, _arm_ids]
            return torch.cat([joint_pos, grip], dim=-1)
        ee_pos_w = robot.data.body_link_pos_w[:, _hand_idx].view(-1, 3)
        ee_quat_w = robot.data.body_link_quat_w[:, _hand_idx].view(-1, 4)
        root_pos_w = robot.data.body_link_pos_w[:, 0].view(-1, 3)
        root_quat_w = robot.data.body_link_quat_w[:, 0].view(-1, 4)
        pos_b, quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        aa = math_utils.axis_angle_from_quat(quat_b)
        return torch.cat([pos_b, aa, grip], dim=-1)  # (N,7): ee_pos(3)+ee_axisangle(3)+gripper_width(1)

    def relative_root_state(asset):
        root_state = asset.data.root_state_w.clone()
        root_state[:, :3] -= env.unwrapped.scene.env_origins
        return root_state

    numeric_keys = (
        ["state", "reset_state_id"]
        if args_cli.outcomes_only
        else ["state", "proprio", "action", "action_std"]
    )
    if args_cli.save_all_episodes:
        numeric_keys.append("episode_success")
    record_placement_metrics = args_cli.record_placement_metrics or args_cli.save_fixed_gate_episodes
    if record_placement_metrics:
        numeric_keys += [
            "placement_xyz_m",
            "placement_orientation_rad",
            "placement_fixed_gate",
        ]
    if args_cli.save_fixed_gate_episodes:
        numeric_keys.append("episode_fixed_gate_success")
    if args_cli.joint_target_bridge and not args_cli.outcomes_only:
        numeric_keys += [
            "teacher_action",
            "teacher_action_mean_raw",
            "teacher_action_sample_raw",
            "teacher_action_std",
            "joint_position_measured",
            "ik_joint_position_target",
            "joint_target_delta",
            "ik_valid",
            "robot_root_state_rel",
            "insertive_root_state_rel",
            "receptive_root_state_rel",
            "robot_joint_position_full",
            "robot_joint_velocity_full",
            "joint_pos_target_sim",
            "joint_vel_target_sim",
            "joint_effort_target_sim",
        ]
    elif args_cli.native_osc_guard_labels:
        numeric_keys += [
            "teacher_action_raw",
            "teacher_action_std_raw",
            "gripper_binary_command",
            "gripper_width_target",
        ]
    elif args_cli.osc_joint_trace:
        numeric_keys += [
            "teacher_action",
            "teacher_action_std",
            "gripper_binary_command",
            "gripper_width_target",
            "joint_position_measured",
            "joint_position_measured_next",
            "osc_joint_position_first_substep",
            "osc_joint_velocity_first_substep",
            "osc_joint_torque_first_substep",
            "joint_target_delta",
        ]
    elif args_cli.joint_teacher:
        numeric_keys += [
            "teacher_action",
            "teacher_action_std",
            "joint_position_measured",
            "joint_target_delta",
        ]
    elif args_cli.native_osc_policy_gripper:
        numeric_keys += [
            "teacher_action_raw",
            "teacher_action_std_raw",
            "ee_linear_speed_max",
            "ee_angular_speed_max",
            "osc_reference_linear_speed_max",
            "osc_reference_angular_speed_max",
            "gripper_binary_command",
            "gripper_width_target",
        ]
    if indexed_reset_mode and "reset_state_id" not in numeric_keys:
        numeric_keys.append("reset_state_id")
    buf = {k: [[] for _ in range(num_envs)] for k in cam_keys + numeric_keys}
    demos = []  # list of dict of stacked arrays
    stream_writer = (
        StreamingZarrWriter(
            args_cli.output,
            cam_keys,
            numeric_keys,
            collection_config,
        )
        if args_cli.successful_repeats_per_state
        else None
    )
    saved_demo_count = 0
    n_frames = 0  # total stored frames (for --target_frames mode)
    by_frames = args_cli.target_frames > 0
    home = torch.tensor(joint_home_arm, device=device, dtype=torch.float32)
    home_checks = 0
    ik_invalid_total = 0
    max_target_write_error = 0.0
    bridge_command_frames = 0
    bridge_raw_delta_l2_sum = 0.0
    bridge_raw_delta_l2_max = 0.0
    bridge_applied_delta_l2_sum = 0.0
    bridge_applied_delta_l2_max = 0.0
    bridge_rate_cap_elements = 0
    bridge_rate_cap_frames = 0
    bridge_tracking_abs_sum = 0.0
    bridge_tracking_abs_max = 0.0
    bridge_tracking_elements = 0
    bridge_gripper_open_frames = 0
    bridge_gripper_close_frames = 0
    completed_episodes = 0
    successful_episodes = 0
    termination_counts: dict[str, int] = {}
    diagnostic_xyz_m: list[float] = []
    diagnostic_orientation_rad: list[float] = []
    outcome_records: list[dict] = []
    completed_forced_state_ids: set[int] = set()
    successful_repeats_by_state = {
        state_id: 0 for state_id in forced_unique_state_ids
    }
    successful_repeat_reset_manager = None
    if args_cli.successful_repeats_per_state:
        successful_repeat_reset_manager = reset_term
        if not hasattr(successful_repeat_reset_manager, "set_forced_state_indices"):
            raise RuntimeError(
                "reset_from_reset_states does not support dynamic forced-state scheduling"
            )
    warmup_remaining = torch.full(
        (num_envs,),
        int(args_cli.post_reset_warmup_steps),
        device=device,
        dtype=torch.long,
    )

    def keep_going():
        if args_cli.successful_repeats_per_state:
            return any(
                count < args_cli.successful_repeats_per_state
                for count in successful_repeats_by_state.values()
            )
        if args_cli.complete_forced_state_set:
            assert forced_reset_state_indices is not None
            return len(completed_forced_state_ids) < len(set(forced_reset_state_indices))
        if args_cli.target_attempts > 0:
            return completed_episodes < args_cli.target_attempts
        return (
            n_frames < args_cli.target_frames
            if by_frames
            else saved_demo_count < args_cli.num_demos
        )

    obs = env.get_observations()
    step = 0
    mean = None
    teacher_action_mean_raw = None
    teacher_action_sample_raw = None
    std = None
    while keep_going() and step < args_cli.max_steps:
        native_teacher_action_raw = None
        native_teacher_action_std_raw = None
        decision_step = step % args_cli.action_hold_steps == 0
        if args_cli.joint_target_bridge:
            arm_term.set_hold_joint_position_target(not decision_step)
            if args_cli.action_hold_steps > 1:
                gripper_term.set_hold_processed_action(not decision_step)
        if decision_step:
            with torch.inference_mode():
                teacher_action_sample_raw = actor.act(obs)
                teacher_action_mean_raw = actor.action_mean
                std = actor.action_std
                mean = (
                    teacher_action_sample_raw
                    if args_cli.stochastic_teacher_actions
                    else teacher_action_mean_raw
                )
        assert (
            mean is not None
            and teacher_action_mean_raw is not None
            and teacher_action_sample_raw is not None
            and std is not None
        )
        if decision_step:
            if args_cli.joint_teacher:
                if mean.shape[-1] != 8:
                    raise ValueError(f"expected 8-D joint teacher action, got shape {tuple(mean.shape)}")
            elif args_cli.osc_joint_trace:
                if mean.shape[-1] != 7:
                    raise ValueError(f"expected 7-D teacher action, got shape {tuple(mean.shape)}")
                mean = mean.clone()
                if args_cli.osc_joint_trace_gripper_mode == "fixed_open":
                    mean[:, -1:] = 1.0
                elif args_cli.osc_joint_trace_gripper_mode == "policy_sign":
                    mean[:, -1:] = torch.where(mean[:, -1:] < 0.0, -1.0, 1.0)
                elif args_cli.osc_joint_trace_gripper_mode == "grasp_guard":
                    mean[:, -1:] = gripper_term.compute_rule_actions()
                else:
                    raise ValueError(
                        f"unknown osc_joint_trace_gripper_mode={args_cli.osc_joint_trace_gripper_mode!r}"
                    )
                std = std.clone()
                std[:, -1:] = 0.0
            elif args_cli.fixed_open_gripper:
                if mean.shape[-1] != 7:
                    raise ValueError(f"expected 7-D teacher action, got shape {tuple(mean.shape)}")
                mean = mean.clone()
                mean[:, -1:] = 1.0
                std = std.clone()
                std[:, -1:] = 0.0
            elif args_cli.native_osc_guard_labels:
                if mean.shape[-1] != 7:
                    raise ValueError(f"expected 7-D teacher action, got shape {tuple(mean.shape)}")
                native_teacher_action_raw = mean.clone()
                native_teacher_action_std_raw = std.clone()
                mean = mean.clone()
                mean[:, -1:] = gripper_term.compute_rule_actions()
                std = std.clone()
                std[:, -1:] = 0.0
            elif args_cli.native_osc_policy_gripper:
                if mean.shape[-1] != 7:
                    raise ValueError(f"expected 7-D teacher action, got shape {tuple(mean.shape)}")
                native_teacher_action_raw = mean.clone()
                native_teacher_action_std_raw = std.clone()
                mean = mean.clone()
                mean[:, -1:] = torch.where(mean[:, -1:] < 0.0, -1.0, 1.0)
                std = std.clone()
                std[:, -1:] = 0.0
            elif not args_cli.raw_teacher_gripper and not args_cli.native_osc_policy_gripper:
                if mean.shape[-1] != 7:
                    raise ValueError(f"expected 7-D teacher action, got shape {tuple(mean.shape)}")
                mean = mean.clone()
                mean[:, -1:] = gripper_term.compute_rule_actions()
                std = std.clone()
                std[:, -1:] = 0.0
        record_mask = warmup_remaining == 0
        if not bool(record_mask.all().item()):
            mean = mean.clone()
            mean[~record_mask, :-1] = 0.0
            std = std.clone()
            std[~record_mask, :-1] = 0.0
        state = _flat_policy(obs)            # 200-dim privileged state (aux target)
        proprio = compute_proprio()
        dc = dc_group() if cam_keys else None

        snap = {
            "state": (
                np.zeros((num_envs, 1), dtype=np.float32)
                if args_cli.outcomes_only
                else state.detach().cpu().numpy()
            ),
        }
        if not args_cli.outcomes_only:
            snap["proprio"] = proprio.detach().cpu().numpy()
        if args_cli.joint_target_bridge and not args_cli.outcomes_only:
            insertive_object = env.unwrapped.scene["insertive_object"]
            receptive_object = env.unwrapped.scene["receptive_object"]
            snap.update({
                "robot_root_state_rel": relative_root_state(robot).detach().cpu().numpy(),
                "insertive_root_state_rel": relative_root_state(insertive_object).detach().cpu().numpy(),
                "receptive_root_state_rel": relative_root_state(receptive_object).detach().cpu().numpy(),
                "robot_joint_position_full": robot.data.joint_pos.detach().cpu().numpy(),
                "robot_joint_velocity_full": robot.data.joint_vel.detach().cpu().numpy(),
                "joint_pos_target_sim": robot._joint_pos_target_sim.detach().cpu().numpy(),
                "joint_vel_target_sim": robot._joint_vel_target_sim.detach().cpu().numpy(),
                "joint_effort_target_sim": robot._joint_effort_target_sim.detach().cpu().numpy(),
            })
        if args_cli.save_all_episodes:
            snap["episode_success"] = np.zeros((num_envs, 1), dtype=np.float32)
        if record_placement_metrics:
            context = progress_term()
            placement_xyz_m = context.xyz_distance.unsqueeze(-1)
            placement_orientation_rad = context.euler_xy_distance.unsqueeze(-1)
            placement_fixed_gate = (
                (placement_xyz_m < args_cli.diagnostic_position_m)
                & (placement_orientation_rad < float(np.deg2rad(args_cli.diagnostic_orientation_deg)))
            )
            first_frame_mask = torch.tensor(
                [len(buf["state"][env_i]) == 0 for env_i in range(num_envs)],
                device=device,
                dtype=torch.bool,
            ).unsqueeze(-1)
            placement_fixed_gate = placement_fixed_gate & ~first_frame_mask
            snap.update({
                "placement_xyz_m": placement_xyz_m.detach().cpu().numpy(),
                "placement_orientation_rad": placement_orientation_rad.detach().cpu().numpy(),
                "placement_fixed_gate": placement_fixed_gate.to(dtype=torch.float32).detach().cpu().numpy(),
            })
        if args_cli.save_fixed_gate_episodes:
            snap["episode_fixed_gate_success"] = np.zeros((num_envs, 1), dtype=np.float32)
        if indexed_reset_mode:
            # env.step() auto-resets terminated environments, so capture the
            # reset-state identity before stepping just like RGB/proprio.
            snap["reset_state_id"] = (
                reset_term.state_id.to(dtype=torch.float32).unsqueeze(-1).cpu().numpy()
            )
        for k in cam_keys:
            assert dc is not None
            frames = dc[k].detach().cpu().numpy()  # (N,224,224,3) uint8
            if k in args_cli.rotate_180_camera:
                frames = np.ascontiguousarray(frames[:, ::-1, ::-1, :])
            snap[k] = frames

        obs, _, dones, _ = env.step(mean)
        if args_cli.joint_teacher:
            q_applied = arm_term.last_applied_actions
            gripper_width_target = gripper_term.processed_actions.sum(dim=-1, keepdim=True)
            action = torch.cat([q_applied, gripper_width_target], dim=-1)
            snap.update({
                "action": action.detach().cpu().numpy(),
                "action_std": torch.zeros_like(action).cpu().numpy(),
                "teacher_action": mean.detach().cpu().numpy(),
                "teacher_action_std": std.detach().cpu().numpy(),
                "joint_position_measured": proprio[:, :7].detach().cpu().numpy(),
                "joint_target_delta": (q_applied - proprio[:, :7]).detach().cpu().numpy(),
            })
        elif args_cli.joint_target_bridge:
            q_target = arm_term.last_joint_position_targets
            q_applied = arm_term.last_applied_joint_position_targets
            target_write_error = torch.max(torch.abs(q_target - q_applied)).item()
            max_target_write_error = max(max_target_write_error, target_write_error)
            if target_write_error > 1e-7:
                raise RuntimeError(
                    "joint target recorded by the collector differs from the target passed to "
                    f"set_joint_position_target: max error={target_write_error:.3e}"
                )
            ik_valid = arm_term.last_ik_valid
            ik_invalid_total += int(((~ik_valid) & record_mask).sum().item())
            gripper_binary_command = gripper_term.raw_actions.reshape(mean.shape[0], -1)
            gripper_width_target = gripper_term.processed_actions.sum(dim=-1, keepdim=True)
            expected_width_target = torch.where(
                mean[:, -1:] < 0.0,
                torch.zeros_like(mean[:, -1:]),
                torch.full_like(mean[:, -1:], 0.08),
            )
            active = ~dones.bool()
            if bool(active.any().item()):
                guard_command_error = torch.max(
                    torch.abs(gripper_binary_command[active] - mean[active, -1:])
                ).item()
                width_target_error = torch.max(
                    torch.abs(gripper_width_target[active] - expected_width_target[active])
                ).item()
                if guard_command_error > 1.0e-7 or width_target_error > 1.0e-7:
                    raise RuntimeError(
                        "joint-target bridge grasp-guard label differs from the executed "
                        "gripper command: "
                        f"binary error={guard_command_error:.3e}, "
                        f"width error={width_target_error:.3e}"
                    )
            raw_delta = arm_term.last_ik_joint_position_targets - proprio[:, :7]
            applied_delta = q_applied - proprio[:, :7]
            raw_delta_l2 = torch.linalg.vector_norm(raw_delta, dim=-1)
            applied_delta_l2 = torch.linalg.vector_norm(applied_delta, dim=-1)
            rate_cap = args_cli.joint_max_velocity * float(env_cfg.decimation * env_cfg.sim.dt)
            at_rate_cap = torch.isclose(
                torch.abs(applied_delta),
                torch.full_like(applied_delta, rate_cap),
                atol=1.0e-5,
                rtol=0.0,
            )
            if bool(record_mask.any().item()):
                bridge_command_frames += int(record_mask.sum().item())
                bridge_raw_delta_l2_sum += float(raw_delta_l2[record_mask].sum().item())
                bridge_raw_delta_l2_max = max(
                    bridge_raw_delta_l2_max, float(raw_delta_l2[record_mask].max().item())
                )
                bridge_applied_delta_l2_sum += float(applied_delta_l2[record_mask].sum().item())
                bridge_applied_delta_l2_max = max(
                    bridge_applied_delta_l2_max, float(applied_delta_l2[record_mask].max().item())
                )
                bridge_rate_cap_elements += int(at_rate_cap[record_mask].sum().item())
                bridge_rate_cap_frames += int(at_rate_cap[record_mask].any(dim=-1).sum().item())
                bridge_gripper_open_frames += int(
                    torch.isclose(
                        gripper_width_target[record_mask],
                        torch.full_like(gripper_width_target[record_mask], 0.08),
                    ).sum().item()
                )
                bridge_gripper_close_frames += int(
                    torch.isclose(
                        gripper_width_target[record_mask],
                        torch.zeros_like(gripper_width_target[record_mask]),
                    ).sum().item()
                )
            tracking_mask = (~dones.bool()) & record_mask
            if bool(tracking_mask.any().item()):
                q_measured_after = robot.data.joint_pos[:, _arm_ids]
                tracking_abs = torch.abs(q_applied[tracking_mask] - q_measured_after[tracking_mask])
                bridge_tracking_abs_sum += float(tracking_abs.sum().item())
                bridge_tracking_abs_max = max(
                    bridge_tracking_abs_max, float(tracking_abs.max().item())
                )
                bridge_tracking_elements += int(tracking_abs.numel())
            action = torch.cat([q_applied, gripper_width_target], dim=-1)
            if not args_cli.outcomes_only:
                snap.update({
                    "action": action.detach().cpu().numpy(),
                    "action_std": torch.zeros_like(action).cpu().numpy(),
                    "teacher_action": mean.detach().cpu().numpy(),
                    "teacher_action_mean_raw": teacher_action_mean_raw.detach().cpu().numpy(),
                    "teacher_action_sample_raw": teacher_action_sample_raw.detach().cpu().numpy(),
                    "teacher_action_std": std.detach().cpu().numpy(),
                    "gripper_binary_command": gripper_binary_command.detach().cpu().numpy(),
                    "gripper_width_target": gripper_width_target.detach().cpu().numpy(),
                    "joint_position_measured": proprio[:, :7].detach().cpu().numpy(),
                    "ik_joint_position_target": arm_term.last_ik_joint_position_targets.detach().cpu().numpy(),
                    "joint_target_delta": (q_applied - proprio[:, :7]).detach().cpu().numpy(),
                    "ik_valid": ik_valid.to(dtype=torch.float32).unsqueeze(-1).cpu().numpy(),
                })
        elif args_cli.osc_joint_trace:
            joint_substeps = arm_term.joint_pos_substeps
            if joint_substeps.shape[0] == 0:
                raise RuntimeError("OSC action term did not record joint-position substeps")
            velocity_substeps = arm_term.joint_vel_substeps
            torque_substeps = arm_term.joint_torque_substeps
            if velocity_substeps.shape[0] == 0:
                raise RuntimeError("OSC action term did not record joint-velocity substeps")
            if torque_substeps.shape[0] == 0:
                raise RuntimeError("OSC action term did not record joint-torque substeps")
            q_first = joint_substeps[0]
            dq_first = velocity_substeps[0]
            tau_first = torque_substeps[0]
            q_achieved = joint_substeps[-1]
            if args_cli.osc_joint_trace_target == "torque_match":
                q_target = q_first + (
                    tau_first + args_cli.joint_position_damping * dq_first
                ) / args_cli.joint_position_stiffness
                rate_cap = args_cli.joint_max_velocity * float(env_cfg.decimation * env_cfg.sim.dt)
                q_target = q_first + torch.clamp(q_target - q_first, min=-rate_cap, max=rate_cap)
            else:
                q_target = q_achieved
            gripper_binary_command = gripper_term.raw_actions.reshape(mean.shape[0], -1)
            gripper_width_target = gripper_term.processed_actions.sum(dim=-1, keepdim=True)
            expected_width_target = torch.where(
                mean[:, -1:] < 0.0,
                torch.zeros_like(mean[:, -1:]),
                torch.full_like(mean[:, -1:], 0.08),
            )
            active = ~dones.bool()
            if bool(active.any().item()):
                gripper_command_error = torch.max(
                    torch.abs(gripper_binary_command[active] - mean[active, -1:])
                ).item()
                width_target_error = torch.max(
                    torch.abs(gripper_width_target[active] - expected_width_target[active])
                ).item()
                if gripper_command_error > 1.0e-7 or width_target_error > 1.0e-7:
                    raise RuntimeError(
                        "OSC joint-trace gripper command was not executed as recorded: "
                        f"raw error={gripper_command_error:.3e}, "
                        f"width error={width_target_error:.3e}"
                    )
            action = torch.cat([q_target, gripper_width_target], dim=-1)
            snap.update({
                "action": action.detach().cpu().numpy(),
                "action_std": torch.zeros_like(action).cpu().numpy(),
                "teacher_action": mean.detach().cpu().numpy(),
                "teacher_action_std": std.detach().cpu().numpy(),
                "gripper_binary_command": gripper_binary_command.detach().cpu().numpy(),
                "gripper_width_target": gripper_width_target.detach().cpu().numpy(),
                "joint_position_measured": proprio[:, :7].detach().cpu().numpy(),
                "joint_position_measured_next": q_achieved.detach().cpu().numpy(),
                "osc_joint_position_first_substep": q_first.detach().cpu().numpy(),
                "osc_joint_velocity_first_substep": dq_first.detach().cpu().numpy(),
                "osc_joint_torque_first_substep": tau_first.detach().cpu().numpy(),
                "joint_target_delta": (q_target - proprio[:, :7]).detach().cpu().numpy(),
            })
        elif args_cli.native_osc_guard_labels:
            assert native_teacher_action_raw is not None
            assert native_teacher_action_std_raw is not None
            guard_binary_command = gripper_term.raw_actions.reshape(mean.shape[0], -1)
            gripper_width_target = gripper_term.processed_actions.sum(dim=-1, keepdim=True)
            expected_width_target = torch.where(
                mean[:, -1:] < 0.0,
                torch.zeros_like(mean[:, -1:]),
                torch.full_like(mean[:, -1:], 0.08),
            )
            active = ~dones.bool()
            if bool(active.any().item()):
                guard_command_error = torch.max(
                    torch.abs(guard_binary_command[active] - mean[active, -1:])
                ).item()
                width_target_error = torch.max(
                    torch.abs(gripper_width_target[active] - expected_width_target[active])
                ).item()
                if guard_command_error > 1.0e-7 or width_target_error > 1.0e-7:
                    raise RuntimeError(
                        "stored grasp-guard label differs from the executed gripper command: "
                        f"binary error={guard_command_error:.3e}, "
                        f"width error={width_target_error:.3e}"
                    )
            snap.update({
                "action": mean.detach().cpu().numpy(),
                "action_std": std.detach().cpu().numpy(),
                "teacher_action_raw": native_teacher_action_raw.detach().cpu().numpy(),
                "teacher_action_std_raw": native_teacher_action_std_raw.detach().cpu().numpy(),
                "gripper_binary_command": mean[:, -1:].detach().cpu().numpy(),
                "gripper_width_target": expected_width_target.detach().cpu().numpy(),
            })
        elif args_cli.native_osc_policy_gripper:
            assert native_teacher_action_raw is not None
            assert native_teacher_action_std_raw is not None
            ee_velocity_substeps = arm_term.ee_velocity_substeps
            reference_velocity_substeps = arm_term.ee_reference_velocity_substeps
            if ee_velocity_substeps.shape[0] == 0 or reference_velocity_substeps.shape[0] == 0:
                raise RuntimeError("native OSC action term did not record velocity substeps")
            gripper_binary_command = gripper_term.raw_actions.reshape(mean.shape[0], -1)
            expected_width_target = torch.where(
                mean[:, -1:] < 0.0,
                torch.zeros_like(mean[:, -1:]),
                torch.full_like(mean[:, -1:], 0.08),
            )
            active = ~dones.bool()
            if bool(active.any().item()):
                gripper_command_error = torch.max(
                    torch.abs(gripper_binary_command[active] - mean[active, -1:])
                ).item()
                gripper_width_target = gripper_term.processed_actions.sum(dim=-1, keepdim=True)
                width_target_error = torch.max(
                    torch.abs(gripper_width_target[active] - expected_width_target[active])
                ).item()
                if gripper_command_error > 1.0e-7 or width_target_error > 1.0e-7:
                    raise RuntimeError(
                        "standard binary gripper did not execute the teacher-sign command: "
                        f"raw error={gripper_command_error:.3e}, width error={width_target_error:.3e}"
                    )
            snap.update({
                "action": mean.detach().cpu().numpy(),
                "action_std": std.detach().cpu().numpy(),
                "teacher_action_raw": native_teacher_action_raw.detach().cpu().numpy(),
                "teacher_action_std_raw": native_teacher_action_std_raw.detach().cpu().numpy(),
                "ee_linear_speed_max": torch.linalg.vector_norm(
                    ee_velocity_substeps[:, :, :3], dim=-1
                ).amax(dim=0, keepdim=False).unsqueeze(-1).detach().cpu().numpy(),
                "ee_angular_speed_max": torch.linalg.vector_norm(
                    ee_velocity_substeps[:, :, 3:], dim=-1
                ).amax(dim=0, keepdim=False).unsqueeze(-1).detach().cpu().numpy(),
                "osc_reference_linear_speed_max": torch.linalg.vector_norm(
                    reference_velocity_substeps[:, :, :3], dim=-1
                ).amax(dim=0, keepdim=False).unsqueeze(-1).detach().cpu().numpy(),
                "osc_reference_angular_speed_max": torch.linalg.vector_norm(
                    reference_velocity_substeps[:, :, 3:], dim=-1
                ).amax(dim=0, keepdim=False).unsqueeze(-1).detach().cpu().numpy(),
                "gripper_binary_command": mean[:, -1:].detach().cpu().numpy(),
                "gripper_width_target": expected_width_target.detach().cpu().numpy(),
            })
        else:
            snap["action"] = mean.detach().cpu().numpy()
            snap["action_std"] = std.detach().cpu().numpy()

        for i in range(num_envs):
            if not bool(record_mask[i].item()):
                continue
            if (
                joint_label_mode
                and not args_cli.skip_joint_home_check
                and len(buf["state"][i]) == 0
            ):
                home_error = torch.max(torch.abs(proprio[i, :7] - home)).item()
                if home_error > args_cli.home_tolerance_rad:
                    raise RuntimeError(
                        f"env {i} first recorded frame is not at team home: "
                        f"max joint error={home_error:.6f}rad > {args_cli.home_tolerance_rad:.6f}rad"
                    )
                home_checks += 1
            for k in cam_keys + numeric_keys:
                buf[k][i].append(snap[k][i])

        success_terminated = env.unwrapped.termination_manager.get_term("success")
        for i in range(num_envs):
            if bool(dones[i].item()):
                if not bool(record_mask[i].item()):
                    # A forced selector state can terminate during the post-reset
                    # warmup step. It has still been attempted, but there is no
                    # action-labelled episode that can be saved as a success.
                    # Without accounting for it here, complete_forced_state_set
                    # cycles forever waiting for an impossible coverage entry.
                    if args_cli.complete_forced_state_set:
                        episode_state_id = int(snap["reset_state_id"][i][0])
                        if episode_state_id < 0:
                            raise RuntimeError(
                                "forced selector terminated during warmup without a valid state ID"
                            )
                        if episode_state_id not in completed_forced_state_ids:
                            completed_forced_state_ids.add(episode_state_id)
                            completed_episodes += 1
                            active_done_terms = []
                            for term_name in env.unwrapped.termination_manager.active_terms:
                                if bool(
                                    env.unwrapped.termination_manager.get_term(term_name)[i].item()
                                ):
                                    active_done_terms.append(term_name)
                                    termination_counts[term_name] = (
                                        termination_counts.get(term_name, 0) + 1
                                    )
                            print(
                                "[vis-collect] attempt result: "
                                f"state={episode_state_id}, steps=0, effective_success=False, "
                                "warmup_termination=True, "
                                f"done_terms={active_done_terms}"
                            )
                            if args_cli.outcomes_only:
                                outcome_records.append(
                                    {
                                        "state_id": episode_state_id,
                                        "steps": 0,
                                        "effective_success": False,
                                        "warmup_termination": True,
                                        "xyz_m": None,
                                        "orientation_rad": None,
                                        "done_terms": active_done_terms,
                                    }
                                )
                    for k in buf:
                        buf[k][i] = []
                    continue
                if args_cli.target_attempts > 0 and completed_episodes >= args_cli.target_attempts:
                    for k in buf:
                        buf[k][i] = []
                    continue
                episode_state_id = (
                    int(buf["reset_state_id"][i][0][0])
                    if indexed_reset_mode and len(buf["reset_state_id"][i]) > 0
                    else -1
                )
                if args_cli.complete_forced_state_set:
                    if episode_state_id in completed_forced_state_ids:
                        for k in buf:
                            buf[k][i] = []
                        continue
                    completed_forced_state_ids.add(episode_state_id)
                completed_episodes += 1
                succeeded = bool(success_terminated[i].item())
                successful_episodes += int(succeeded)
                endpoint_xyz_m = None
                endpoint_orientation_rad = None
                if args_cli.diagnostic_done_metrics:
                    context = progress_term()
                    endpoint_xyz_m = float(context.xyz_distance[i].item())
                    endpoint_orientation_rad = float(context.euler_xy_distance[i].item())
                    diagnostic_xyz_m.append(endpoint_xyz_m)
                    diagnostic_orientation_rad.append(endpoint_orientation_rad)
                active_done_terms = []
                for term_name in env.unwrapped.termination_manager.active_terms:
                    if bool(env.unwrapped.termination_manager.get_term(term_name)[i].item()):
                        active_done_terms.append(term_name)
                        termination_counts[term_name] = termination_counts.get(term_name, 0) + 1
                if endpoint_xyz_m is not None and endpoint_orientation_rad is not None:
                    print(
                        "[vis-collect] attempt result: "
                        f"state={episode_state_id}, steps={len(buf['state'][i])}, "
                        f"effective_success={succeeded}, xyz_mm={endpoint_xyz_m * 1000:.2f}, "
                        f"orientation_deg={np.rad2deg(endpoint_orientation_rad):.2f}, "
                        f"done_terms={active_done_terms}"
                    )
                if args_cli.outcomes_only:
                    outcome_records.append(
                        {
                            "state_id": episode_state_id,
                            "steps": len(buf["state"][i]),
                            "effective_success": succeeded,
                            "warmup_termination": False,
                            "xyz_m": endpoint_xyz_m,
                            "orientation_rad": endpoint_orientation_rad,
                            "done_terms": active_done_terms,
                        }
                    )
                fixed_gate_hit = False
                fixed_gate_index = None
                if record_placement_metrics and len(buf["placement_fixed_gate"][i]) > 0:
                    fixed_gate_flags = (
                        np.asarray(buf["placement_fixed_gate"][i], dtype=np.float32)
                        .reshape(-1)
                        > 0.5
                    )
                    fixed_gate_hit = bool(fixed_gate_flags.any())
                    if fixed_gate_hit:
                        fixed_gate_index = int(np.argmax(fixed_gate_flags))
                save_episode = (not args_cli.outcomes_only) and (
                    succeeded
                    or args_cli.save_all_episodes
                    or (args_cli.save_fixed_gate_episodes and fixed_gate_hit)
                ) and len(buf["state"][i]) > 0
                if save_episode and args_cli.successful_repeats_per_state:
                    if episode_state_id not in successful_repeats_by_state:
                        raise RuntimeError(
                            f"completed unexpected forced reset state {episode_state_id}"
                        )
                    if (
                        successful_repeats_by_state[episode_state_id]
                        >= args_cli.successful_repeats_per_state
                    ):
                        save_episode = False
                    else:
                        successful_repeats_by_state[episode_state_id] += 1
                        if (
                            successful_repeats_by_state[episode_state_id]
                            == args_cli.successful_repeats_per_state
                        ):
                            remaining_state_ids = [
                                state_id
                                for state_id in forced_unique_state_ids
                                if successful_repeats_by_state[state_id]
                                < args_cli.successful_repeats_per_state
                            ]
                            if remaining_state_ids:
                                successful_repeat_reset_manager.set_forced_state_indices(
                                    remaining_state_ids
                                )
                                print(
                                    "[vis-collect] successful-repeat scheduler: "
                                    f"state={episode_state_id} reached quota; "
                                    f"remaining_states={len(remaining_state_ids)}"
                                )
                if save_episode:
                    episode = {k: np.stack(buf[k][i], 0) for k in buf}
                    if (
                        args_cli.truncate_to_fixed_gate
                        and fixed_gate_hit
                        and fixed_gate_index is not None
                    ):
                        # placement_fixed_gate is measured before action_t. To replay into
                        # that gate frame, keep actions through t-1; keep one frame for
                        # already-satisfied resets so the zarr episode is non-empty.
                        trunc_len = max(fixed_gate_index, 1)
                        episode = {k: v[:trunc_len] for k, v in episode.items()}
                    if args_cli.save_all_episodes:
                        episode["episode_success"].fill(float(succeeded))
                    if args_cli.save_fixed_gate_episodes:
                        episode["episode_fixed_gate_success"].fill(float(fixed_gate_hit))
                    if stream_writer is not None:
                        stream_writer.append(episode)
                    else:
                        demos.append(episode)
                    saved_demo_count += 1
                    n_frames += len(episode["state"])
                    if by_frames:
                        if n_frames // 2000 != (n_frames - len(buf["state"][i])) // 2000:
                            print(
                                f"[vis-collect] {n_frames}/{args_cli.target_frames} frames, "
                                f"{saved_demo_count} demos (step {step})"
                            )
                    elif saved_demo_count % 10 == 0 or saved_demo_count == 1:
                        print(
                            f"[vis-collect] {saved_demo_count}/{target_demo_count} demos "
                            f"(step {step})"
                        )
                for k in buf:
                    buf[k][i] = []
        warmup_remaining = torch.where(
            dones.bool(),
            torch.full_like(warmup_remaining, int(args_cli.post_reset_warmup_steps)),
            torch.clamp(warmup_remaining - 1, min=0),
        )
        if indexed_reset_mode and completed_episodes > 0 and step % 100 == 0:
            print(
                f"[vis-collect] indexed attempts: {successful_episodes}/{completed_episodes} "
                f"successful ({successful_episodes / completed_episodes:.3f}), "
                f"saved={saved_demo_count}/{target_demo_count}"
            )
        step += 1

    print(
        f"[vis-collect] done: {saved_demo_count} demos over {step} steps; "
        f"attempt_success={successful_episodes}/{completed_episodes}; "
        f"termination_counts={termination_counts}"
    )
    if args_cli.complete_forced_state_set:
        print(
            f"[vis-collect] forced-state coverage: {len(completed_forced_state_ids)}/"
            f"{len(set(forced_reset_state_indices or []))} unique states attempted"
        )
    if args_cli.successful_repeats_per_state:
        reached = sum(
            count >= args_cli.successful_repeats_per_state
            for count in successful_repeats_by_state.values()
        )
        print(
            f"[vis-collect] successful-repeat coverage: {reached}/"
            f"{len(successful_repeats_by_state)} states x "
            f"{args_cli.successful_repeats_per_state} successes"
        )
        quota_shortfall = {
            state_id: count
            for state_id, count in successful_repeats_by_state.items()
            if count < args_cli.successful_repeats_per_state
        }
    else:
        quota_shortfall = {}
    if args_cli.joint_teacher:
        print(
            f"[vis-collect] joint-teacher checks: home_starts={home_checks}, "
            "labels are exact post-limit targets executed in successful episodes"
        )
    elif args_cli.joint_target_bridge:
        print(
            f"[vis-collect] joint-target checks: home_starts={home_checks}, "
            f"invalid_ik={ik_invalid_total}, max_record_vs_write_error={max_target_write_error:.3e}"
        )
        if bridge_command_frames:
            joint_elements = bridge_command_frames * 7
            tracking_mean = (
                bridge_tracking_abs_sum / bridge_tracking_elements
                if bridge_tracking_elements
                else float("nan")
            )
            print(
                "[vis-collect] bridge diagnostics: "
                f"raw_delta_l2_mean/max={bridge_raw_delta_l2_sum / bridge_command_frames:.4f}/"
                f"{bridge_raw_delta_l2_max:.4f}rad, "
                f"applied_delta_l2_mean/max={bridge_applied_delta_l2_sum / bridge_command_frames:.4f}/"
                f"{bridge_applied_delta_l2_max:.4f}rad, "
                f"rate_cap_elements={bridge_rate_cap_elements}/{joint_elements} "
                f"({bridge_rate_cap_elements / joint_elements:.3f}), "
                f"rate_cap_frames={bridge_rate_cap_frames}/{bridge_command_frames} "
                f"({bridge_rate_cap_frames / bridge_command_frames:.3f}), "
                f"tracking_abs_mean/max={tracking_mean:.4f}/{bridge_tracking_abs_max:.4f}rad, "
                f"gripper_open/close={bridge_gripper_open_frames}/{bridge_gripper_close_frames}"
            )
    elif args_cli.osc_joint_trace:
        print(
            f"[vis-collect] OSC joint-trace checks: home_starts={home_checks}, "
            f"gripper_mode={args_cli.osc_joint_trace_gripper_mode}, "
            "candidate targets require a separate joint-controller replay gate"
        )
    elif args_cli.native_osc_guard_labels:
        print(
            "[vis-collect] native OSC guard-label checks: stored gripper action is the guard's "
            "executed -1/+1 command; raw ignored teacher output is preserved separately; "
            "student eval must remove the guard"
        )
    elif args_cli.native_osc_policy_gripper:
        print(
            "[vis-collect] native OSC checks: stored arm action is the teacher mean; stored gripper "
            "action is its executed sign; raw 7-D teacher outputs are preserved separately"
        )
    if diagnostic_xyz_m:
        xyz = np.asarray(diagnostic_xyz_m, dtype=np.float64)
        orientation = np.asarray(diagnostic_orientation_rad, dtype=np.float64)
        relaxed = (xyz < args_cli.diagnostic_position_m) & (
            orientation < np.deg2rad(args_cli.diagnostic_orientation_deg)
        )
        print(
            "[vis-collect] endpoint placement diagnostics: "
            f"relaxed={int(relaxed.sum())}/{len(relaxed)} "
            f"(<{args_cli.diagnostic_position_m * 1000:g}mm, "
            f"<{args_cli.diagnostic_orientation_deg:g}deg); "
            f"xyz_mm[min/median/max]={xyz.min() * 1000:.2f}/"
            f"{np.median(xyz) * 1000:.2f}/{xyz.max() * 1000:.2f}; "
            f"orientation_deg[min/median/max]={np.rad2deg(orientation.min()):.2f}/"
            f"{np.rad2deg(np.median(orientation)):.2f}/"
            f"{np.rad2deg(orientation.max()):.2f}"
        )
    if args_cli.outcomes_only:
        if len(outcome_records) != len(set(forced_reset_state_indices or [])):
            raise RuntimeError(
                f"outcome-only audit has {len(outcome_records)} records, expected "
                f"{len(set(forced_reset_state_indices or []))}"
            )
        os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
        with open(args_cli.output, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "collection_config": _plain_value(collection_config),
                    "completed_states": len(outcome_records),
                    "successful_states": successful_episodes,
                    "outcomes": sorted(outcome_records, key=lambda item: item["state_id"]),
                },
                stream,
                indent=2,
            )
            stream.write("\n")
        print(f"[vis-collect] compact outcome audit -> {args_cli.output}")
    elif stream_writer is not None:
        stream_writer.close()
    elif demos:
        save_zarr(demos, cam_keys, numeric_keys, args_cli.output, collection_config)
    else:
        print("[vis-collect] no successful demos; output dataset was not written")
    env.close()
    if quota_shortfall:
        raise RuntimeError(
            "collection stopped before reaching the successful-repeat quota; "
            f"shortfall={quota_shortfall}"
        )


def save_zarr(demos, cam_keys, numeric_keys, output_path, collection_config):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ends = np.cumsum([len(d["state"]) for d in demos]).astype(np.int64)
    root = zarr.open(output_path, mode="w")
    root.attrs["collection_config"] = collection_config
    compressor = _zarr_compressor(collection_config["zarr_compressor"])
    data = root.create_group("data")
    for k in cam_keys:
        arr = np.concatenate([d[k] for d in demos], 0).astype(np.uint8)  # (T,224,224,3)
        data.create_dataset(
            k, data=arr, chunks=(64,) + arr.shape[1:], compressor=compressor
        )
        print(f"[vis-collect]   {k}: {arr.shape} {arr.dtype}")
    for k in numeric_keys:
        dtype = np.int64 if k == "reset_state_id" else np.float32
        arr = np.concatenate([d[k] for d in demos], 0).astype(dtype)
        if k == "action" and collection_config["gripper_supervision"] in {
            "grasp_guard_binary",
            "grasp_guard_binary_verified",
        }:
            labels, counts = np.unique(arr[:, -1], return_counts=True)
            if not np.all(np.isin(labels, (-1.0, 1.0))):
                raise ValueError(f"invalid grasp-guard labels in action[:, -1]: {labels.tolist()}")
            label_counts = {float(label): int(count) for label, count in zip(labels, counts)}
            print(f"[vis-collect]   gripper labels: {label_counts} (-1 close, +1 open)")
        elif k == "action" and collection_config["gripper_supervision"] in {
            "grasp_guard_joint_width_target",
            "fixed_open_joint_width_target",
        }:
            close = np.isclose(arr[:, -1], 0.0, atol=1e-6)
            opened = np.isclose(arr[:, -1], 0.08, atol=1e-6)
            valid = opened | (
                close
                if collection_config["gripper_supervision"] == "grasp_guard_joint_width_target"
                else np.zeros_like(close)
            )
            if not np.all(valid):
                values = np.unique(arr[~valid, -1])
                raise ValueError(f"invalid gripper width targets in action[:, -1]: {values.tolist()}")
            print(
                "[vis-collect]   gripper width targets: "
                f"close={int(close.sum())}, open={int(opened.sum())}"
            )
        data.create_dataset(
            k,
            data=arr,
            chunks=(min(1024, arr.shape[0]), arr.shape[1]),
            compressor=compressor,
        )
        print(f"[vis-collect]   {k}: {arr.shape} {arr.dtype}")
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=ends, compressor=compressor)
    if "episode_success" in numeric_keys:
        meta.create_dataset(
            "episode_success",
            data=np.asarray(
                [bool(np.asarray(d["episode_success"])[0, 0]) for d in demos],
                dtype=np.bool_,
            ),
            compressor=compressor,
        )
    if "episode_fixed_gate_success" in numeric_keys:
        meta.create_dataset(
            "episode_fixed_gate_success",
            data=np.asarray(
                [bool(np.asarray(d["episode_fixed_gate_success"])[0, 0]) for d in demos],
                dtype=np.bool_,
            ),
            compressor=compressor,
        )
    if "reset_state_id" in numeric_keys:
        episode_state_ids = np.asarray([int(d["reset_state_id"][0, 0]) for d in demos], dtype=np.int64)
        for demo, state_id in zip(demos, episode_state_ids):
            if not np.all(np.asarray(demo["reset_state_id"]) == state_id):
                raise ValueError("reset_state_id changed within an episode")
        meta.create_dataset(
            "reset_state_ids", data=episode_state_ids, compressor=compressor
        )
    print(f"[vis-collect] saved {len(ends)} demos -> {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()  # always close, else a failed run leaves a hung Isaac process
