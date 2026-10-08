#!/usr/bin/env bash
set -Eeuo pipefail

RELEASE_ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: sim2sim.sh CONFIG.env TASK [raw|rollouts|all]}"
TASK="${2:?set TASK to peg, stackcube, or cupcake}"
STAGE="${3:-all}"
CONFIG="$(readlink -f -- "${CONFIG}")"
set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a

: "${OMNIRESET_ROOT:?set OMNIRESET_ROOT}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${WORK_ROOT:?set WORK_ROOT}"
PY_ISAAC="${PY_ISAAC:-python}"
PY_MUJOCO="${PY_MUJOCO:-python}"
GPU="${GPU:-0}"
GPUS="${GPUS:-${GPU}}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
[[ "${TASK}" =~ ^(peg|stackcube|cupcake)$ ]] || { printf 'unknown task: %s\n' "${TASK}" >&2; exit 2; }
[[ "${STAGE}" =~ ^(raw|rollouts|all)$ ]] || { printf 'stage must be raw, rollouts, or all\n' >&2; exit 2; }

OMNIRESET_ROOT="$(readlink -f -- "${OMNIRESET_ROOT}")"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${RELEASE_ROOT}/artifacts}"
export PYTHONNOUSERSITE=1
export OMNIRESET_DATASET_DIR="${ARTIFACT_ROOT}/datasets/OmniReset"
export UWLAB_CLOUD_ASSETS_DIR="${ARTIFACT_ROOT}/cloud"
export UWLAB_LOCAL_ASSETS_DIR="${ARTIFACT_ROOT}/local_assets"
export TUCO_ARTIFACTS="${ARTIFACT_ROOT}"
export TUCO_ISAAC_ASSETS="${ARTIFACT_ROOT}/native/Isaac"
export PYTHONPATH="${RELEASE_ROOT}/src:${RELEASE_ROOT}/third_party/rsl_rl"
for package_root in "${OMNIRESET_ROOT}"/UWLab/_isaaclab/IsaacLab/source/* "${OMNIRESET_ROOT}"/UWLab/source/*; do
  [[ ! -d "${package_root}" ]] || PYTHONPATH="${PYTHONPATH}:${package_root}"
done
cd "${OMNIRESET_ROOT}"
OUT="${DATA_ROOT}/sim2sim/${TASK}"
WORK="${WORK_ROOT}/sim2sim/${TASK}"
TARGET_ZARR="${TARGET_ZARR:-${OUT}/target_mujoco.zarr}"
SOURCE_ZARR="${SOURCE_ZARR:-${OUT}/source_isaacsim.zarr}"
ROLLOUT_ZARR="${ROLLOUT_ZARR:-${OUT}/target_policy_rollouts.zarr}"
BASE_DIR="${BASE_DIR:-${WORK}/target_only}"
BASE_CKPT="${BASE_CKPT:-${BASE_DIR}/final.pt}"
COLLECT_STATE="${RELEASE_ROOT}/experiments/data_generation/collect_state_demos.py"

run() {
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_file() {
  [[ "${DRY_RUN}" == 1 || -f "$1" ]] || { printf 'missing file: %s\n' "$1" >&2; exit 1; }
}

require_dir() {
  [[ "${DRY_RUN}" == 1 || -d "$1" ]] || { printf 'missing directory: %s\n' "$1" >&2; exit 1; }
}

run_if_missing() {
  local output="$1"; shift
  if [[ -e "${output}" ]]; then
    printf '[skip] %s\n' "${output}"
  else
    run "$@"
    [[ "${DRY_RUN}" == 1 || -e "${output}" ]] || {
      printf 'command returned without creating %s\n' "${output}" >&2
      exit 1
    }
  fi
}

common_isaac_env=(
  "CUDA_VISIBLE_DEVICES=${GPU}"
  "OMNIRESET_ROOT=${OMNIRESET_ROOT}"
  "OMNI_KIT_ACCEPT_EULA=YES"
  "UWLAB_LOCAL_ASSETS_DIR=${ARTIFACT_ROOT}/local_assets"
)

collect_peg_raw() {
  : "${PEG_TEACHER:?set PEG_TEACHER}"
  local reset="${PEG_RESET_PT:-${OMNIRESET_ROOT}/Datasets/OmniReset/Resets/Peg__PegHole/resets_ObjectAnywhereEEAnywhere_upright_yaw_3cm.pt}"
  require_file "${PEG_TEACHER}"; require_file "${reset}"; require_file "${COLLECT_STATE}"
  run mkdir -p "${OUT}" "${WORK}"
  run_if_missing "${TARGET_ZARR}" env MUJOCO_GL=egl \
    "${PY_MUJOCO}" -u "${RELEASE_ROOT}/third_party/omnireset_eval/scripts/peg_reversed_ab/mujoco_active_big.py" \
      --mode collect --reset_pt "${reset}" --checkpoint "${PEG_TEACHER}" \
      --output "${TARGET_ZARR}" --num_demos 500 --candidate_start 0 \
      --candidate_stop 5000 --workers "${MUJOCO_WORKERS:-8}"
  run_if_missing "${SOURCE_ZARR}" env "${common_isaac_env[@]}" \
    "${PY_ISAAC}" -u "${COLLECT_STATE}" \
      --task OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-v0 \
      --checkpoint "${PEG_TEACHER}" --num_envs 256 --num_demos 3000 \
      --output "${SOURCE_ZARR}" --reset_type ObjectAnywhereEEAnywhere_upright_yaw_3cm \
      --expected_obs_dim 200 --expected_act_dim 7 --expected_episode_steps 160 \
      --expected_receptive_usd_basename peg_hole_big.usd --max_steps 200000 \
      --seed "${SEED}" --headless --device cuda:0 \
      env.scene.insertive_object=peg env.observations.policy.enable_corruption=False \
      'env.scene.table.init_state.pos=[0.4,0.0,-0.881]' \
      env.scene.robot.actuators.panda_hand.stiffness=1000.0 \
      env.scene.robot.actuators.panda_hand.damping=14.0 \
      env.scene.robot.actuators.panda_hand.effort_limit_sim=60.0
  PEG_RESET_PT="${reset}"
}

collect_stackcube_raw() {
  : "${STACKCUBE_TEACHER:?set STACKCUBE_TEACHER}"
  local reset="${STACKCUBE_RESET_PT:-${OMNIRESET_ROOT}/Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube/resets_ObjectAnywhereEEAnywhere.pt}"
  local collector="${COLLECT_STATE}"
  local subset="${RELEASE_ROOT}/third_party/omnireset_eval/scripts/stackcube/subset_episodes.py"
  local manifest="${RELEASE_ROOT}/third_party/omnireset_eval/scripts/stackcube/make_reset_index_manifest.py"
  local export_runtime="${RELEASE_ROOT}/experiments/data_generation/export_stackcube_runtime.sh"
  local replay="${RELEASE_ROOT}/experiments/data_generation/collect_stackcube_mujoco.sh"
  local aggregate="${RELEASE_ROOT}/third_party/omnireset_eval/scripts/stackcube/aggregate_mujoco_matched_full160.py"
  for file in "${STACKCUBE_TEACHER}" "${reset}" "${collector}" "${subset}" \
    "${manifest}" "${export_runtime}" "${replay}" "${aggregate}"; do require_file "${file}"; done
  run mkdir -p "${OUT}" "${WORK}"

  local source10k="${WORK}/source_isaacsim_10000.zarr"
  run_if_missing "${source10k}" env "${common_isaac_env[@]}" \
    "${PY_ISAAC}" -u "${collector}" \
      --task OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-Finetune-Play-v0 \
      --checkpoint "${STACKCUBE_TEACHER}" --num_envs 256 --num_demos 10000 \
      --output "${source10k}" --seed "${SEED}" \
      --expected_receptive_usd_basename receptive_cube.usd \
      --headless --device cuda:0 \
      env.scene.insertive_object=cube env.scene.receptive_object=cube \
      'env.scene.table.init_state.pos=[0.4,0.0,-0.881]' \
      env.scene.robot.actuators.panda_hand.stiffness=1000.0 \
      env.scene.robot.actuators.panda_hand.damping=14.0 \
      env.scene.robot.actuators.panda_hand.effort_limit_sim=60.0
  run_if_missing "${SOURCE_ZARR}" "${PY_ISAAC}" -u "${subset}" \
    --source "${source10k}" --out "${SOURCE_ZARR}" --episodes 5000 \
    --seed "${SEED}" --sampling nested_prefix

  local train_manifest="${WORK}/train_candidates.zarr"
  local score_manifest="${WORK}/score_resets.zarr"
  local eval_manifest="${WORK}/eval_resets.zarr"
  run_if_missing "${train_manifest}" "${PY_ISAAC}" "${manifest}" \
    --out "${train_manifest}" --start 0 --stride 1 \
    --count "${STACKCUBE_TRAIN_CANDIDATES:-1800}" --reset_total 2500 --random_seed 4646
  run_if_missing "${score_manifest}" "${PY_ISAAC}" "${manifest}" \
    --out "${score_manifest}" --start 0 --stride 1 --count 100 \
    --reset_total 2500 --random_seed 4747 --exclude_zarr "${train_manifest}"
  run_if_missing "${eval_manifest}" "${PY_ISAAC}" "${manifest}" \
    --out "${eval_manifest}" --start 0 --stride 1 --count 100 \
    --reset_total 2500 --random_seed 4848 --exclude_zarr "${train_manifest}" \
    --exclude_zarr "${score_manifest}"

  local train_runtime="${WORK}/train_runtime.zarr"
  STACKCUBE_SCORE_RUNTIME="${STACKCUBE_SCORE_RUNTIME:-${OUT}/score_runtime.zarr}"
  STACKCUBE_EVAL_RUNTIME="${STACKCUBE_EVAL_RUNTIME:-${OUT}/eval_runtime.zarr}"
  run_if_missing "${train_runtime}" env GPU="${GPU}" bash "${export_runtime}" \
    "${OMNIRESET_ROOT}" "${PY_ISAAC}" "${STACKCUBE_TEACHER}" "${reset}" \
    "${train_manifest}" "${train_runtime}"
  run_if_missing "${STACKCUBE_SCORE_RUNTIME}" env GPU="${GPU}" bash "${export_runtime}" \
    "${OMNIRESET_ROOT}" "${PY_ISAAC}" "${STACKCUBE_TEACHER}" "${reset}" \
    "${score_manifest}" "${STACKCUBE_SCORE_RUNTIME}"
  run_if_missing "${STACKCUBE_EVAL_RUNTIME}" env GPU="${GPU}" bash "${export_runtime}" \
    "${OMNIRESET_ROOT}" "${PY_ISAAC}" "${STACKCUBE_TEACHER}" "${reset}" \
    "${eval_manifest}" "${STACKCUBE_EVAL_RUNTIME}"

  local parts="${WORK}/mujoco_parts"
  if [[ ! -d "${parts}" || -z "$(find "${parts}" -maxdepth 1 -name '*.zarr' -print -quit 2>/dev/null)" ]]; then
    run env PY_MUJOCO="${PY_MUJOCO}" bash "${replay}" "${train_runtime}" \
      "${STACKCUBE_TEACHER}" "${parts}" 8 "${MUJOCO_WORKERS:-16}"
  fi
  run_if_missing "${TARGET_ZARR}" "${PY_MUJOCO}" -u "${aggregate}" \
    --parts "${parts}" --out "${TARGET_ZARR}" --demos 1000 \
    --seed "${SEED}" --max_abs_action 100
}

collect_cupcake_raw() {
  : "${CUPCAKE_TEACHER:?set CUPCAKE_TEACHER}"
  local build="${RELEASE_ROOT}/third_party/omnireset_eval/scripts/sim2sim/franka/build_cupcake_fixedhome_reset.py"
  local panels_py="${RELEASE_ROOT}/third_party/omnireset_eval/scripts/cupcake/prepare_fixedhome_panels.py"
  local collect_b="${RELEASE_ROOT}/experiments/data_generation/collect_cupcake_isaac.sh"
  local collect_a="${RELEASE_ROOT}/third_party/omnireset_eval/scripts/cupcake/collect_cupcake_mujoco_demos.py"
  for file in "${CUPCAKE_TEACHER}" "${build}" "${panels_py}" "${collect_b}" "${collect_a}"; do require_file "${file}"; done
  run mkdir -p "${OUT}" "${WORK}"
  local raw_root="${WORK}/reset_raw"
  local raw="${raw_root}/Resets/CupCake__Plate/resets_CupCakeSideLyingFront3cm.pt"
  local filtered_root="${WORK}/reset_fixedhome"
  local filtered="${filtered_root}/Resets/CupCake__Plate/resets_CupCakeSideLyingFront3cmFixedHome.pt"
  local panels="${WORK}/panels"
  if [[ ! -f "${raw}" ]]; then
    run env "${common_isaac_env[@]}" "${PY_ISAAC}" -u \
      "${OMNIRESET_ROOT}/UWLab/scripts_v2/tools/record_reset_states.py" \
      --task OmniReset-FrankaPanda-CupCakeSideLyingFront3cm-v0 \
      --reset_type CupCakeSideLyingFront3cm --num_envs 128 \
      --num_reset_states 400 --seed 20260818 --dataset_dir "${raw_root}" \
      --headless --device cuda:0 env.scene.insertive_object=cupcake \
      env.scene.receptive_object=plate
  fi
  run_if_missing "${filtered}" "${PY_ISAAC}" "${build}" --input "${raw}" \
    --output "${filtered}" --report "${filtered_root}/provenance.json" --min-count 300
  run_if_missing "${panels}/panel_provenance.json" "${PY_ISAAC}" "${panels_py}" \
    --reset-state "${filtered}" --output-dir "${panels}" --split-seed 20260819 \
    --attempt-seed 43 --train-count 250 --eval-count 50 --a-attempt-count 1800
  run_if_missing "${SOURCE_ZARR}" env OMNIRESET_ROOT="${OMNIRESET_ROOT}" \
    PY_ISAAC="${PY_ISAAC}" CUPCAKE_TEACHER="${CUPCAKE_TEACHER}" GPUS="${GPUS}" \
    SEED="${SEED}" OUTPUT="${SOURCE_ZARR}" RESET_DATASET_DIR="${panels}/isaac_train" \
    bash "${collect_b}"
  run_if_missing "${TARGET_ZARR}" env CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PY_MUJOCO}" -u "${collect_a}" --source-zarr "${SOURCE_ZARR}" \
      --reset-pool-npz "${panels}/mujoco_A_attempt_pool_n1800_seed43.npz" \
      --checkpoint "${CUPCAKE_TEACHER}" --output "${TARGET_ZARR}" \
      --num-demos 1000 --attempt-limit 1800 --max-abs-action 100 \
      --batch-size 32 --policy-microbatch-size 1 --steps 160 \
      --device cuda:0 --seed 43
  CUPCAKE_SOURCE_RUNTIME="${SOURCE_ZARR}"
  CUPCAKE_SCORE_RESETS="${panels}/mujoco_A_attempt_pool_n1800_seed43.npz"
  CUPCAKE_EVAL_RESETS="${panels}/mujoco_eval_holdout_n50_seed20260819.npz"
}

train_target_base() {
  if [[ ! -f "${BASE_CKPT}" ]]; then
    run "${PY_MUJOCO}" "${RELEASE_ROOT}/experiments/sim2sim/train_base.py" \
      --target "${TARGET_ZARR}" --output "${BASE_DIR}" --steps 50000 \
      --checkpoint-every 1000 --keep-last 5 --num-workers 4 \
      --seed "${SEED}" --device cuda:0
  fi
}

collect_rollouts() {
  train_target_base
  case "${TASK}" in
    peg)
      local reset="${PEG_RESET_PT:-${OMNIRESET_ROOT}/Datasets/OmniReset/Resets/Peg__PegHole/resets_ObjectAnywhereEEAnywhere_upright_yaw_3cm.pt}"
      run_if_missing "${ROLLOUT_ZARR}" env MUJOCO_GL=egl \
        "${PY_MUJOCO}" -u "${RELEASE_ROOT}/third_party/omnireset_eval/scripts/peg_reversed_ab/mujoco_active_big.py" \
          --mode eval --reset_pt "${reset}" --checkpoints "${BASE_CKPT}" \
          --eval_start 300 --eval_stop 400 --num_episodes 100 \
          --workers "${MUJOCO_WORKERS:-8}" --save_rollouts "${ROLLOUT_ZARR}"
      ;;
    stackcube)
      local runtime="${STACKCUBE_SCORE_RUNTIME:-${OUT}/score_runtime.zarr}"
      run_if_missing "${ROLLOUT_ZARR}" env MUJOCO_GL=egl \
        "${PY_MUJOCO}" -u "${RELEASE_ROOT}/third_party/omnireset_eval/scripts/stackcube/eval_mujoco_matched_runtime.py" \
          --runtime_source "${runtime}" --checkpoint "${BASE_CKPT}" \
          --out "${WORK}/score_rollout_eval" --episodes 100 --horizon 160 \
          --release_steps 60 --stable_window 10 --device cpu \
          --save_rollouts "${ROLLOUT_ZARR}"
      ;;
    cupcake)
      local runtime="${CUPCAKE_SOURCE_RUNTIME:-${SOURCE_ZARR}}"
      local resets="${CUPCAKE_SCORE_RESETS:-${WORK}/panels/mujoco_A_attempt_pool_n1800_seed43.npz}"
      local episodes
      episodes="$(seq -s, 0 99)"
      run_if_missing "${ROLLOUT_ZARR}" env CUDA_VISIBLE_DEVICES="${GPU}" \
        "${PY_MUJOCO}" -u "${RELEASE_ROOT}/third_party/omnireset_eval/scripts/sim2sim/franka/eval_cupcake_mujoco_rl.py" \
          --policy-type mlp_bc --source-zarr "${runtime}" --reset-pool-npz "${resets}" \
          --checkpoint "${BASE_CKPT}" --out "${WORK}/score_rollout_eval" \
          --save-rollouts "${ROLLOUT_ZARR}" --episodes "${episodes}" \
          --batch-size 25 --steps 160 --device cuda:0 --seed 20260819 \
          --physics-substeps 16 --friction-combine physx_average \
          --cupcake-collision convex_decomposition --hand-collision source_usd \
          --cupcake-plate-collision base_cylinder --finger-collision mimic
      ;;
  esac
}

if [[ "${STAGE}" == raw || "${STAGE}" == all ]]; then
  "collect_${TASK}_raw"
fi
if [[ "${STAGE}" == rollouts || "${STAGE}" == all ]]; then
  collect_rollouts
fi
if [[ "${DRY_RUN}" != 1 ]]; then
  rollout_args=()
  [[ "${STAGE}" == raw ]] || rollout_args+=(--rollouts "${ROLLOUT_ZARR}")
  "${PY_MUJOCO}" "${RELEASE_ROOT}/scripts/validate_data.py" sim2sim \
    --task "${TASK}" --target "${TARGET_ZARR}" --source "${SOURCE_ZARR}" \
    "${rollout_args[@]}"
fi
printf '[done] sim2sim %s data: %s\n' "${TASK}" "${OUT}"
