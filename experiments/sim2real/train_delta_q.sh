#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CUPID_ROOT="${CUPID_ROOT:-${ROOT}/third_party/cupid}"
PY="${PY:-python}"
: "${TASK:?set TASK to peg or stackcube}"
: "${DATASET_NAME:?set DATASET_NAME}"
: "${HDF5:?set HDF5}"
: "${CACHE:?set CACHE}"
: "${REAL_ROOT:?set REAL_ROOT}"
: "${RUN_DIR:?set RUN_DIR}"
: "${PHYSICAL_STATES:?set PHYSICAL_STATES from the checked task protocol}"
: "${SELECTED_STATES:?set SELECTED_STATES from the checked task protocol}"
: "${TRAINING_VISUAL_REPEATS:?set TRAINING_VISUAL_REPEATS from the checked task protocol}"
: "${REAL_SAMPLING_RATIO:?set REAL_SAMPLING_RATIO from the checked task protocol}"
: "${NUM_EPOCH_INDICES:?set NUM_EPOCH_INDICES from the checked task protocol}"
: "${TERMINAL_EPOCH:?set TERMINAL_EPOCH from the checked task protocol}"
: "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE from the checked task protocol}"
: "${MAX_STEPS_PER_EPOCH:?set MAX_STEPS_PER_EPOCH from the checked task protocol}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"

case "${TASK}" in
  peg)
    mode=filtered
    config_dir=configs/image/omnireset_peg_image_3cam_jointtarget_teamhome_300x10
    real_action_steps=1
    ;;
  stackcube)
    mode=materialized
    config_dir=configs/image/omnireset_stackcube_image_3cam_jointtarget_model1900_axis_v06_lift2cm_400x10
    real_action_steps=2
    ;;
  *) printf 'paper-aligned Sim-to-Real TASK must be peg or stackcube\n' >&2; exit 2 ;;
esac
if [[ "${PY}" == */* ]]; then
  [[ -x "${PY}" ]] || { printf 'python is not executable: %s\n' "${PY}" >&2; exit 1; }
else
  command -v "${PY}" >/dev/null || { printf 'python not found: %s\n' "${PY}" >&2; exit 1; }
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
for path in "${HDF5}" "${REAL_ROOT}"; do
  [[ -e "${path}" ]] || { printf 'missing input: %s\n' "${path}" >&2; exit 1; }
done
mkdir -p "$(dirname -- "${CACHE}")"
[[ ! -e "${RUN_DIR}" ]] || {
  printf 'refusing to overwrite %s\n' "${RUN_DIR}" >&2
  exit 2
}
"${PY}" "${ROOT}/experiments/sim2real/validate_real_data.py" \
  --real-root "${REAL_ROOT}" --action-steps "${real_action_steps}" \
  --cupid-root "${CUPID_ROOT}"

: "${SELECTION:?paper-facing training requires SELECTION}"
[[ -f "${SELECTION}" ]] || { printf 'missing input: %s\n' "${SELECTION}" >&2; exit 1; }
METHOD="${METHOD:-$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["method"])' "$(dirname -- "${SELECTION}")/metadata.json")}"
if [[ "${mode}" == filtered ]]; then
  : "${SOURCE_ZARR:?filtered mode requires SOURCE_ZARR}"
  [[ -e "${SOURCE_ZARR}" ]] || { printf 'missing input: %s\n' "${SOURCE_ZARR}" >&2; exit 1; }
else
  "${PY}" "${ROOT}/experiments/sim2real/validate_stackcube_hdf5.py" \
    --hdf5 "${HDF5}" --selected-ids "${SELECTION}"
fi

dataset_path="data/omnireset/datasets/${DATASET_NAME}/image.hdf5"
link="${CUPID_ROOT}/${dataset_path}"
mkdir -p "$(dirname -- "${link}")"
hdf5_target="$(readlink -f -- "${HDF5}")"
if [[ -e "${link}" || -L "${link}" ]]; then
  [[ "$(readlink -f -- "${link}")" == "${hdf5_target}" ]] || {
    printf 'dataset link points to a different HDF5: %s\n' "${link}" >&2
    exit 1
  }
else
  ln -s "${hdf5_target}" "${link}"
fi
mkdir -p "${RUN_DIR}"

dataset_overrides=(
  "task.dataset.dataset_path=${dataset_path}"
  task.dataset.val_ratio=0.04
  +task.dataset.joint_action_representation=delta_joint_step_v1
  +task.dataset.joint_control_dt_s=0.1
  +task.dataset.joint_max_velocity_rad_s=0.2
  +task.dataset.joint_gripper_max_width_m=0.08
  +task.dataset.normalizer_episode_limit=6000
  +task.dataset.normalizer_train_split_only=true
  "+task.dataset.cache_path=${CACHE}"
  "+task.dataset.real_decision_root=${REAL_ROOT}"
  "+task.dataset.real_sampling_ratio=${REAL_SAMPLING_RATIO}"
  "+task.dataset.domain_sampling_seed=${SEED}"
  "+task.dataset.real_action_steps=${real_action_steps}"
)
if [[ "${mode}" == filtered ]]; then
  dataset_module=diffusion_policy.dataset
  dataset_module+=.physical_state_filtered_decision_aligned_domain_balanced_image_dataset
  dataset_class=PhysicalStateFilteredDecisionAlignedDomainBalancedImageDataset
  dataset_target="${dataset_module}.${dataset_class}"
  dataset_overrides+=(
    "task.dataset._target_=${dataset_target}"
    "+task.dataset.source_zarr=${SOURCE_ZARR}"
    "+task.dataset.selected_state_ids_path=${SELECTION}"
    "+task.dataset.expected_selected_states=${SELECTED_STATES}"
    "+task.dataset.expected_repeats_per_state=${TRAINING_VISUAL_REPEATS}"
  )
else
  dataset_module=diffusion_policy.dataset.decision_aligned_domain_balanced_image_dataset
  dataset_class=DecisionAlignedDomainBalancedImageDataset
  dataset_target="${dataset_module}.${dataset_class}"
  dataset_overrides+=(
    "task.dataset._target_=${dataset_target}"
  )
fi

cd "${CUPID_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=offline \
  "${PY}" -u train.py --config-dir="${config_dir}" --config-name=config.yaml \
  training.device=cuda:0 "training.seed=${SEED}" training.resume=false \
  "training.num_epochs=${NUM_EPOCH_INDICES}" training.checkpoint_every=5 \
  training.val_every=5 "training.max_train_steps=${MAX_STEPS_PER_EPOCH}" \
  +training.camera_mask_mode=none \
  +training.camera_mask_keys='[front_rgb,side_rgb,wrist_rgb]' \
  policy.down_dims='[256,512,1024]' policy.obs_encoder.share_rgb_model=false \
  checkpoint.topk.k=0 "dataloader.batch_size=${TRAIN_BATCH_SIZE}" \
  "val_dataloader.batch_size=${TRAIN_BATCH_SIZE}" \
  dataloader.num_workers=8 val_dataloader.num_workers=4 \
  "hydra.run.dir=${RUN_DIR}" "logging.name=${TASK}_${METHOD}_real10" \
  "logging.tags=[${TASK},${METHOD},real${REAL_ROLLOUTS},selected${SELECTED_STATES},ratio${REAL_SAMPLING_RATIO},nomask,scratch]" \
  "${dataset_overrides[@]}"
