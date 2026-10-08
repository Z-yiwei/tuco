#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: bash experiments/sim2real/run.sh CONFIG.env [all|select|train]}"
STAGE="${2:-all}"
[[ -f "${CONFIG}" ]] || { printf 'missing config: %s\n' "${CONFIG}" >&2; exit 1; }
CONFIG="$(readlink -f -- "${CONFIG}")"
set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a

PY="${PY:-python}"
GPU="${GPU:-0}"
GPUS="${GPUS:-0 1 2 3}"
SEED="${SEED:-42}"
: "${TASK:?set TASK in ${CONFIG}}"
: "${REAL_ROOT:?set REAL_ROOT in ${CONFIG}}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT in ${CONFIG}}"

protocol() {
  PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PY}" -m tuco.cli.protocol sim2real "${TASK}" "$1"
}

# The checked task protocol is the single source of truth.  Launch files only
# carry machine-local paths, devices, and the experiment seed.
REAL_ROLLOUTS="$(protocol data.real_rollouts)"
REAL_ACTION_STEPS="$(protocol data.real_action_steps)"
PHYSICAL_STATES="$(protocol data.physical_states)"
SELECTED_STATES="$(protocol data.selected_states)"
ATTRIBUTION_REPEAT_INDEX="$(protocol data.attribution_repeat_index)"
TRAINING_VISUAL_REPEATS="$(protocol data.training_visual_repeats)"
REAL_SAMPLING_RATIO="$(protocol training.real_sampling_ratio)"
NUM_EPOCH_INDICES="$(protocol training.num_epoch_indices)"
TERMINAL_EPOCH="$(protocol training.terminal_epoch)"
TRAIN_BATCH_SIZE="$(protocol training.batch_size)"
MAX_STEPS_PER_EPOCH="$(protocol training.max_steps_per_epoch)"
export REAL_ROLLOUTS REAL_ACTION_STEPS PHYSICAL_STATES SELECTED_STATES
export ATTRIBUTION_REPEAT_INDEX TRAINING_VISUAL_REPEATS REAL_SAMPLING_RATIO
export NUM_EPOCH_INDICES TERMINAL_EPOCH TRAIN_BATCH_SIZE MAX_STEPS_PER_EPOCH

run_selection() {
  : "${CHECKPOINT:?set CHECKPOINT in ${CONFIG}}"
  : "${SOURCE_HDF5:?set SOURCE_HDF5 in ${CONFIG}}"
  : "${SOURCE_CACHE:?set SOURCE_CACHE in ${CONFIG}}"
  : "${SOURCE_ZARR:?set SOURCE_ZARR in ${CONFIG}}"
  if [[ -f "${OUTPUT_ROOT}/.complete" ]]; then
    printf 'selection already complete: %s\n' "${OUTPUT_ROOT}/selection"
    return
  fi
  RESUME=0
  [[ ! -e "${OUTPUT_ROOT}" ]] || RESUME=1
  TASK="${TASK}" PY="${PY}" CHECKPOINT="${CHECKPOINT}" \
    SOURCE_HDF5="${SOURCE_HDF5}" SOURCE_CACHE="${SOURCE_CACHE}" \
    SOURCE_ZARR="${SOURCE_ZARR}" REAL_ROOT="${REAL_ROOT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" GPUS="${GPUS}" SEED="${SEED}" \
    RESUME="${RESUME}" bash "${ROOT}/experiments/sim2real/run_selection.sh"
}

run_training() {
  RUN_DIR="${RUN_DIR_OVERRIDE:-${RUN_DIR:-}}"
  : "${RUN_DIR:?set RUN_DIR in ${CONFIG}}"
  SELECTION="${SELECTION:-${OUTPUT_ROOT}/selection/selected_ids.json}"
  [[ -f "${SELECTION}" ]] || { printf 'missing selection: %s\n' "${SELECTION}" >&2; exit 1; }
  METHOD=tuco
  case "${TASK}" in
    peg)
      : "${SOURCE_HDF5:?set SOURCE_HDF5 in ${CONFIG}}"
      : "${SOURCE_ZARR:?set SOURCE_ZARR in ${CONFIG}}"
      : "${TRAIN_CACHE:?set TRAIN_CACHE in ${CONFIG}}"
      DATASET_NAME="${DATASET_NAME:-peg_${METHOD}}"
      TASK=peg DATASET_NAME="${DATASET_NAME}" HDF5="${SOURCE_HDF5}" \
        CACHE="${TRAIN_CACHE}" SOURCE_ZARR="${SOURCE_ZARR}" \
        SELECTION="${SELECTION}" REAL_ROOT="${REAL_ROOT}" RUN_DIR="${RUN_DIR}" \
        PY="${PY}" GPU="${GPU}" SEED="${SEED}" METHOD="${METHOD}" \
        bash "${ROOT}/experiments/sim2real/train_delta_q.sh"
      ;;
    stackcube)
      : "${SOURCE_HDF5:?set SOURCE_HDF5 in ${CONFIG}}"
      : "${SOURCE_ZARR:?set SOURCE_ZARR in ${CONFIG}}"
      : "${VARIANTS_ROOT:?set VARIANTS_ROOT in ${CONFIG}}"
      : "${MATERIALIZED_HDF5:?set MATERIALIZED_HDF5 in ${CONFIG}}"
      : "${TRAIN_CACHE:?set TRAIN_CACHE in ${CONFIG}}"
      DATASET_NAME="${DATASET_NAME:-stackcube_${METHOD}}"
      if [[ ! -f "${MATERIALIZED_HDF5}" ]]; then
        if [[ "${GENERATE_VARIANTS:-1}" == 1 ]]; then
          SELECTION="${SELECTION}" VARIANTS_ROOT="${VARIANTS_ROOT}" \
            GPU="${GPU}" bash "${ROOT}/experiments/sim2real/generate_variants.sh"
        fi
        "${PY}" "${ROOT}/experiments/sim2real/materialize_stackcube.py" \
          --source-zarr "${SOURCE_ZARR}" --source-hdf5 "${SOURCE_HDF5}" \
          --variants-root "${VARIANTS_ROOT}" --selected-ids "${SELECTION}" \
          --output-hdf5 "${MATERIALIZED_HDF5}"
      fi
      TASK=stackcube DATASET_NAME="${DATASET_NAME}" HDF5="${MATERIALIZED_HDF5}" \
        CACHE="${TRAIN_CACHE}" SELECTION="${SELECTION}" REAL_ROOT="${REAL_ROOT}" \
        RUN_DIR="${RUN_DIR}" PY="${PY}" GPU="${GPU}" SEED="${SEED}" METHOD="${METHOD}" \
        bash "${ROOT}/experiments/sim2real/train_delta_q.sh"
      ;;
    cupcake)
      : "${BASE_CONFIG:?set BASE_CONFIG in ${CONFIG}}"
      BASE_CONFIG="${BASE_CONFIG}" SELECTION="${SELECTION}" \
        REAL_ROOT="${REAL_ROOT}" RUN_DIR="${RUN_DIR}" \
        PY="${PY}" GPU="${GPU}" SEED="${SEED}" \
        bash "${ROOT}/experiments/sim2real/train_cupcake.sh"
      ;;
    *) printf 'TASK must be peg, stackcube, or cupcake\n' >&2; exit 2 ;;
  esac
}

case "${STAGE}" in
  all) run_selection; run_training ;;
  select) run_selection ;;
  train) run_training ;;
  *) printf 'stage must be all, select, or train\n' >&2; exit 2 ;;
esac
