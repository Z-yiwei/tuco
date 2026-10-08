#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CUPID_ROOT="${CUPID_ROOT:-${ROOT}/third_party/cupid}"
PY="${PY:-python}"

: "${BASE_CONFIG:?set BASE_CONFIG to the resolved absolute-Q CupCake YAML}"
: "${SELECTION:?set SELECTION to a curated selected_ids.json}"
: "${REAL_ROOT:?set REAL_ROOT to the prepared nine-rollout directory}"
: "${RUN_DIR:?set RUN_DIR}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
for path in "${BASE_CONFIG}" "${SELECTION}" "${REAL_ROOT}"; do
  [[ -e "${path}" ]] || { printf 'missing input: %s\n' "${path}" >&2; exit 1; }
done
[[ ! -e "${RUN_DIR}" ]] || {
  printf 'refusing to overwrite %s\n' "${RUN_DIR}" >&2
  exit 2
}

export PYTHONPATH="${ROOT}/src:${ROOT}/experiments/sim2real:${CUPID_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 WANDB_MODE=offline \
  "${PY}" -m cupcake.train \
  --base-config "${BASE_CONFIG}" \
  --selection "${SELECTION}" \
  --real-root "${REAL_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --seed "${SEED}" \
  --physical-states "${PHYSICAL_STATES}" \
  --selected-states "${SELECTED_STATES}" \
  --visual-repeats "${TRAINING_VISUAL_REPEATS}" \
  --real-ratio "${REAL_SAMPLING_RATIO}" \
  --num-epochs "${NUM_EPOCH_INDICES}" \
  --terminal-epoch "${TERMINAL_EPOCH}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --max-train-steps "${MAX_STEPS_PER_EPOCH}" \
  --device cuda:0
