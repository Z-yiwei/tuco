#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: bash experiments/sim2sim/run.sh CONFIG.env}"
[[ -f "${CONFIG}" ]] || { printf 'missing config: %s\n' "${CONFIG}" >&2; exit 1; }
CONFIG="$(readlink -f -- "${CONFIG}")"
set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a

PY="${PY:-python}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
TASK="${TASK:?set TASK to peg, stackcube, or cupcake in ${CONFIG}}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
protocol() { "${PY}" -m tuco.cli.protocol sim2sim "${TASK}" "$1"; }
STEPS="${STEPS:-$(protocol training.steps)}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-$(protocol training.checkpoint_every)}"
KEEP_LAST="${KEEP_LAST:-$(protocol training.keep_last)}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTION_DIM="${PROJECTION_DIM:-4000}"
GRADIENT_BATCH="${GRADIENT_BATCH:-32}"
ALLOW_PROTOCOL_SUBSET="${ALLOW_PROTOCOL_SUBSET:-0}"
: "${TARGET_ZARR:?set TARGET_ZARR in ${CONFIG}}"
: "${SOURCE_ZARR:?set SOURCE_ZARR in ${CONFIG}}"
: "${ROLLOUT_ZARR:?set ROLLOUT_ZARR in ${CONFIG}}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT in ${CONFIG}}"
: "${BUDGET:?set BUDGET in ${CONFIG}}"

for path in "${TARGET_ZARR}" "${SOURCE_ZARR}" "${ROLLOUT_ZARR}"; do
  [[ -e "${path}" ]] || { printf 'missing input: %s\n' "${path}" >&2; exit 1; }
done

validate_args=()
[[ "${ALLOW_PROTOCOL_SUBSET}" == 1 ]] && validate_args+=(--allow-subset)
"${PY}" "${ROOT}/scripts/validate_data.py" sim2sim --task "${TASK}" \
  --target "${TARGET_ZARR}" --source "${SOURCE_ZARR}" \
  --rollouts "${ROLLOUT_ZARR}" "${validate_args[@]}" >/dev/null

BASE_DIR="${OUTPUT_ROOT}/target_only"
INFLUENCE="${OUTPUT_ROOT}/influence.npz"
SELECTION_DIR="${OUTPUT_ROOT}/tuco_${BUDGET}"
RUN_DIR="${OUTPUT_ROOT}/cotrain_${BUDGET}"
mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${BASE_DIR}/final.pt" ]]; then
  [[ ! -e "${BASE_DIR}" ]] || { printf 'incomplete base run: %s\n' "${BASE_DIR}" >&2; exit 2; }
  "${PY}" "${ROOT}/experiments/sim2sim/train_base.py" \
    --target "${TARGET_ZARR}" --output "${BASE_DIR}" \
    --steps "${STEPS}" --checkpoint-every "${CHECKPOINT_EVERY}" \
    --keep-last "${KEEP_LAST}" --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" --device "${DEVICE}"
fi

if [[ ! -f "${INFLUENCE}" ]]; then
  "${PY}" "${ROOT}/experiments/sim2sim/build_influence.py" \
    --base "${BASE_DIR}/final.pt" \
    --candidates "${SOURCE_ZARR}" \
    --rollouts "${ROLLOUT_ZARR}" \
    --output "${INFLUENCE}" --device "${DEVICE}" \
    --projection-dim "${PROJECTION_DIM}" --gradient-batch "${GRADIENT_BATCH}"
fi

if [[ ! -f "${SELECTION_DIR}/selected_ids.json" ]]; then
  [[ ! -e "${SELECTION_DIR}" ]] || {
    printf 'incomplete selection directory: %s\n' "${SELECTION_DIR}" >&2
    exit 2
  }
  "${PY}" -m tuco.cli.select \
    --input "${INFLUENCE}" --budget "${BUDGET}" \
    --output-dir "${SELECTION_DIR}"
fi

if [[ ! -f "${RUN_DIR}/final.pt" ]]; then
  [[ ! -e "${RUN_DIR}" ]] || { printf 'incomplete cotrain run: %s\n' "${RUN_DIR}" >&2; exit 2; }
  "${PY}" "${ROOT}/experiments/sim2sim/train.py" \
    --base "${BASE_DIR}/final.pt" \
    --target "${TARGET_ZARR}" \
    --source "${SOURCE_ZARR}" \
    --selection "${SELECTION_DIR}/selected_ids.json" \
    --output "${RUN_DIR}" --steps "${STEPS}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" --keep-last "${KEEP_LAST}" \
    --num-workers "${NUM_WORKERS}" --seed "${SEED}" --device "${DEVICE}"
fi

printf 'completed: %s\n' "${RUN_DIR}"
