#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: bash experiments/sim2real/run_baseline.sh CONFIG.env}"
[[ -f "${CONFIG}" ]] || { printf 'missing config: %s\n' "${CONFIG}" >&2; exit 1; }
CONFIG="$(readlink -f -- "${CONFIG}")"
set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a
: "${METHOD:?set METHOD in ${CONFIG}}"
: "${FEATURE_NPZ:?set FEATURE_NPZ in ${CONFIG}}"
: "${BUDGET:?set BUDGET in ${CONFIG}}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT in ${CONFIG}}"
[[ -f "${FEATURE_NPZ}" ]] || { printf 'missing features: %s\n' "${FEATURE_NPZ}" >&2; exit 1; }
PY="${PY:-python}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}_${BUDGET}"
if [[ ! -f "${OUTPUT_DIR}/selected_ids.json" ]]; then
  [[ ! -e "${OUTPUT_DIR}" ]] || {
    printf 'incomplete selection output exists: %s\n' "${OUTPUT_DIR}" >&2
    exit 2
  }
  "${PY}" "${ROOT}/experiments/sim2real/select_baseline.py" \
    --method "${METHOD}" --input "${FEATURE_NPZ}" --budget "${BUDGET}" \
    --seed "${SEED}" --output-dir "${OUTPUT_DIR}"
fi

if [[ "${TRAIN_POLICY:-1}" == 1 ]]; then
  : "${TASK:?set TASK in ${CONFIG} when TRAIN_POLICY=1}"
  run_dir="${BASELINE_RUN_DIR:-${OUTPUT_ROOT}/policy_${METHOD}_${BUDGET}}"
  SELECTION="${OUTPUT_DIR}/selected_ids.json" METHOD="${METHOD}" \
    RUN_DIR_OVERRIDE="${run_dir}" \
    bash "${ROOT}/experiments/sim2real/run.sh" "${CONFIG}" train
fi
