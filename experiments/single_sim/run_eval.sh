#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$(readlink -f -- "${1:?usage: run_eval.sh CONFIG.env [CHECKPOINT]}")"
set -a
source "${CONFIG}"
set +a
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/cupid:${ROOT}/third_party/cupid/third_party/trak${PYTHONPATH:+:${PYTHONPATH}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT:?set OUTPUT_ROOT}/policy}"
args=()
[[ -z "${2:-}" ]] || args+=(--checkpoint "$(readlink -f -- "$2")")
cd "${ROOT}/third_party/cupid"
"${PY:-python}" "${ROOT}/experiments/single_sim/evaluate.py" \
  --run-dir "${RUN_DIR}" --output "${EVAL_OUTPUT:-${RUN_DIR}/evaluation}" \
  --device "${DEVICE:-cuda:0}" "${args[@]}"
