#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: bash experiments/sim2sim/run_eval.sh CONFIG.env [CHECKPOINT_DIR]}"
[[ -f "${CONFIG}" ]] || { printf 'missing config: %s\n' "${CONFIG}" >&2; exit 1; }
CONFIG="$(readlink -f -- "${CONFIG}")"
set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a

PY="${PY:-python}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
TASK="${TASK:?set TASK in ${CONFIG}}"
CHECKPOINT_DIR="${2:-${CHECKPOINT_DIR:-${OUTPUT_ROOT:?set OUTPUT_ROOT}/cotrain_${BUDGET:?set BUDGET}}}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${CHECKPOINT_DIR}/eval_last5}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
EVAL_EPISODE_START="${EVAL_EPISODE_START:-0}"

command=(
  "${PY}" -u "${ROOT}/experiments/sim2sim/evaluate.py"
  --task "${TASK}" --checkpoint-dir "${CHECKPOINT_DIR}"
  --output "${EVAL_OUTPUT}" --device "${EVAL_DEVICE}"
  --workers "${EVAL_WORKERS}" --episode-start "${EVAL_EPISODE_START}"
)
case "${TASK}" in
  peg)
    command+=(--peg-reset-pt "${PEG_RESET_PT:?set PEG_RESET_PT in ${CONFIG}}")
    ;;
  stackcube)
    command+=(--stackcube-runtime-zarr "${STACKCUBE_RUNTIME_ZARR:?set STACKCUBE_RUNTIME_ZARR in ${CONFIG}}")
    ;;
  cupcake)
    command+=(
      --cupcake-runtime-zarr "${CUPCAKE_RUNTIME_ZARR:?set CUPCAKE_RUNTIME_ZARR in ${CONFIG}}"
      --cupcake-reset-pool-npz "${CUPCAKE_RESET_POOL_NPZ:?set CUPCAKE_RESET_POOL_NPZ in ${CONFIG}}"
    )
    ;;
  *) printf 'TASK must be peg, stackcube, or cupcake\n' >&2; exit 2 ;;
esac
"${command[@]}"
