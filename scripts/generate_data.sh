#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TUCO_ROOT="${ROOT}"
SETTING="${1:?usage: bash scripts/generate_data.sh SETTING CONFIG.env [TASK] [STAGE]}"
CONFIG="${2:-}"

case "${SETTING}" in
  single-sim)
    destination="${CONFIG:-${ROOT}/third_party/cupid/data}"
    exec python "${ROOT}/scripts/download_data.py" single-sim \
      --destination "${destination}"
    ;;
  sim2sim|sim2real)
    [[ -n "${CONFIG}" && -f "${CONFIG}" ]] || {
      printf 'missing generation config: %s\n' "${CONFIG}" >&2
      exit 2
    }
    task="${3:?set TASK to peg, stackcube, or cupcake}"
    stage="${4:-all}"
    exec bash "${ROOT}/experiments/data_generation/${SETTING}.sh" \
      "${CONFIG}" "${task}" "${stage}"
    ;;
  *)
    printf 'SETTING must be single-sim, sim2sim, or sim2real\n' >&2
    exit 2
    ;;
esac
