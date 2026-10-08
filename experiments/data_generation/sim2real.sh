#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

RELEASE_ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: sim2real.sh CONFIG.env TASK [sim|prepare|all|validate-real]}"
TASK="${2:?set TASK to peg, stackcube, or cupcake}"
STAGE="${3:-all}"
CONFIG="$(readlink -f -- "${CONFIG}")"
set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a

: "${OMNIRESET_ROOT:?set OMNIRESET_ROOT}"
: "${SIM2REAL_DATA_ROOT:?set SIM2REAL_DATA_ROOT}"
PY_ISAAC="${PY_ISAAC:-python}"
GPUS="${GPUS:-0,1,2,3}"
DRY_RUN="${DRY_RUN:-0}"
[[ "${TASK}" =~ ^(peg|stackcube|cupcake)$ ]] || { printf 'unknown task: %s\n' "${TASK}" >&2; exit 2; }
[[ "${STAGE}" =~ ^(sim|prepare|all|validate-real)$ ]] || {
  printf 'stage must be sim, prepare, all, or validate-real\n' >&2; exit 2;
}

OMNIRESET_ROOT="$(readlink -f -- "${OMNIRESET_ROOT}")"
OUT="${SIM2REAL_DATA_ROOT}/${TASK}"

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

collect_sim() {
  local args=()
  [[ "${DRY_RUN}" != 1 ]] || args+=(--dry-run)
  "${PY_ISAAC}" "${RELEASE_ROOT}/tools/replay_expert.py" \
    --task "${TASK}" --artifacts "${ARTIFACT_ROOT:-${RELEASE_ROOT}/artifacts}" \
    --output "${OUT}/collection" --gpu "${GPU:-0}" "${args[@]}"
}

prepare_sim() {
  local final hdf5 cache env_name
  case "${TASK}" in
    peg)
      final="${OUT}/collection/states.zarr"
      env_name=OmniReset-Peg-FR3-XY5-JointTarget-Image
      ;;
    stackcube)
      final="${OUT}/collection/states.zarr"
      env_name=OmniReset-StackCube-FR3-XY5-JointTarget-Image
      ;;
    cupcake)
      final="${OUT}/collection/states.zarr"
      env_name=OmniReset-CupCake-FR3-XY5-JointTarget-Image
      ;;
  esac
  hdf5="${OUT}/prepared/${TASK}_sim6000/image.hdf5"
  cache="${OUT}/prepared/${TASK}_sim6000/replay_cache.zarr.zip"
  if [[ ! -f "${hdf5}" ]]; then
    run mkdir -p "$(dirname -- "${hdf5}")"
    run "${PY_ISAAC}" "${RELEASE_ROOT}/third_party/cupid/scripts/tools/convert_omnireset_image_to_cupid.py" \
      --zarr_path "${final}" --hdf5_path "${hdf5}" --val_ratio 0.04 \
      --env_name "${env_name}" --image_compression none --image_size 84
  fi
  if [[ ! -f "${cache}" ]]; then
    run "${PY_ISAAC}" "${RELEASE_ROOT}/experiments/data_generation/build_vision_cache.py" \
      --task "${TASK}" --hdf5 "${hdf5}" --cache "${cache}" --episodes 6000
  fi
}

validate_real() {
  : "${REAL_ROOT:?set REAL_ROOT to user-collected successful robot rollouts}"
  case "${TASK}" in
    peg)
      run "${PY_ISAAC}" "${RELEASE_ROOT}/experiments/sim2real/validate_real_data.py" \
        --real-root "${REAL_ROOT}" --action-steps 1 \
        --cupid-root "${RELEASE_ROOT}/third_party/cupid"
      ;;
    stackcube)
      run "${PY_ISAAC}" "${RELEASE_ROOT}/experiments/sim2real/validate_real_data.py" \
        --real-root "${REAL_ROOT}" --action-steps 2 \
        --cupid-root "${RELEASE_ROOT}/third_party/cupid"
      ;;
    cupcake)
      run "${PY_ISAAC}" -c \
        'from pathlib import Path; import sys; p=Path(sys.argv[1]); files=list(p.glob("**/rollout_sync.h5")); assert len(files)==9, f"expected 9 successful CupCake rollouts, found {len(files)}"; print("validated CupCake real9")' \
        "${REAL_ROOT}"
      ;;
  esac
}

case "${STAGE}" in
  sim) collect_sim ;;
  prepare) prepare_sim ;;
  all) collect_sim; prepare_sim ;;
  validate-real) validate_real ;;
esac

printf '[done] sim2real %s stage=%s\n' "${TASK}" "${STAGE}"
