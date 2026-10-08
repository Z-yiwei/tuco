#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if (( $# < 3 || $# > 5 )); then
  printf 'usage: %s RUNTIME_ZARR TEACHER PARTS_DIR [CHUNK] [MAX_JOBS]\n' "$0" >&2
  exit 2
fi
RUNTIME="$1"; TEACHER="$2"; PARTS="$3"
CHUNK="${4:-8}"; MAX_JOBS="${5:-16}"
PY="${PY_MUJOCO:-python}"
COLLECTOR="${ROOT}/third_party/omnireset_eval/scripts/stackcube/collect_mujoco_matched_cut0.py"
mkdir -p "${PARTS}"
episodes="$(${PY} - "${RUNTIME}" <<'PY'
import sys, zarr
print(len(zarr.open(sys.argv[1], mode="r")["meta/episode_ends"]))
PY
)"

pids=()
for ((start=0; start<episodes; start+=CHUNK)); do
  count=$((episodes - start < CHUNK ? episodes - start : CHUNK))
  output="${PARTS}/start${start}_count${count}.zarr"
  [[ ! -d "${output}" ]] || continue
  MUJOCO_GL= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "${PY}" -u "${COLLECTOR}" --source "${RUNTIME}" \
      --checkpoint "${TEACHER}" --out "${output}" --start "${start}" \
      --count "${count}" --device cpu --full_horizon \
      >"${PARTS}/start${start}_count${count}.log" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} >= MAX_JOBS )); then
    status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
    (( status == 0 )) || exit 1
    pids=()
  fi
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 ))
