#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${SOURCE_HDF5:?set SOURCE_HDF5}"
: "${SOURCE_CACHE:?set SOURCE_CACHE}"
: "${SOURCE_ZARR:?set SOURCE_ZARR}"
: "${REAL_ROOT:?set REAL_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${TASK:?set TASK to peg, stackcube, or cupcake}"
: "${REAL_ROLLOUTS:?set REAL_ROLLOUTS from the checked task protocol}"
: "${REAL_ACTION_STEPS:?set REAL_ACTION_STEPS from the checked task protocol}"
: "${PHYSICAL_STATES:?set PHYSICAL_STATES from the checked task protocol}"
: "${SELECTED_STATES:?set SELECTED_STATES from the checked task protocol}"
: "${ATTRIBUTION_REPEAT_INDEX:?set ATTRIBUTION_REPEAT_INDEX from the checked task protocol}"

case "${TASK}" in
  peg)
    real_contract=delta_q
    candidate_contract=repository_relative
    ;;
  stackcube)
    real_contract=delta_q
    candidate_contract=repository_relative
    ;;
  cupcake)
    real_contract=cupcake_absolute_q
    candidate_contract=cupcake_absolute_q
    ;;
  *) printf 'TASK must be peg, stackcube, or cupcake\n' >&2; exit 2 ;;
esac

STATE_ORDER="${STATE_ORDER:-AUTO_SORTED}"
GPUS="${GPUS:-0 1 2 3}"
SEED="${SEED:-42}"
RESUME="${RESUME:-0}"
[[ "${RESUME}" == 0 || "${RESUME}" == 1 ]] || {
  printf 'RESUME must be 0 or 1\n' >&2
  exit 2
}

if [[ "${PY}" == */* ]]; then
  [[ -x "${PY}" ]] || { printf 'python is not executable: %s\n' "${PY}" >&2; exit 1; }
else
  command -v "${PY}" >/dev/null || { printf 'python not found: %s\n' "${PY}" >&2; exit 1; }
fi
for path in "${CHECKPOINT}" "${SOURCE_HDF5}" "${SOURCE_CACHE}" \
  "${SOURCE_ZARR}" "${REAL_ROOT}"; do
  [[ -e "${path}" ]] || { printf 'missing input: %s\n' "${path}" >&2; exit 1; }
done
if [[ "${STATE_ORDER}" != AUTO_SORTED && ! -f "${STATE_ORDER}" ]]; then
  printf 'missing state order: %s\n' "${STATE_ORDER}" >&2
  exit 1
fi

if [[ "${candidate_contract}" == repository_relative ]]; then
  # The Delta-Q dataset factory dispatches from a repository-relative path.
  source_hdf5="$(readlink -f -- "${SOURCE_HDF5}")"
  dataset_dir="$(basename -- "$(dirname -- "${source_hdf5}")")"
  dataset_file="$(basename -- "${source_hdf5}")"
  link="${ROOT}/third_party/cupid/data/omnireset/datasets/${dataset_dir}/${dataset_file}"
  mkdir -p "$(dirname -- "${link}")"
  if [[ -e "${link}" || -L "${link}" ]]; then
    [[ "$(readlink -f -- "${link}")" == "${source_hdf5}" ]] || {
      printf 'source link points to a different dataset: %s\n' "${link}" >&2
      exit 1
    }
  else
    ln -s "${source_hdf5}" "${link}"
  fi
fi

cupcake_code="${ROOT}/experiments/sim2real/cupcake"
export PYTHONPATH="${ROOT}/src:${cupcake_code}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -e "${OUTPUT_ROOT}" && "${RESUME}" != 1 ]]; then
  printf 'refusing to overwrite %s; set RESUME=1 to continue it\n' "${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/shards" "${OUTPUT_ROOT}/logs"
read -r -a gpu_ids <<<"${GPUS}"
num_shards="${#gpu_ids[@]}"
candidate_outputs=()
pids=()

common=(
  --checkpoint "${CHECKPOINT}"
  --source-hdf5 "${SOURCE_HDF5}"
  --source-cache "${SOURCE_CACHE}"
  --source-zarr "${SOURCE_ZARR}"
  --state-order "${STATE_ORDER}"
  --real-root "${REAL_ROOT}"
  --real-contract "${real_contract}"
  --candidate-contract "${candidate_contract}"
  --expected-rollouts "${REAL_ROLLOUTS}"
  --expected-action-steps "${REAL_ACTION_STEPS}"
  --repeat-index "${ATTRIBUTION_REPEAT_INDEX}"
  --batch-size 8
  --seed "${SEED}"
)

for worker in "${!gpu_ids[@]}"; do
  output="${OUTPUT_ROOT}/shards/candidate_${worker}.npz"
  candidate_outputs+=("${output}")
  CUDA_VISIBLE_DEVICES="${gpu_ids[$worker]}" "${PY}" -u \
    "${ROOT}/experiments/sim2real/extract_window_trak.py" \
    --mode candidate "${common[@]}" --device cuda:0 \
    --shard "${worker}" --num-shards "${num_shards}" --resume \
    --output "${output}" \
    >>"${OUTPUT_ROOT}/logs/candidate_${worker}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
[[ "${failed}" == 0 ]] || { printf 'candidate extraction failed\n' >&2; exit 1; }

if [[ ! -f "${OUTPUT_ROOT}/target_rollouts.npz" ]]; then
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${PY}" -u \
    "${ROOT}/experiments/sim2real/extract_window_trak.py" \
    --mode target "${common[@]}" --device cuda:0 \
    --output "${OUTPUT_ROOT}/target_rollouts.npz" \
    >"${OUTPUT_ROOT}/logs/target.log" 2>&1
fi

if [[ ! -f "${OUTPUT_ROOT}/influence.npz" ]]; then
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${PY}" -m tuco.cli.finalize \
    --candidate-shards "${candidate_outputs[@]}" \
    --target "${OUTPUT_ROOT}/target_rollouts.npz" \
    --output "${OUTPUT_ROOT}/influence.npz" --device cuda:0 \
    >"${OUTPUT_ROOT}/logs/finalize.log" 2>&1
fi

if [[ ! -d "${OUTPUT_ROOT}/selection" ]]; then
  "${PY}" -m tuco.cli.select \
    --input "${OUTPUT_ROOT}/influence.npz" \
    --output-dir "${OUTPUT_ROOT}/selection" \
    --budget "${SELECTED_STATES}" \
    >"${OUTPUT_ROOT}/logs/select.log" 2>&1
fi
"${PY}" -m tuco.cli.verify_selection \
  --selected-ids "${OUTPUT_ROOT}/selection/selected_ids.json" \
  --budget "${SELECTED_STATES}" --num-candidates "${PHYSICAL_STATES}"
touch "${OUTPUT_ROOT}/.complete"
printf 'selection complete: %s/selection\n' "${OUTPUT_ROOT}"
