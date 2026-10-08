#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: bash experiments/single_sim/run.sh CONFIG.env}"
[[ -f "${CONFIG}" ]] || { printf 'missing config: %s\n' "${CONFIG}" >&2; exit 1; }
CONFIG="$(readlink -f -- "${CONFIG}")"
set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a

PY="${PY:-python}"
CUPID_ROOT="${CUPID_ROOT:-${ROOT}/third_party/cupid}"
export PYTHONPATH="${ROOT}/src:${CUPID_ROOT}:${CUPID_ROOT}/third_party/trak${PYTHONPATH:+:${PYTHONPATH}}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT in ${CONFIG}}"
: "${TASK_MH:?set TASK_MH in ${CONFIG}}"
: "${SPLIT:?set SPLIT to filter or select in ${CONFIG}}"

# A fresh release checkout has no precomputed TRAK arrays.  When PREP_ROOT is
# supplied, prepare them from the RoboMimic HDF5 and reuse them on later runs.
if [[ -z "${DATA_ROOT:-}" ]]; then
  : "${PREP_ROOT:?set PREP_ROOT or DATA_ROOT in ${CONFIG}}"
  DATA_ROOT="${PREP_ROOT}/inputs_${SPLIT}"
fi
if [[ ! -f "${DATA_ROOT}/manifest.json" ]]; then
  : "${PREP_ROOT:?DATA_ROOT is not prepared; set PREP_ROOT in ${CONFIG}}"
  bash "${ROOT}/experiments/single_sim/prepare.sh" "${CONFIG}"
fi

PAIRWISE="${PAIRWISE:-${DATA_ROOT}/PAIRWISE.npy}"
TARGET_ENDS="${TARGET_ENDS:-${DATA_ROOT}/TARGET_ENDS.npy}"
CANDIDATE_ENDS="${CANDIDATE_ENDS:-${DATA_ROOT}/CANDIDATE_ENDS.npy}"
RETURNS="${RETURNS:-${DATA_ROOT}/RETURNS.npy}"
CANDIDATE_IDS="${CANDIDATE_IDS:-${DATA_ROOT}/CANDIDATE_IDS.npy}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/policy}"
CURATION_DIR="${OUTPUT_ROOT}/curation/seed_${SEED}"

case "${SPLIT}" in
  filter)
    RANKING_FILE=train_config.yaml
    TRAIN_RATIO="${TRAIN_RATIO:-0.64}"
    FILTER_RATIO="${FILTER_RATIO:-0.25}"
    SELECT_RATIO=0
    ;;
  select)
    RANKING_FILE=holdout_config.yaml
    TRAIN_RATIO="${TRAIN_RATIO:-0.16}"
    FILTER_RATIO=0
    SELECT_RATIO="${SELECT_RATIO:-0.25}"
    ;;
  *) printf 'SPLIT must be filter or select\n' >&2; exit 2 ;;
esac

protocol() {
  PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PY}" -m tuco.cli.protocol single_sim robomimic "$1"
}
if [[ "${SPLIT}" == filter ]]; then RATIO="${FILTER_RATIO}"; else RATIO="${SELECT_RATIO}"; fi
RATIO_KEY="$(${PY} -c 'import sys; print(f"r{round(float(sys.argv[1])*100):03d}")' "${RATIO}")"
NUM_EPOCHS="${NUM_EPOCHS:-$(protocol "curation_epochs.${SPLIT}.${TASK_MH}.${RATIO_KEY}")}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-$(protocol protocol.checkpoint_every)}"
ROLLOUT_LAST_N="${ROLLOUT_LAST_N:-$(protocol protocol.reported_late_evaluations)}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-$(protocol protocol.evaluation_rollouts)}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
VAL_RATIO="${VAL_RATIO:-0.04}"

for path in "${PAIRWISE}" "${TARGET_ENDS}" "${CANDIDATE_ENDS}" \
  "${RETURNS}" "${CANDIDATE_IDS}"; do
  [[ -f "${path}" ]] || { printf 'missing input: %s\n' "${path}" >&2; exit 1; }
done
[[ ! -e "${RUN_DIR}" ]] || { printf 'refusing to overwrite %s\n' "${RUN_DIR}" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}" "${CURATION_DIR}"
INFLUENCE="${OUTPUT_ROOT}/influence_${SPLIT}.npz"
SELECTION_DIR="${OUTPUT_ROOT}/tuco_${SPLIT}_full"
RANKING="${CURATION_DIR}/${RANKING_FILE}"

if [[ ! -f "${INFLUENCE}" ]]; then
  "${PY}" "${ROOT}/experiments/single_sim/build_influence.py" \
    --pairwise "${PAIRWISE}" \
    --pairwise-layout "${PAIRWISE_LAYOUT:-candidate_by_target}" \
    --target-episode-ends "${TARGET_ENDS}" \
    --candidate-episode-ends "${CANDIDATE_ENDS}" \
    --returns "${RETURNS}" \
    --candidate-ids "${CANDIDATE_IDS}" \
    --output "${INFLUENCE}"
fi

NUM_CANDIDATES="${NUM_CANDIDATES:-$("${PY}" -c 'import numpy as np,sys; print(len(np.load(sys.argv[1])))' "${CANDIDATE_ENDS}")}"
if [[ ! -f "${SELECTION_DIR}/selected_ids.json" ]]; then
  [[ ! -e "${SELECTION_DIR}" ]] || {
    printf 'incomplete selection directory already exists: %s\n' "${SELECTION_DIR}" >&2
    exit 2
  }
  "${PY}" -m tuco.cli.select \
    --input "${INFLUENCE}" \
    --budget "${NUM_CANDIDATES}" \
    --output-dir "${SELECTION_DIR}"
fi

if [[ ! -f "${RANKING}" ]]; then
  "${PY}" "${ROOT}/experiments/single_sim/write_cupid_ranking.py" \
    --selection-dir "${SELECTION_DIR}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --output "${RANKING}"
fi

cd "${CUPID_ROOT}"
train_overrides=(
  "training.num_epochs=${NUM_EPOCHS}"
  "training.checkpoint_every=${CHECKPOINT_EVERY}"
  "training.rollout_every=${CHECKPOINT_EVERY}"
  "+training.rollout_last_n_checkpoints=${ROLLOUT_LAST_N}"
  "task.env_runner.n_test=${EVAL_ROLLOUTS}"
  task.env_runner.n_test_vis=0 task.env_runner.n_train_vis=0
)
[[ "${MAX_TRAIN_STEPS}" == 0 ]] || train_overrides+=("training.max_train_steps=${MAX_TRAIN_STEPS}")
[[ -z "${DATASET_PATH:-}" ]] || train_overrides+=("task.dataset.dataset_path=${DATASET_PATH}")
"${PY}" train.py \
  --config-dir="configs/low_dim/${TASK_MH}/diffusion_policy_cnn" \
  --config-name=config.yaml \
  "hydra.run.dir=${RUN_DIR}" \
  "training.seed=${SEED}" \
  "task.dataset.seed=${SEED}" \
  "task.dataset.val_ratio=${VAL_RATIO}" \
  +task.dataset.dataset_mask_kwargs.uniform_quality=true \
  +task.dataset.dataset_mask_kwargs.curate_dataset=true \
  "+task.dataset.dataset_mask_kwargs.curation_config_dir=${CURATION_DIR}" \
  +task.dataset.dataset_mask_kwargs.curation_method=tuco \
  "+task.dataset.dataset_mask_kwargs.train_ratio=${TRAIN_RATIO}" \
  "+task.dataset.dataset_mask_kwargs.filter_ratio=${FILTER_RATIO}" \
  "+task.dataset.dataset_mask_kwargs.select_ratio=${SELECT_RATIO}" \
  training.resume=false logging.mode=disabled checkpoint.topk.k=10 \
  "${train_overrides[@]}"

if [[ "${SKIP_SUMMARY}" != 1 ]]; then
  "${PY}" "${ROOT}/experiments/single_sim/summarize.py" \
    --run-dir "${RUN_DIR}" --num-evaluations "${ROLLOUT_LAST_N}" \
    --rollouts-per-evaluation "${EVAL_ROLLOUTS}"
fi
