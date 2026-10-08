#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: bash experiments/single_sim/run_baseline.sh CONFIG.env}"
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
: "${TASK_MH:?set TASK_MH in ${CONFIG}}"
: "${SPLIT:?set SPLIT to filter or select in ${CONFIG}}"
[[ -f "${FEATURE_NPZ}" ]] || { printf 'missing features: %s\n' "${FEATURE_NPZ}" >&2; exit 1; }
FEATURE_NPZ="$(readlink -f -- "${FEATURE_NPZ}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(readlink -f -- "${OUTPUT_ROOT}")"
PY="${PY:-python}"
CUPID_ROOT="${CUPID_ROOT:-${ROOT}/third_party/cupid}"
export PYTHONPATH="${ROOT}/src:${CUPID_ROOT}:${CUPID_ROOT}/third_party/trak${PYTHONPATH:+:${PYTHONPATH}}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD}_${BUDGET}"
[[ ! -e "${OUTPUT_DIR}" ]] || { printf 'output exists: %s\n' "${OUTPUT_DIR}" >&2; exit 2; }

CURATION_DIR="${OUTPUT_ROOT}/curation/${METHOD}_${BUDGET}_seed_${SEED}"
mkdir -p "${CURATION_DIR}"
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
NUM_CANDIDATES="$("${PY}" -c 'import numpy as np,sys; print(len(np.load(sys.argv[1], allow_pickle=False)["candidate_ids"]))' "${FEATURE_NPZ}")"
if [[ "${SPLIT}" == filter ]]; then
  EXPECTED_BUDGET="$("${PY}" -c 'import sys; n=int(sys.argv[1]); r=float(sys.argv[2]); print(n-int(n*r))' "${NUM_CANDIDATES}" "${FILTER_RATIO}")"
else
  EXPECTED_BUDGET="$("${PY}" -c 'import sys; print(int(int(sys.argv[1])*float(sys.argv[2])))' "${NUM_CANDIDATES}" "${SELECT_RATIO}")"
fi
[[ "${BUDGET}" -eq "${EXPECTED_BUDGET}" ]] || {
  printf 'BUDGET=%s does not match %s protocol budget=%s\n' \
    "${BUDGET}" "${SPLIT}" "${EXPECTED_BUDGET}" >&2
  exit 2
}
"${PY}" "${ROOT}/experiments/single_sim/select_baseline.py" \
  --method "${METHOD}" --input "${FEATURE_NPZ}" --budget "${BUDGET}" \
  --seed "${SEED}" --output-dir "${OUTPUT_DIR}"
"${PY}" "${ROOT}/experiments/single_sim/write_cupid_ranking.py" \
  --selection-dir "${OUTPUT_DIR}" --split "${SPLIT}" --seed "${SEED}" \
  --output "${CURATION_DIR}/${RANKING_FILE}"

RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/policy_${METHOD}_${BUDGET}}"
[[ ! -e "${RUN_DIR}" ]] || { printf 'run exists: %s\n' "${RUN_DIR}" >&2; exit 2; }
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
  "hydra.run.dir=${RUN_DIR}" "training.seed=${SEED}" \
  "task.dataset.seed=${SEED}" "task.dataset.val_ratio=${VAL_RATIO}" \
  +task.dataset.dataset_mask_kwargs.uniform_quality=true \
  +task.dataset.dataset_mask_kwargs.curate_dataset=true \
  "+task.dataset.dataset_mask_kwargs.curation_config_dir=${CURATION_DIR}" \
  "+task.dataset.dataset_mask_kwargs.curation_method=${METHOD}" \
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
