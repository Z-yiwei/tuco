#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: bash experiments/single_sim/prepare.sh CONFIG.env}"
[[ -f "${CONFIG}" ]] || { printf 'missing config: %s\n' "${CONFIG}" >&2; exit 1; }
CONFIG="$(readlink -f -- "${CONFIG}")"
set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a

PY="${PY:-python}"
TRAK_PY="${TRAK_PY:-${PY}}"
CUPID_ROOT="${CUPID_ROOT:-${ROOT}/third_party/cupid}"
export PYTHONPATH="${ROOT}/src:${CUPID_ROOT}:${CUPID_ROOT}/third_party/trak${PYTHONPATH:+:${PYTHONPATH}}"
: "${TASK_MH:?set TASK_MH in ${CONFIG}}"
: "${SPLIT:?set SPLIT to filter or select in ${CONFIG}}"
: "${PREP_ROOT:?set PREP_ROOT in ${CONFIG}}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
protocol() {
  PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PY}" -m tuco.cli.protocol single_sim robomimic "$1"
}
NUM_EPOCHS="${NUM_EPOCHS:-$(protocol "base_epochs.${SPLIT}.${TASK_MH}")}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-$(protocol protocol.rollouts)}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-$(protocol protocol.checkpoint_every)}"
if [[ "${SPLIT}" == filter ]]; then
  TRAIN_RATIO="${TRAIN_RATIO:-$(protocol filtering.train_fraction)}"
else
  TRAIN_RATIO="${TRAIN_RATIO:-$(protocol selection.anchor_fraction)}"
fi
VAL_RATIO="${VAL_RATIO:-0.04}"
TRAK_PROJECTION_DIM="${TRAK_PROJECTION_DIM:-$(protocol protocol.trak_projection_dim)}"
TRAK_LOSS="${TRAK_LOSS:-$(protocol protocol.trak_loss)}"
TRAK_NUM_TIMESTEPS="${TRAK_NUM_TIMESTEPS:-$(protocol protocol.diffusion_samples_per_transition)}"
TRAK_RIDGE="${TRAK_RIDGE:-$(protocol protocol.trak_ridge)}"
TRAK_BATCH_SIZE="${TRAK_BATCH_SIZE:-32}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
TRAIN_DIR="${PREP_ROOT}/base_policy"
EVAL_DIR="${PREP_ROOT}/rollouts"
TRAK_NAME="trak-proj${TRAK_PROJECTION_DIM}-seed${SEED}"
TRAK_DIR="${EVAL_DIR}/${TRAK_NAME}"
INPUT_DIR="${PREP_ROOT}/inputs_${SPLIT}"

[[ "${SPLIT}" == filter || "${SPLIT}" == select ]] || {
  printf 'SPLIT must be filter or select\n' >&2; exit 2;
}
mkdir -p "${PREP_ROOT}"
cd "${CUPID_ROOT}"

if [[ ! -f "${TRAIN_DIR}/checkpoints/latest.ckpt" ]]; then
  train_overrides=()
  [[ "${MAX_TRAIN_STEPS}" == 0 ]] || \
    train_overrides+=("training.max_train_steps=${MAX_TRAIN_STEPS}")
  [[ -z "${DATASET_PATH:-}" ]] || \
    train_overrides+=("task.dataset.dataset_path=${DATASET_PATH}")
  "${PY}" train.py \
    --config-dir="configs/low_dim/${TASK_MH}/diffusion_policy_cnn" \
    --config-name=config.yaml \
    "hydra.run.dir=${TRAIN_DIR}" "training.seed=${SEED}" \
    "training.num_epochs=${NUM_EPOCHS}" "training.checkpoint_every=${CHECKPOINT_EVERY}" \
    "training.rollout_every=${CHECKPOINT_EVERY}" +training.rollout_last_n_checkpoints=0 \
    "training.device=${DEVICE}" \
    "task.dataset.seed=${SEED}" "task.dataset.val_ratio=${VAL_RATIO}" \
    "+task.dataset.dataset_mask_kwargs.train_ratio=${TRAIN_RATIO}" \
    +task.dataset.dataset_mask_kwargs.uniform_quality=true \
    task.env_runner.n_test_vis=0 task.env_runner.n_train_vis=0 \
    checkpoint.topk.k=5 training.resume=false logging.mode=disabled \
    "${train_overrides[@]}"
fi

if [[ ! -f "${EVAL_DIR}/episodes/metadata.yaml" ]]; then
  "${PY}" eval_save_episodes.py \
    --output_dir="${EVAL_DIR}" --train_dir="${TRAIN_DIR}" --train_ckpt=latest \
    --num_episodes="${NUM_ROLLOUTS}" --test_start_seed=100000 \
    --overwrite=0 "--device=${DEVICE}"
fi

if [[ ! -f "${TRAK_DIR}/scores/all_episodes.mmap" ]]; then
  "${TRAK_PY}" train_trak_diffusion.py \
    --exp_name="${TRAK_NAME}" --eval_dir="${EVAL_DIR}" \
    --train_dir="${TRAIN_DIR}" --train_ckpt=latest --model_id=0 \
    --model_keys=model. --modelout_fn=DiffusionLowdimFunctionalModelOutput \
    --gradient_co=DiffusionLowdimFunctionalGradientComputer \
    --proj_dim="${TRAK_PROJECTION_DIM}" --proj_max_batch_size=32 \
    --lambda_reg="${TRAK_RIDGE}" --use_half_precision=0 --loss_fn="${TRAK_LOSS}" \
    --num_timesteps="${TRAK_NUM_TIMESTEPS}" --batch_size="${TRAK_BATCH_SIZE}" \
    "--device=${DEVICE}" --seed="${SEED}" \
    --featurize_holdout=true --finalize_scores=true
fi

if [[ ! -f "${INPUT_DIR}/manifest.json" ]]; then
  "${TRAK_PY}" "${ROOT}/experiments/single_sim/prepare_trak_inputs.py" \
    --checkpoint "${TRAIN_DIR}/checkpoints/latest.ckpt" \
    --eval-dir "${EVAL_DIR}" --trak-dir "${TRAK_DIR}" \
    --split "${SPLIT}" --output "${INPUT_DIR}" --device cpu
fi
printf 'prepared single-simulator inputs: %s\n' "${INPUT_DIR}"
