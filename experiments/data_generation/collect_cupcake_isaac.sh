#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
COLLECTOR="${ROOT}/experiments/data_generation/collect_state_demos.py"
: "${OMNIRESET_ROOT:?set OMNIRESET_ROOT}"
: "${PY_ISAAC:?set PY_ISAAC}"
: "${CUPCAKE_TEACHER:?set CUPCAKE_TEACHER}"
: "${RESET_DATASET_DIR:?set RESET_DATASET_DIR}"
: "${OUTPUT:?set OUTPUT}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-42}"
SHARDS="${OUTPUT}.shards"
LOGS="${OUTPUT}.logs"
IFS=',' read -r -a gpu_ids <<<"${GPUS}"
(( ${#gpu_ids[@]} > 0 ))
mkdir -p "${SHARDS}" "${LOGS}"

pids=(); paths=()
base=$((5000 / ${#gpu_ids[@]})); remainder=$((5000 % ${#gpu_ids[@]}))
for rank in "${!gpu_ids[@]}"; do
  count="${base}"; (( rank >= remainder )) || count=$((count + 1))
  shard="${SHARDS}/rank${rank}.zarr"; paths+=("${shard}")
  CUDA_VISIBLE_DEVICES="${gpu_ids[$rank]}" OMNI_KIT_ACCEPT_EULA=YES \
  OMNIRESET_ROOT="${OMNIRESET_ROOT}" \
  UWLAB_LOCAL_ASSETS_DIR="${UWLAB_LOCAL_ASSETS_DIR:?set UWLAB_LOCAL_ASSETS_DIR}" \
    "${PY_ISAAC}" -u "${COLLECTOR}" \
      --task OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-Finetune-Play-v0 \
      --checkpoint "${CUPCAKE_TEACHER}" --reset_type CupCakeSideLyingFront3cmFixedHome \
      --dataset_dir "${RESET_DATASET_DIR}" --num_envs 256 --num_demos "${count}" \
      --max_steps 200000 --output "${shard}" --seed "$((SEED + rank))" \
      --expected_obs_dim 200 --expected_act_dim 7 --expected_episode_steps 160 \
      --expected_receptive_usd_basename plate.usd --keep_dynamics_dr \
      --headless --device cuda:0 env.scene.insertive_object=cupcake \
      env.scene.receptive_object=plate env.observations.policy.enable_corruption=False \
      env.scene.robot.actuators.panda_hand.stiffness=1000.0 \
      env.scene.robot.actuators.panda_hand.damping=14.0 \
      env.scene.robot.actuators.panda_hand.effort_limit_sim=60.0 \
      >"${LOGS}/rank${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 ))
"${PY_ISAAC}" "${ROOT}/third_party/omnireset_eval/scripts/cupcake/merge_isaac_zarr_shards.py" \
  --output "${OUTPUT}" --expected-demos 5000 --delete-shards "${paths[@]}"
