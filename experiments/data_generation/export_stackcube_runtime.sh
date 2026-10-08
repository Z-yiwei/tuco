#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -Eeuo pipefail

if (( $# != 6 )); then
  printf 'usage: %s OMNIRESET_ROOT PY TEACHER RESET_PT MANIFEST_ZARR OUTPUT_ZARR\n' "$0" >&2
  exit 2
fi
OMNI="$(readlink -f -- "$1")"
PY="$2"
TEACHER="$3"
RESET="$4"
MANIFEST="$5"
OUTPUT="$6"
GPU="${GPU:-0}"

pythonpath=""
for package_root in "${OMNI}"/UWLab/_isaaclab/IsaacLab/source/* "${OMNI}"/UWLab/source/*; do
  [[ -d "${package_root}" ]] || continue
  pythonpath="${pythonpath:+${pythonpath}:}${package_root}"
done

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONHASHSEED=42 \
PYTHONPATH="${pythonpath}${PYTHONPATH:+:${PYTHONPATH}}" \
SIM2SIM_COTRAIN_ROOT="${OMNI}" OMNI_KIT_ACCEPT_EULA=YES \
UWLAB_LOCAL_ASSETS_DIR="${UWLAB_LOCAL_ASSETS_DIR:?set UWLAB_LOCAL_ASSETS_DIR}" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${PY}" -u "${OMNI}/UWLab/scripts/reinforcement_learning/rsl_rl/export_stackcube_cotrain_pairs.py" \
    --checkpoint "${TEACHER}" --reset_state "${RESET}" \
    --reset_indices_zarr "${MANIFEST}" --out "${OUTPUT}" \
    --episode_steps 160 --preserve_controller_events \
    --source_policy_type rsl_rl --seed 42 --device cuda:0 --headless \
    env.scene.insertive_object=cube env.scene.receptive_object=cube \
    env.scene.robot.actuators.panda_hand.stiffness=1000.0 \
    env.scene.robot.actuators.panda_hand.damping=14.0 \
    env.scene.robot.actuators.panda_hand.effort_limit_sim=60.0
