#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-omnireset_isaac}"
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f "${ROOT}/environments/omnireset.yaml"
fi
conda run --no-capture-output -n "${ENV_NAME}" python -m pip install \
  -r "${ROOT}/environments/isaacsim.requirements.txt"
conda run --no-capture-output -n "${ENV_NAME}" python -m pip install --no-deps \
  -e "${ROOT}" -e "${ROOT}/third_party/rsl_rl"
conda run --no-capture-output -n "${ENV_NAME}" python -m pip check
