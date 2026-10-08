#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${1:?usage: bash scripts/setup_environment.sh cupid|omnireset}"

case "${KIND}" in
  cupid)
    ENV_NAME=cupid
    ENV_FILE="${ROOT}/environments/cupid.yaml"
    RUNTIME_FILE="${ROOT}/environments/cupid-runtime.txt"
    ;;
  omnireset)
    ENV_NAME=omnireset_release
    ENV_FILE="${ROOT}/environments/omnireset.yaml"
    RUNTIME_FILE="${ROOT}/environments/omnireset-runtime.txt"
    ;;
  *) printf 'environment must be cupid or omnireset\n' >&2; exit 2 ;;
esac

command -v conda >/dev/null || { printf 'conda is required\n' >&2; exit 1; }
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env create -f "${ENV_FILE}"
fi

run=(conda run --no-capture-output -n "${ENV_NAME}")
if [[ "${KIND}" == omnireset ]]; then
  # Install the observed dependency closure without upgrading the numerical stack.
  "${run[@]}" python -m pip install --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "${ROOT}/environments/omnireset.requirements.txt"
else
  "${run[@]}" python -m pip install -r "${RUNTIME_FILE}"
fi
"${run[@]}" python -m pip install "pytest==8.4.2" "tomli==2.2.1"
"${run[@]}" python -m pip install --no-deps \
  -e "${ROOT}/third_party/cupid/third_party/trak" \
  -e "${ROOT}/third_party/cupid" \
  -e "${ROOT}"

torch_cuda="$("${run[@]}" python -c 'import torch; print(torch.version.cuda or "")')"
cuda_home="/usr/local/cuda-${torch_cuda}"
if [[ -n "${torch_cuda}" && -x "${cuda_home}/bin/nvcc" ]] && \
   command -v nvidia-smi >/dev/null; then
  CUDA_HOME="${cuda_home}" PATH="${cuda_home}/bin:${PATH}" \
    "${run[@]}" python -m pip install --no-deps --no-build-isolation \
      "${ROOT}/third_party/cupid/third_party/trak/fast_jl"
else
  printf 'warning: matching CUDA toolkit %s is unavailable; CUDA TRAK attribution requires fast_jl before use\n' \
    "${torch_cuda:-none}" >&2
fi

"${run[@]}" python - <<'PY'
import torch
import tuco
print(f"environment ready: torch={torch.__version__} tuco={tuco.__file__}")
PY
