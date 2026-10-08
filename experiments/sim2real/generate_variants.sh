#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SELECTION:?set SELECTION}"
: "${VARIANTS_ROOT:?set VARIANTS_ROOT}"
PY_ISAAC="${PY_ISAAC:-$(conda run -n omnireset_isaac python -c 'import sys; print(sys.executable)')}"

for variant in 6 7 8 9 10; do
  destination="${VARIANTS_ROOT}/variant${variant}"
  if [[ -f "${destination}/completion.json" && -d "${destination}/chunks/states.zarr" ]]; then
    if "${PY_ISAAC}" -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("returncode")==0 and d.get("episodes")==600 else 1)' "${destination}/completion.json"; then
      continue
    fi
  fi
  "${PY_ISAAC}" "${ROOT}/tools/replay_expert.py" \
    --task stackcube --artifacts "${ARTIFACT_ROOT:-${ROOT}/artifacts}" \
    --output "${destination}" --gpu "${GPU:-0}" \
    --state-ids "${SELECTION}" --topup-variant "${variant}"
done
