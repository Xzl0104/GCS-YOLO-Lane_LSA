#!/usr/bin/env bash
set -euo pipefail

# Official-val/train-side diagnostics for the Q24 dual-head bneg0 probe.
# This uses GT only for diagnostics and must not run TEST.

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_q24_dualhead_quality_count_bneg0_probe100_v1}"
export PROJECT="${PROJECT:-runs/gcs_lane}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OVERWRITE_DIAGS="${OVERWRITE_DIAGS:-0}"

RUN_DIR="${PROJECT}/${RUN_NAME}"
ARGS_YAML="${ARGS_YAML:-${RUN_DIR}/args.yaml}"

echo "Q24 dual-head bneg0 diagnostic gate: run=${RUN_NAME}, tests=${RUN_TESTS}"
echo "TEST remains closed; this wraps scripts/run_q24_event_mined_gate_v1.sh."

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-ssh_lane}"
fi

if [[ ! -f "${ARGS_YAML}" ]]; then
  echo "Missing training args for contract check: ${ARGS_YAML}" >&2
  exit 2
fi

python - "${ARGS_YAML}" <<'PY'
import math
import sys

import yaml

args_path = sys.argv[1]
with open(args_path, "r", encoding="utf-8") as f:
    args = yaml.safe_load(f) or {}

expected_zero = [
    "gcs_boundary_pseudo_neg",
    "gcs_count",
    "gcs_count_under5",
    "gcs_count_boundary",
    "gcs_role_contain",
    "gcs_q24_event_contain",
    "gcs_q24_event_score_calib",
]
bad = []
for key in expected_zero:
    value = args.get(key)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        bad.append(f"{key}={value!r}")
        continue
    if not math.isclose(numeric, 0.0, abs_tol=1e-12):
        bad.append(f"{key}={value!r}")

model = str(args.get("model", ""))
if not model.endswith("gcs-yolo-lane-s-q24-k56-dualhead.yaml"):
    bad.append(f"model={model!r}")

if bad:
    raise SystemExit("BNEG0 contract check failed: " + ", ".join(bad))

print(
    "BNEG0 contract check passed: boundary pseudo, score-sum count losses, "
    "role/event containment disabled; Q24 dual-head model confirmed."
)
PY

bash scripts/run_q24_event_mined_gate_v1.sh
