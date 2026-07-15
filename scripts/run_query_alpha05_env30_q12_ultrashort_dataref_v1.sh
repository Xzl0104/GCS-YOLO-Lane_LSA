#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the remote CUDA server.
# Q12-dataref-ultrashort-v1: reference coverage first, then training only after the static gate passes.

RUN_NAME="${RUN_NAME:-query_alpha05_env30_q12_ultrashort_dataref_v1}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-ultrashort-dataref.yaml}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}"
PRETRAINED="${PRETRAINED:-yolo11s-seg.pt}"
BANK="${BANK:-data/gcs_reference_banks/q12_ultrashort_env30_train_v1.json}"
REFERENCE_AUDIT="${REFERENCE_AUDIT:-data/gcs_reference_banks/q12_ultrashort_env30_train_v1_val_reference_audit.json}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-220}"
WORKERS="${WORKERS:-4}"
OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"
OFFICIAL_WARMUP="${OFFICIAL_WARMUP:-5}"
HALF="${HALF:-1}"
ENFORCE_REFERENCE_GATE="${ENFORCE_REFERENCE_GATE:-1}"

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-ssh_lane}"
fi

if [[ ! -f "${BANK}" ]]; then
  echo "Missing reference bank: ${BANK}" >&2
  echo "Build it with tools/build_gcs_ultrashort_reference_bank.py before training." >&2
  exit 2
fi

if is_true "${ENFORCE_REFERENCE_GATE}"; then
  if [[ ! -f "${REFERENCE_AUDIT}" ]]; then
    echo "Missing static reference audit: ${REFERENCE_AUDIT}" >&2
    echo "Run tools/check_gcs_reference_bank_coverage.py and pass the static gate before training." >&2
    exit 2
  fi
  python - "${REFERENCE_AUDIT}" "${BANK}" "${GT_JSON}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
bank_path = Path(sys.argv[2])
expected_gt_json = str(sys.argv[3]).replace("\\", "/")
with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
actual_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
recorded_hash = data.get("bank_sha256", "")
if recorded_hash != actual_hash:
    raise SystemExit(
        f"Static reference audit does not match bank. audit_hash={recorded_hash!r} bank_hash={actual_hash!r}"
    )
if data.get("schema") != "gcs_q12_fixed_y_reference_bank_coverage_audit_v1":
    raise SystemExit(f"Unexpected static audit schema: {data.get('schema')!r}")
if data.get("bank_schema") != "gcs_q12_fixed_y_reference_bank_v1":
    raise SystemExit(f"Unexpected bank schema recorded in audit: {data.get('bank_schema')!r}")
gt_json = [str(p).replace("\\", "/") for p in data.get("gt_json", [])]
if gt_json != [expected_gt_json]:
    raise SystemExit(f"Static audit GT mismatch. expected={[expected_gt_json]!r} actual={gt_json!r}")
gate = data.get("gate", {})
if not bool(gate.get("pass", False)):
    raise SystemExit(f"Static reference gate failed in {path}: {gate}")
print(f"Static reference gate passed: {path}")
PY
else
  echo "ENFORCE_REFERENCE_GATE=0: bypassing static reference gate. Use only for debugging." >&2
fi

OFFICIAL_HALF_ARGS=()
if is_true "${HALF}"; then
  OFFICIAL_HALF_ARGS=(--gcs-official-half)
fi

python tools/train_gcs.py \
  --dataset tusimple \
  --model "${MODEL}" \
  --data "${DATA}" \
  --pretrained "${PRETRAINED}" \
  --imgsz 544 960 \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --scale 0.15 \
  --erasing 0.10 \
  --mosaic 0.0 \
  --gcs-exist-quality-alpha 0.5 \
  --gcs-count 0.0 \
  --gcs-count-under5 0.0 \
  --gcs-count-boundary 0.0 \
  --gcs-query-count-ce 0.0 \
  --gcs-short-geom 0.0 \
  --gcs-gt4-short-point-valid-weight 1.0 \
  --gcs-gt5-short-point-valid-weight 1.0 \
  --gcs-boundary-pseudo-neg 0.02 \
  --gcs-boundary-pseudo-visible-thr 10 \
  --gcs-boundary-pseudo-dist-thr 80 \
  --gcs-boundary-pseudo-valid-thr 0.5 \
  --gcs-boundary-pseudo-min-valid 4 \
  --gcs-boundary-pseudo-gt-count 5 \
  --gcs-boundary-pseudo-score-thr 0.2 \
  --gcs-boundary-pseudo-envelope-margin-px 30 \
  --gcs-boundary-pseudo-envelope-ratio-thr 0.75 \
  --gcs-official-best \
  --gcs-official-valid-before-maxdet \
  --gcs-official-interval "${OFFICIAL_INTERVAL}" \
  --gcs-official-archive-root "${ARCHIVE_ROOT}" \
  --gcs-official-gt-json "${GT_JSON}" \
  --gcs-official-warmup "${OFFICIAL_WARMUP}" \
  "${OFFICIAL_HALF_ARGS[@]}" \
  --project "${PROJECT}" \
  --name "${RUN_NAME}"
