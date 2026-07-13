#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the remote CUDA server.
# Full reporting protocol for the checked tiered GT4/GT5 + GT4PV env30 run:
# train with official_best, sweep official_best.pt and best.pt on official-val,
# run reporting-only TEST ACC for both val-selected decodes, and run raw-Q12 diagnostics.
# TEST outputs are reporting-only and must not be used to select thresholds, checkpoints, or gains.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2}"
export RUN_TESTS="${RUN_TESTS:-1}"
export RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
export RUN_TEST_DIAGNOSTICS="${RUN_TEST_DIAGNOSTICS:-1}"

export SHORT_GEOM="${SHORT_GEOM:-1.0}"
export SHORT_GEOM_TIERED="${SHORT_GEOM_TIERED:-1}"
export SHORT_GEOM_GT4_ULTRA_VISIBLE_THR="${SHORT_GEOM_GT4_ULTRA_VISIBLE_THR:-10}"
export SHORT_GEOM_GT4_ULTRA_WEIGHT="${SHORT_GEOM_GT4_ULTRA_WEIGHT:-1.35}"
export SHORT_GEOM_GT4_MID_VISIBLE_THR="${SHORT_GEOM_GT4_MID_VISIBLE_THR:-20}"
export SHORT_GEOM_GT4_MID_WEIGHT="${SHORT_GEOM_GT4_MID_WEIGHT:-1.0}"
export SHORT_GEOM_GT5_ULTRA_VISIBLE_THR="${SHORT_GEOM_GT5_ULTRA_VISIBLE_THR:-10}"
export SHORT_GEOM_GT5_ULTRA_WEIGHT="${SHORT_GEOM_GT5_ULTRA_WEIGHT:-2.25}"
export SHORT_GEOM_GT5_MID_VISIBLE_THR="${SHORT_GEOM_GT5_MID_VISIBLE_THR:-20}"
export SHORT_GEOM_GT5_MID_WEIGHT="${SHORT_GEOM_GT5_MID_WEIGHT:-1.5}"
export SHORT_GEOM_MAX_WEIGHT="${SHORT_GEOM_MAX_WEIGHT:-3.0}"
export SHORT_GEOM_CURVE="${SHORT_GEOM_CURVE:-1.0}"
export GT4_SHORT_VISIBLE_THR="${GT4_SHORT_VISIBLE_THR:-10}"
export GT4_SHORT_POINT_VALID_WEIGHT="${GT4_SHORT_POINT_VALID_WEIGHT:-1.2}"
export GT5_SHORT_VISIBLE_THR="${GT5_SHORT_VISIBLE_THR:-10}"
export GT5_SHORT_POINT_VALID_WEIGHT="${GT5_SHORT_POINT_VALID_WEIGHT:-1.5}"

python - \
  "${RUN_TESTS}" "${RUN_DIAGNOSTICS}" "${RUN_TEST_DIAGNOSTICS}" \
  "${SHORT_GEOM}" "${SHORT_GEOM_TIERED}" "${SHORT_GEOM_MAX_WEIGHT}" "${SHORT_GEOM_CURVE}" \
  "${SHORT_GEOM_GT4_ULTRA_VISIBLE_THR}" "${SHORT_GEOM_GT4_ULTRA_WEIGHT}" \
  "${SHORT_GEOM_GT4_MID_VISIBLE_THR}" "${SHORT_GEOM_GT4_MID_WEIGHT}" \
  "${SHORT_GEOM_GT5_ULTRA_VISIBLE_THR}" "${SHORT_GEOM_GT5_ULTRA_WEIGHT}" \
  "${SHORT_GEOM_GT5_MID_VISIBLE_THR}" "${SHORT_GEOM_GT5_MID_WEIGHT}" \
  "${GT4_SHORT_VISIBLE_THR}" "${GT4_SHORT_POINT_VALID_WEIGHT}" \
  "${GT5_SHORT_VISIBLE_THR}" "${GT5_SHORT_POINT_VALID_WEIGHT}" <<'PY'
import sys

(
    run_tests,
    run_diagnostics,
    run_test_diagnostics,
    short_geom,
    short_geom_tiered,
    short_geom_max_weight,
    short_geom_curve,
    gt4_ultra_thr,
    gt4_ultra_weight,
    gt4_mid_thr,
    gt4_mid_weight,
    gt5_ultra_thr,
    gt5_ultra_weight,
    gt5_mid_thr,
    gt5_mid_weight,
    gt4_pv_thr,
    gt4_pv_weight,
    gt5_pv_thr,
    gt5_pv_weight,
) = sys.argv[1:]


def as_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be bool-like, got {value!r}.")


def as_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}.") from exc


def as_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a float, got {value!r}.") from exc


for name, value in (
    ("RUN_TESTS", run_tests),
    ("RUN_DIAGNOSTICS", run_diagnostics),
    ("RUN_TEST_DIAGNOSTICS", run_test_diagnostics),
    ("SHORT_GEOM_TIERED", short_geom_tiered),
):
    as_bool(name, value)

if not as_bool("SHORT_GEOM_TIERED", short_geom_tiered):
    raise SystemExit("SHORT_GEOM_TIERED must be enabled for this tiered full-protocol script.")

if as_float("SHORT_GEOM", short_geom) < 0.0:
    raise SystemExit("SHORT_GEOM must be >= 0.")
if as_float("SHORT_GEOM_MAX_WEIGHT", short_geom_max_weight) < 1.0:
    raise SystemExit("SHORT_GEOM_MAX_WEIGHT must be >= 1.")
if as_float("SHORT_GEOM_CURVE", short_geom_curve) < 0.0:
    raise SystemExit("SHORT_GEOM_CURVE must be >= 0.")

gt4_ultra_thr_i = as_int("SHORT_GEOM_GT4_ULTRA_VISIBLE_THR", gt4_ultra_thr)
gt4_mid_thr_i = as_int("SHORT_GEOM_GT4_MID_VISIBLE_THR", gt4_mid_thr)
gt5_ultra_thr_i = as_int("SHORT_GEOM_GT5_ULTRA_VISIBLE_THR", gt5_ultra_thr)
gt5_mid_thr_i = as_int("SHORT_GEOM_GT5_MID_VISIBLE_THR", gt5_mid_thr)
gt4_pv_thr_i = as_int("GT4_SHORT_VISIBLE_THR", gt4_pv_thr)
gt5_pv_thr_i = as_int("GT5_SHORT_VISIBLE_THR", gt5_pv_thr)

for name, value in (
    ("SHORT_GEOM_GT4_ULTRA_VISIBLE_THR", gt4_ultra_thr_i),
    ("SHORT_GEOM_GT4_MID_VISIBLE_THR", gt4_mid_thr_i),
    ("SHORT_GEOM_GT5_ULTRA_VISIBLE_THR", gt5_ultra_thr_i),
    ("SHORT_GEOM_GT5_MID_VISIBLE_THR", gt5_mid_thr_i),
    ("GT4_SHORT_VISIBLE_THR", gt4_pv_thr_i),
    ("GT5_SHORT_VISIBLE_THR", gt5_pv_thr_i),
):
    if value < 0:
        raise SystemExit(f"{name} must be >= 0.")

if gt4_mid_thr_i < gt4_ultra_thr_i:
    raise SystemExit("SHORT_GEOM_GT4_MID_VISIBLE_THR must be >= SHORT_GEOM_GT4_ULTRA_VISIBLE_THR.")
if gt5_mid_thr_i < gt5_ultra_thr_i:
    raise SystemExit("SHORT_GEOM_GT5_MID_VISIBLE_THR must be >= SHORT_GEOM_GT5_ULTRA_VISIBLE_THR.")

for name, value in (
    ("SHORT_GEOM_GT4_ULTRA_WEIGHT", gt4_ultra_weight),
    ("SHORT_GEOM_GT4_MID_WEIGHT", gt4_mid_weight),
    ("SHORT_GEOM_GT5_ULTRA_WEIGHT", gt5_ultra_weight),
    ("SHORT_GEOM_GT5_MID_WEIGHT", gt5_mid_weight),
    ("GT4_SHORT_POINT_VALID_WEIGHT", gt4_pv_weight),
    ("GT5_SHORT_POINT_VALID_WEIGHT", gt5_pv_weight),
):
    if as_float(name, value) < 1.0:
        raise SystemExit(f"{name} must be >= 1.0 for rescue-only behavior.")

print(
    "Parameter preflight ok: tiered GT4/GT5 geometry rescue, GT4/GT5 short point-valid rescue, "
    "and reporting-only TEST protocol are explicit."
)
PY

exec bash "${SCRIPT_DIR}/run_query_alpha05_gt4gt5weak_geom_w15w2_env30_nocount_v1.sh" "$@"
