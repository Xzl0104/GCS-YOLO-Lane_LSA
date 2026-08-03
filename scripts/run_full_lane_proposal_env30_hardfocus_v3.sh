#!/usr/bin/env bash
set -euo pipefail

# Full-lane proposal v3 hard-miss focused warmup.
# This is diagnostic-only: TEST stays closed and full-lane decode stays off.
# It reuses the v2 aux-warmup trainer with hard-focus target weighting enabled.

export RUN_NAME="${RUN_NAME:-full_lane_proposal_env30_hardfocus_v3}"
export EPOCHS="${EPOCHS:-20}"
export BATCH="${BATCH:-16}"
export WORKERS="${WORKERS:-4}"
export RUN_TESTS="${RUN_TESTS:-0}"
export FULL_LANE_DECODE="${FULL_LANE_DECODE:-0}"
export FULL_LANE_HARD_FOCUS="${FULL_LANE_HARD_FOCUS:-1}"

# Keep the base path frozen and train the full head toward the lanes that can
# actually add ACC: base-miss, GT4/GT5, and visible-short lanes.
export FULL_LANE_BASE_HIT_WEIGHT="${FULL_LANE_BASE_HIT_WEIGHT:-0.05}"
export FULL_LANE_BASE_MISS_WEIGHT="${FULL_LANE_BASE_MISS_WEIGHT:-4.0}"
export FULL_LANE_GT4_WEIGHT="${FULL_LANE_GT4_WEIGHT:-2.0}"
export FULL_LANE_GT5_WEIGHT="${FULL_LANE_GT5_WEIGHT:-3.0}"
export FULL_LANE_SHORT_VISIBLE_THR="${FULL_LANE_SHORT_VISIBLE_THR:-10}"
export FULL_LANE_SHORT_VISIBLE_WEIGHT="${FULL_LANE_SHORT_VISIBLE_WEIGHT:-4.0}"
export FULL_LANE_FOCUS_HIT_PX="${FULL_LANE_FOCUS_HIT_PX:-20.0}"
export FULL_LANE_FOCUS_BASE_VALID_THR="${FULL_LANE_FOCUS_BASE_VALID_THR:-0.6}"
export FULL_LANE_FOCUS_BASE_MIN_COVERAGE="${FULL_LANE_FOCUS_BASE_MIN_COVERAGE:-1.0}"

export FULL_LANE_UNMATCHED_WEIGHT="${FULL_LANE_UNMATCHED_WEIGHT:-0.0}"
export FULL_LANE_UNMATCHED_VALID_WEIGHT="${FULL_LANE_UNMATCHED_VALID_WEIGHT:-0.0}"
export FULL_LANE_AUX_MATCH_MIN_OVERLAP="${FULL_LANE_AUX_MATCH_MIN_OVERLAP:-2}"
export FULL_LANE_AUX_MATCH_GATE_PX="${FULL_LANE_AUX_MATCH_GATE_PX:-0.0}"

# Base decode official hooks do not evaluate full-lane proposal usefulness.
# Keep a final base sanity check only; hard oracle diagnostics after training
# are the gate for this diagnostic experiment.
export OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-99}"
export OFFICIAL_WARMUP="${OFFICIAL_WARMUP:-5}"
export ORACLE_WARMUP="${ORACLE_WARMUP:-20}"

bash scripts/run_full_lane_proposal_env30_auxwarm_v2.sh
