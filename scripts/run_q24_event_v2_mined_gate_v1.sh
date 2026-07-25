#!/usr/bin/env bash
set -euo pipefail

# Event-mined gate wrapper for the Q24 event-v2 dynamic score probe.
# Keeps TEST closed and only relabels query roles for diagnostics.

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_q24_event_v2_dynamic_score_probe40_v1}"
export GT5_QUERIES="${GT5_QUERIES:-13,15}"
export GT4_QUERIES="${GT4_QUERIES:-14,17,18,19}"
export WATCH_QUERIES="${WATCH_QUERIES:-0,1,11,16,20,21,22,23}"

bash scripts/run_q24_event_mined_gate_v1.sh
