#!/usr/bin/env bash
set -euo pipefail

# Raw/refined geometry gate for the v2 auxiliary short-local-refine probe.

export RUN_NAME="${RUN_NAME:-query_short_local_refine_env30_aux_v2_probe20}"

bash scripts/run_query_short_local_refine_env30_gate_v1.sh
