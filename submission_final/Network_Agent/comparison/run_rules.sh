#!/usr/bin/env bash
# Drafted with Claude Code (Anthropic), for the AI declaration.
# Hand-rule study (sensing -> tune/evaluate R1-R3 -> summary -> stiffness sweep), nice 10,
# appending to comparison/results/log.txt. Resumable.
# Usage (from Network_Agent/):  bash comparison/run_rules.sh [python]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
LOG=comparison/results/log.txt
run() { env -u PYTHONPATH nice -n 10 "$PY" -m "$@" 2>&1 | tee -a "$LOG"; }
stamp() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
stamp "run_rules start ($(nproc) cpus)"
stamp "Step 1: sensing";                     run comparison.sensing --episodes 100
stamp "Step 2+3: tune and evaluate rules";   run comparison.rules --episodes 100 --eval-episodes 200
stamp "Step 3: summary";                     run comparison.stats
stamp "Step 3: stiffness sweep with rules";  run comparison.robustness --episodes 100
stamp "run_rules done"
