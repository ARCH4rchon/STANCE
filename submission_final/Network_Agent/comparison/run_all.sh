#!/usr/bin/env bash
# Drafted with Claude Code (Anthropic), for the AI declaration.
# Full comparison run at low priority, logging to comparison/results/log.txt.
# Safe to rerun after an interruption: finished cells are skipped (resume).
# Usage (from Network_Agent/):  bash comparison/run_all.sh [python]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
LOG=comparison/results/log.txt
mkdir -p comparison/results
run() { env -u PYTHONPATH nice -n 10 "$PY" -m "$@" 2>&1 | tee -a "$LOG"; }
echo "=== $(date '+%F %T') run_all start ($(nproc) cpus)" >> "$LOG"
run comparison.compare --episodes 200
run comparison.stats
run comparison.robustness --episodes 100
echo "=== $(date '+%F %T') run_all done" >> "$LOG"
