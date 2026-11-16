#!/usr/bin/env bash
# Drafted with Claude Code (Anthropic), for the AI declaration.
# Tasks F and G at low priority, appending to comparison/results/log.txt. Resumable.
# Usage (from Network_Agent/):  bash comparison/run_fg.sh [python]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
LOG=comparison/results/log.txt
run() { env -u PYTHONPATH nice -n 10 "$PY" -m "$@" 2>&1 | tee -a "$LOG"; }
stamp() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
stamp "run_fg start ($(nproc) cpus)"
stamp "Task F: tune and evaluate R4";   run comparison.rules --r4 --episodes 100 --eval-episodes 200
stamp "Task F: summary";                run comparison.stats
stamp "Task F: stiffness sweep";        run comparison.robustness --episodes 100
stamp "Task G: figures and cost table"; run comparison.figures
stamp "run_fg done"
