#!/usr/bin/env bash
# Drafted with Claude Code (Anthropic), for the AI declaration.
# Tasks C, D, E at low priority, appending to comparison/results/log.txt. Resumable.
# Usage (from Network_Agent/):  bash comparison/run_cde.sh [python]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
LOG=comparison/results/log.txt
run() { env -u PYTHONPATH nice -n 10 "$PY" -m "$@" 2>&1 | tee -a "$LOG"; }
stamp() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
stamp "run_cde start ($(nproc) cpus)"
stamp "Task C: grid + best_fixed_tuned";   run comparison.tune --episodes 100 --eval-episodes 200
stamp "Task C: summary with tuned row";    run comparison.stats
stamp "Task D: noise check";               run comparison.noise_check
stamp "Task D: noise sweep 0-2.0";         run comparison.robustness --episodes 100
stamp "Task E: actions";                   run comparison.actions --episodes 200
stamp "run_cde done"
