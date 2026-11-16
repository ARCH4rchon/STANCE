"""Task A: run every controller on the training ground and the test (out-of-range) ground.

Drafted with Claude Code (Anthropic), for the AI declaration.

Controllers: "reference" (-20, 500, 10), "best_fixed" (-20, 500, 6), and every model seed
found under trained_networks/<network>/final_s*/. Episodes are seeds 1000.. (first --episodes),
deterministic, played by common.evaluate.run_episodes. One row per episode goes to
<out>/episodes.csv. Finished cells are skipped on a rerun (resume).

    python -m comparison.compare                          200 episodes, results/episodes.csv
    python -m comparison.compare --episodes 20 --out comparison/results/smoke
    python -m comparison.stats                            summary.csv / summary.md from it
"""
import argparse
from pathlib import Path

from comparison.harness import (EP_COLUMNS, RESULTS, ctrl_label, default_workers,
                                discover_controllers, eval_seeds, log, merge_cells, run_jobs)

GROUNDS = ("train", "test")


def jobs_for(out, seeds):
    jobs = []
    for ground in GROUNDS:
        for c in discover_controllers():
            cid = f"A__{ground}__{ctrl_label(c)}"
            jobs.append({"id": cid, "ctrl": c, "ground": ground, "seeds": seeds,
                         "path": str(Path(out) / "cells" / f"{cid}.csv"), "columns": EP_COLUMNS})
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--workers", type=int, default=default_workers())
    a = p.parse_args()
    seeds = eval_seeds(a.episodes)
    ctrls = discover_controllers()
    log("controllers: " + ", ".join(ctrl_label(c) for c in ctrls))
    jobs = jobs_for(a.out, seeds)
    dt = run_jobs(jobs, a.workers)
    rows = merge_cells(jobs, Path(a.out) / "episodes.csv", EP_COLUMNS)
    log(f"Task A done: {len(rows)} episode rows -> {Path(a.out) / 'episodes.csv'} ({dt:.0f} s this run)")


if __name__ == "__main__":
    main()
