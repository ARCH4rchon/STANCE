"""Runs every controller on the evaluation episodes (both grounds) and on the ground-stiffness sweep."""

import argparse
from pathlib import Path

import pandas as pd

from common.evaluate import EVAL_SEEDS
from comparison.harness import RESULTS, controllers, run_all

# Ground stiffness levels in N/m: from well below to well above the training range (5e3 to 1e5).
K_LEVELS = [1e3, 3.2e3, 7e3, 2.2e4, 7e4, 1.4e5, 3e5, 6e5]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--sweep-episodes", type=int, default=100)
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--workers", type=int)
    a = p.parse_args()

    ctrls = controllers()
    evaluation = [(c, g, EVAL_SEEDS[:a.episodes], None) for g in ("train", "test") for c in ctrls]
    sweep = [(c, "train", EVAL_SEEDS[:a.sweep_episodes], k) for c in ctrls for k in K_LEVELS]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_all(evaluation, a.workers)).drop(columns="level").to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(run_all(sweep, a.workers)).drop(columns="ground").to_csv(out / "sweep.csv", index=False)