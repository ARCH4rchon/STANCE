"""Paired comparisons of the controllers on the same episodes, printed to the terminal."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparison.harness import RESULTS

CTRLS = ["reference", "best_fixed_tuned", "R1_timed_dip", "R3_oracle_dip", "networkA_mlp", "networkB_lstm"]
PAIRS = [("R3_oracle_dip", "R1_timed_dip"), ("R1_timed_dip", "networkA_mlp"), ("R1_timed_dip", "networkB_lstm"),
         ("R3_oracle_dip", "networkA_mlp"), ("R3_oracle_dip", "networkB_lstm")]


def per_episode(df, ground, ctrl):
    """Return and fall (0/1) per episode. Networks are averaged over their model seeds."""
    d = df[(df.ground == ground) & (df.controller == ctrl)]
    return d.groupby("episode_seed")[["return", "fell"]].mean()


def boot_ci(d, n_boot=10_000, seed=5418):
    """Mean and 95% bootstrap interval of d (fixed seed, so it reproduces)."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired(a, b):
    """Mean difference a - b with its interval, on the episodes both share (same seed = same ground)."""
    common = a.index.intersection(b.index)
    d = a.loc[common] - b.loc[common]
    return {col: boot_ci(d[col]) for col in ("return", "fell")}


def show(res):
    r, f = res["return"], res["fell"]
    return f"return {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}]   fall {100 * f[0]:+.1f} [{100 * f[1]:+.1f}, {100 * f[2]:+.1f}] pts"


if __name__ == "__main__":
    # Parts of this specific section are written by Claude (Anthropic)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(RESULTS))
    a = p.parse_args()
    df = pd.read_csv(Path(a.out) / "episodes.csv")
    for ground in ("train", "test"):
        ref = per_episode(df, ground, "reference")
        print(f"\n{ground} ground: mean return, fall rate, and difference vs the reference (95% bootstrap CI)")
        for c in CTRLS:
            e = per_episode(df, ground, c)
            print(f"  {c:20s} {e['return'].mean():6.2f}  {100 * e.fell.mean():5.1f}%" +
                  ("" if c == "reference" else "   " + show(paired(e, ref))))
        print("  direct comparisons (first minus second):")
        for x, y in PAIRS:
            print(f"    {x} - {y}: {show(paired(per_episode(df, ground, x), per_episode(df, ground, y)))}")

    sweep = pd.read_csv(Path(a.out) / "sweep.csv")
    per_seed = sweep.groupby(["controller", "model_seed", "level"], dropna=False)[["return", "fell"]].mean()
    table = per_seed.groupby(["controller", "level"])[["return", "fell"]].mean()
    levels = sorted(sweep.level.unique())
    print("\nstiffness sweep, mean return (fall %) at each k0 in N/m")
    print(f"  {'':20s}" + "".join(f"{k:>12g}" for k in levels))
    for c in CTRLS:
        print(f"  {c:20s}" + "".join(f"{table.loc[(c, k), 'return']:7.1f} ({100 * table.loc[(c, k), 'fell']:.0f})" for k in levels))