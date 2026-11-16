"""Task C: establish the best fixed action with a grid search on tuning seeds.

Drafted with Claude Code (Anthropic), for the AI declaration.

1. Grid of fixed actions (theta_d deg, K N.m/rad, B N.m.s/rad) on the training ground, tuning
   seeds 2000-2099 (100 episodes per cell). These seeds are disjoint from the evaluation seeds
   1000-1199, so the chosen action is not tuned on the seeds it is scored on.
   Requested grid: theta {-26,-20,-14,-8}, K {250,375,500,625}, B {4,6,8,10,12}. The env's valid
   ranges are theta [-20, 20] deg, K [5, 500], B [0.5, 10] (Gym/stance_env/ankle_gym_env.py);
   out-of-range values would be clipped onto an edge and duplicate another cell, so each one is
   replaced by an in-range value:  -26 -> -17,  625 -> 440,  12 -> 2.
       theta {-20, -17, -14, -8}  x  K {250, 375, 440, 500}  x  B {2, 4, 6, 8, 10}  = 80 cells
2. Writes results/grid.csv and results/grid_heatmap.png/.pdf (one panel per B).
3. Best cell = highest mean return among cells whose fall rate is not above the reference
   (-20, 500, 10) on the same tuning seeds (the reference is a grid cell). It is saved to
   results/best_fixed_tuned.json and evaluated as "best_fixed_tuned" on seeds 1000-1199, train
   and test ground, into results/episodes_tuned.csv (episodes.csv is not touched). Then
   `python -m comparison.stats` adds its row to summary.md.

    python -m comparison.tune [--episodes 100] [--eval-episodes 200] [--out DIR]
    python -m comparison.tune --plot-only
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparison.harness import (EP_COLUMNS, RESULTS, default_workers, eval_seeds, log,
                                merge_cells, run_jobs)
from common.evaluate import CURVE_SEEDS, REFERENCE

THETAS = [-20.0, -17.0, -14.0, -8.0]
KS = [250.0, 375.0, 440.0, 500.0]
BS = [2.0, 4.0, 6.0, 8.0, 10.0]
TUNE_SEEDS_ALL = list(range(2000, 2100))
GRID_COLUMNS = ["theta_d", "K", "B"] + EP_COLUMNS


def tag(a):
    return "%g_%g_%g" % a


def grid_jobs(out, seeds):
    jobs = []
    for a in itertools.product(THETAS, KS, BS):
        ctrl = {"controller": f"fixed({tag(a)})", "model_seed": "", "kind": "fixed", "action": a}
        cid = f"C_grid__{tag(a)}"
        jobs.append({"id": cid, "ctrl": ctrl, "ground": "train", "seeds": seeds,
                     "tags": {"theta_d": a[0], "K": a[1], "B": a[2]},
                     "path": str(Path(out) / "cells" / f"{cid}.csv"), "columns": GRID_COLUMNS})
    return jobs


def grid_table(rows):
    df = pd.DataFrame(rows)
    for c in ("theta_d", "K", "B", "return", "fell", "toe_peak_bw"):
        df[c] = df[c].astype(float)
    g = (df.groupby(["theta_d", "K", "B"])
           .agg(n_episodes=("return", "size"), mean_return=("return", "mean"),
                return_sd=("return", "std"), fall_rate=("fell", "mean"),
                toe_peak_bw=("toe_peak_bw", "mean"))
           .reset_index())
    ref = g[(g.theta_d == REFERENCE[0]) & (g.K == REFERENCE[1]) & (g.B == REFERENCE[2])].iloc[0]
    g["is_reference"] = (g.theta_d == REFERENCE[0]) & (g.K == REFERENCE[1]) & (g.B == REFERENCE[2])
    g["eligible"] = g.fall_rate <= ref.fall_rate + 1e-12
    g["rank_return"] = g.mean_return.rank(ascending=False, method="min").astype(int)
    best = g[g.eligible].sort_values("mean_return", ascending=False).iloc[0]
    g["is_best"] = (g.theta_d == best.theta_d) & (g.K == best.K) & (g.B == best.B)
    return g, ref, best


def plot(g, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    vmin, vmax = g.mean_return.min(), g.mean_return.max()
    fig, axes = plt.subplots(1, len(BS), figsize=(7.2, 2.3), sharey=True, constrained_layout=True)
    for ax, b in zip(axes, BS):
        sub = g[g.B == b]
        m = np.full((len(THETAS), len(KS)), np.nan)
        f = np.full_like(m, np.nan)
        for _, r in sub.iterrows():
            i, j = THETAS.index(r.theta_d), KS.index(r.K)
            m[i, j], f[i, j] = r.mean_return, r.fall_rate
        im = ax.imshow(m, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        for i, j in itertools.product(range(len(THETAS)), range(len(KS))):
            light = (m[i, j] - vmin) / (vmax - vmin + 1e-9) < 0.55
            ax.text(j, i, f"{m[i, j]:.1f}\n{100 * f[i, j]:.0f}%", ha="center", va="center",
                    fontsize=5.5, color="white" if light else "black")
        for _, r in sub[sub.is_best | sub.is_reference].iterrows():
            i, j = THETAS.index(r.theta_d), KS.index(r.K)
            ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, lw=1.4,
                                   ls="-" if r.is_best else "--",
                                   ec="#D55E00" if r.is_best else "#FFFFFF"))
        ax.set_title(f"B = {b:g}")
        ax.set_xticks(range(len(KS)), [f"{k:g}" for k in KS])
        ax.set_yticks(range(len(THETAS)), [f"{t:g}" for t in THETAS])
        ax.set_xlabel("K (N.m/rad)")
    axes[0].set_ylabel("theta_d (deg)")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label("mean return (tuning seeds 2000+)")
    fig.suptitle("Fixed-action grid, training ground. Cell text: mean return / fall rate. "
                 "Solid orange box: best_fixed_tuned; dashed white: reference (-20, 500, 10).",
                 fontsize=6.5)
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"grid_heatmap.{ext}", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=100, help="tuning episodes per grid cell")
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    if a.plot_only:
        plot(pd.read_csv(out / "grid.csv"), out)
        return
    assert set(TUNE_SEEDS_ALL).isdisjoint(range(1000, 1200)) and TUNE_SEEDS_ALL[:50] == CURVE_SEEDS
    tune_seeds = TUNE_SEEDS_ALL[: a.episodes]
    jobs = grid_jobs(out, tune_seeds)
    dt = run_jobs(jobs, a.workers)
    log(f"grid: {len(jobs)} cells x {len(tune_seeds)} episodes ({dt:.0f} s this run)")
    rows, keep = [], set(tune_seeds)
    for j in jobs:
        rows += [r for r in pd.read_csv(j["path"]).to_dict("records") if r["episode_seed"] in keep]
    g, ref, best = grid_table(rows)
    g.to_csv(out / "grid.csv", index=False, float_format="%.5g")
    plot(g, out)
    action = (float(best.theta_d), float(best.K), float(best.B))
    info = {"action": action, "tuning_seeds": [tune_seeds[0], tune_seeds[-1]],
            "tuning_mean_return": float(best.mean_return), "tuning_fall_rate": float(best.fall_rate),
            "reference_tuning_mean_return": float(ref.mean_return),
            "reference_tuning_fall_rate": float(ref.fall_rate),
            "rule": "max mean return among grid cells with fall rate <= reference fall rate"}
    (out / "best_fixed_tuned.json").write_text(json.dumps(info, indent=2))
    log(f"reference on tuning seeds: return {ref.mean_return:.2f}, fall {100 * ref.fall_rate:.1f}%")
    log(f"best_fixed_tuned = {action}: return {best.mean_return:.2f}, fall {100 * best.fall_rate:.1f}% "
        f"(overall rank {int(best.rank_return)} of {len(g)})")
    top = g.sort_values("mean_return", ascending=False).head(8)
    print(top[["theta_d", "K", "B", "mean_return", "fall_rate", "toe_peak_bw", "eligible"]].to_string(index=False))

    # Evaluate the chosen action with the standard protocol, into a separate file.
    seeds = eval_seeds(a.eval_episodes)
    ctrl = {"controller": "best_fixed_tuned", "model_seed": "", "kind": "fixed", "action": action}
    ev = []
    for ground in ("train", "test"):
        cid = f"C_eval__{ground}__best_fixed_tuned_{tag(action)}"
        ev.append({"id": cid, "ctrl": ctrl, "ground": ground, "seeds": seeds,
                   "path": str(out / "cells" / f"{cid}.csv"), "columns": EP_COLUMNS})
    dt = run_jobs(ev, a.workers)
    merged = merge_cells(ev, out / "episodes_tuned.csv", EP_COLUMNS)
    log(f"best_fixed_tuned evaluated: {len(merged)} rows -> episodes_tuned.csv ({dt:.0f} s)")


if __name__ == "__main__":
    main()
