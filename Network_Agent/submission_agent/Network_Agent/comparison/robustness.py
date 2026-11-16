"""Task B: robustness sweeps (evaluation only, no training).

Drafted with Claude Code (Anthropic), for the AI declaration.

1. Ground stiffness. The base stiffness k0 is set to one fixed value per level after each
   reset (harness.GroundStiffness); every other ground parameter (damping ratio, stiffening,
   yield force, friction) and the start pose keep their seeded draws from the TRAINING ranges,
   so only k0 moves. Ranges in Gym/stance_env/ground.py: train k0 5e3-1e5 N/m (log-uniform);
   test k0 2e3-5e3 or 1e5-2e5 N/m. Levels (N/m):
       1e3    beyond the test soft band (2x softer than its minimum)
       3.2e3  inside the test soft band
       7e3    training range, near the soft end
       2.2e4  training range, log-centre
       7e4    training range, near the hard end
       1.4e5  inside the test hard band
       3e5    beyond the test hard band (1.5x its maximum)
       6e5    far beyond (3x the test maximum)
2. Sensor noise on the training ground: Gaussian noise on sensor channels 0-7 at
   0, 0.05, 0.1, 0.2 x each sensor's std (harness.SensorNoise; std from the trained runs'
   VecNormalize statistics, one vector shared by all networks). Only the networks read the
   sensors; the fixed actions are unaffected by construction, so they are not rerun.
Both use 100 episodes per cell (seeds 1000-1099, or the first --episodes), deterministic.

Outputs (in --out): robustness_episodes.csv (per episode), robustness.csv (per cell and
mean ± sd over model seeds), robustness.png/.pdf (return vs stiffness).

    python -m comparison.robustness                 full run (resumes finished cells)
    python -m comparison.robustness --episodes 20 --out comparison/results/smoke
    python -m comparison.robustness --plot-only
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparison.harness import (EP_COLUMNS, RESULTS, ctrl_label, default_workers,
                                discover_controllers, eval_seeds, log, merge_cells, run_jobs,
                                sensor_std)

K_LEVELS = [1e3, 3.2e3, 7e3, 2.2e4, 7e4, 1.4e5, 3e5, 6e5]
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2]
TRAIN_K = (5e3, 1e5)
TEST_K = [(2e3, 5e3), (1e5, 2e5)]
COLUMNS = ["sweep", "level"] + EP_COLUMNS
LABELS = {"reference": "fixed (-20, 500, 10)", "best_fixed": "fixed (-20, 500, 6)",
          "networkA_mlp": "MLP (A)", "networkB_lstm": "LSTM (B)"}
# Okabe-Ito colours (colour-blind safe), plus distinct markers and line styles.
STYLE = {"reference": dict(color="#000000", marker="s", ls="--"),
         "best_fixed": dict(color="#E69F00", marker="^", ls=":"),
         "networkA_mlp": dict(color="#0072B2", marker="o", ls="-"),
         "networkB_lstm": dict(color="#D55E00", marker="D", ls="-.")}


def build_jobs(out, seeds, std):
    jobs = []
    for c in discover_controllers():
        for k in K_LEVELS:
            cid = f"B_stiff__{k:g}__{ctrl_label(c)}"
            jobs.append({"id": cid, "ctrl": c, "ground": "train", "seeds": seeds, "k0": k,
                         "tags": {"sweep": "stiffness", "level": k},
                         "path": str(Path(out) / "cells" / f"{cid}.csv"), "columns": COLUMNS})
        if c["kind"] != "network":
            continue
        for n in NOISE_LEVELS:
            cid = f"B_noise__{n:g}__{ctrl_label(c)}"
            jobs.append({"id": cid, "ctrl": c, "ground": "train", "seeds": seeds, "noise": n,
                         "std": std, "tags": {"sweep": "noise", "level": n},
                         "path": str(Path(out) / "cells" / f"{cid}.csv"), "columns": COLUMNS})
    return jobs


def aggregate(df):
    """Per-cell rows, then for each network the mean and sd (ddof=1) of the per-seed means."""
    df = df.copy()
    df["model_seed"] = df["model_seed"].astype("Int64")
    keys = ["sweep", "level", "controller"]
    cell = (df.groupby(keys + ["model_seed"], dropna=False, sort=False)
              .agg(n_episodes=("return", "size"), mean_return=("return", "mean"),
                   fall_rate=("fell", "mean"), toe_peak_bw=("toe_peak_bw", "mean"),
                   impact_sum=("impact_sum", "mean"))
              .reset_index())
    cell["row"] = np.where(cell.model_seed.isna(), "fixed", "seed")
    nets = cell[cell.row == "seed"]
    agg = (nets.groupby(keys, sort=False)
               .agg(n_seeds=("model_seed", "nunique"), n_episodes=("n_episodes", "sum"),
                    mean_return=("mean_return", "mean"), mean_return_sd=("mean_return", "std"),
                    fall_rate=("fall_rate", "mean"), fall_rate_sd=("fall_rate", "std"),
                    toe_peak_bw=("toe_peak_bw", "mean"), toe_peak_bw_sd=("toe_peak_bw", "std"),
                    impact_sum=("impact_sum", "mean"))
               .reset_index())
    agg["row"] = "mean over seeds"
    out = pd.concat([cell, agg], ignore_index=True)
    cols = ["sweep", "level", "controller", "row", "model_seed", "n_seeds", "n_episodes",
            "mean_return", "mean_return_sd", "fall_rate", "fall_rate_sd",
            "toe_peak_bw", "toe_peak_bw_sd", "impact_sum"]
    order = {"stiffness": 0, "noise": 1}
    out = out.reindex(columns=cols)
    out["_o"] = out.sweep.map(order)
    return out.sort_values(["_o", "controller", "row", "model_seed", "level"]).drop(columns="_o")


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    s = summary[summary.sweep == "stiffness"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axvspan(*TRAIN_K, color="#009E73", alpha=0.12, lw=0, label="training k0 range")
    for i, band in enumerate(TEST_K):
        ax.axvspan(*band, facecolor="none", edgecolor="#BBBBBB", hatch="///", lw=0,
                   label="test k0 bands" if i == 0 else None)
    for ctrl in [c for c in STYLE if c in set(s.controller)]:
        st = STYLE[ctrl]
        c = s[s.controller == ctrl]
        r = c[c.row.isin(["fixed", "mean over seeds"])].sort_values("level")
        ax.plot(r.level, r.mean_return, color=st["color"], marker=st["marker"], ls=st["ls"],
                ms=3.5, lw=1.2, label=LABELS.get(ctrl, ctrl))
        if (r.row == "mean over seeds").any():
            sd = r.mean_return_sd.fillna(0)
            ax.fill_between(r.level, r.mean_return - sd, r.mean_return + sd,
                            color=st["color"], alpha=0.2, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("ground base stiffness k0 (N/m)")
    ax.set_ylabel("mean episode return")
    ax.grid(alpha=0.3, which="major")
    ax.legend(loc="lower center", frameon=False, ncol=2, handlelength=2.2, columnspacing=1.0)
    lo = s.mean_return.min() - (s.mean_return_sd.max() if s.mean_return_sd.notna().any() else 0)
    ax.set_ylim(min(lo, s.mean_return.min()) - 12, 101)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"robustness.{ext}", dpi=300)
    plt.close(fig)


def write_outputs(out):
    df = pd.read_csv(Path(out) / "robustness_episodes.csv")
    summary = aggregate(df)
    summary.to_csv(Path(out) / "robustness.csv", index=False, float_format="%.5g")
    plot(summary, out)
    m = summary[summary.row != "seed"]
    pd.set_option("display.width", 160)
    print(m[["sweep", "level", "controller", "mean_return", "mean_return_sd", "fall_rate",
             "toe_peak_bw"]].to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if not a.plot_only:
        seeds = eval_seeds(a.episodes)
        std = sensor_std()
        log("sensor std used for noise (channels 0-7): " + ", ".join(f"{x:.4g}" for x in std[:8]))
        jobs = build_jobs(a.out, seeds, std)
        dt = run_jobs(jobs, a.workers)
        rows = merge_cells(jobs, Path(a.out) / "robustness_episodes.csv", COLUMNS)
        log(f"Task B done: {len(rows)} episode rows ({dt:.0f} s this run)")
    write_outputs(a.out)


if __name__ == "__main__":
    main()
