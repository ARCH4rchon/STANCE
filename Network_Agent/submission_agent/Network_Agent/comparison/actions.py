"""Task E: what did the networks learn? Per-step actions against ground stiffness.

Drafted with Claude Code (Anthropic), for the AI declaration.

For every trained seed of each network, plays the standard evaluation episodes (seeds 1000-1199,
training ground, deterministic) and records, per step, the action the env actually applied
(after the residual offset and clipping), in physical units, plus the episode's ground base
stiffness k0. Outputs (in --out):
  actions.csv           controller, model_seed, episode_seed, step, ground_stiffness, theta_d, K, B
                        (step 0 = the first action of the episode)
  actions_time.png/.pdf mean theta_d, K, B over the episode in 4 stiffness bins (equal width in
                        log k0 over the training range 5e3-1e5); rows = networks; line = mean over
                        model seeds, band = sd over seeds. Late steps average only episodes still
                        running (falls end episodes early).
  actions_early.png/.pdf mean action over the first 20 steps vs k0, one dot per episode (colour =
                        model seed), black line = binned mean over all seeds.
  actions_slopes.csv    per seed: least-squares slope of the early mean action vs log10 k0, and
                        the early mean in the softest and hardest bin.

    python -m comparison.actions [--episodes 200] [--out DIR] [--plot-only]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from comparison.harness import (RESULTS, ctrl_label, default_workers, discover_controllers,
                                eval_seeds, load_network, log, write_csv)
from common.env_setup import make_env, to_physical

COLUMNS = ["controller", "model_seed", "episode_seed", "step", "ground_stiffness", "theta_d", "K", "B"]
OUTPUTS = [("theta_d", "theta_d (deg)"), ("K", "K (N.m/rad)"), ("B", "B (N.m.s/rad)")]
EARLY = 20
K_EDGES = np.logspace(np.log10(5e3), np.log10(1e5), 5)          # 4 bins over the training range
NET_LABEL = {"networkA_mlp": "MLP (A)", "networkB_lstm": "LSTM (B)"}
SEED_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def record(job):
    """Play one network seed over the seeds and return per-step rows."""
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    c = job["ctrl"]
    net, model, vn, wrappers, cfg = load_network(c["run"])
    env = make_env("train", wrappers)
    base = env.unwrapped
    policy = net.POLICY(model, vn)
    rows = []
    for s in job["seeds"]:
        obs, _ = env.reset(seed=int(s))
        policy.reset()
        k0 = float(base.ground.k0[0])
        step, done = 0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy.act(obs))
            th, K, B = to_physical(env, base.prev_action)            # the action actually applied
            rows.append({"controller": c["controller"], "model_seed": c["model_seed"],
                         "episode_seed": int(s), "step": step, "ground_stiffness": k0,
                         "theta_d": th, "K": K, "B": B})
            step += 1
            done = term or trunc
    env.close()
    write_csv(job["path"], rows, COLUMNS)
    return job["id"], time.time() - t0, len(rows)


def run(out, seeds, workers):
    jobs = []
    for c in discover_controllers():
        if c["kind"] == "network":
            cid = f"E_actions__{ctrl_label(c)}"
            jobs.append({"id": cid, "ctrl": c, "seeds": seeds,
                         "path": str(Path(out) / "cells" / f"{cid}.csv")})

    def done(j):
        p = Path(j["path"])
        return p.exists() and set(seeds) <= set(pd.read_csv(p, usecols=["episode_seed"]).episode_seed)

    todo = [j for j in jobs if not done(j)]
    log(f"{len(jobs)} cells, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    t0 = time.time()
    if todo:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
            for cid, dt, n in pool.imap_unordered(record, todo):
                log(f"  {cid}: {n} steps in {dt:.0f} s")
    df = pd.concat([pd.read_csv(j["path"]) for j in jobs], ignore_index=True)
    df = df[df.episode_seed.isin(seeds)]
    df.to_csv(Path(out) / "actions.csv", index=False, float_format="%.5g")
    log(f"actions.csv: {len(df)} rows ({time.time() - t0:.0f} s this run)")


def _bin_labels():
    return [f"{K_EDGES[i] / 1e3:.0f}-{K_EDGES[i + 1] / 1e3:.0f} kN/m" for i in range(len(K_EDGES) - 1)]


def plot_time(df, out, anchor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    df = df.assign(kbin=pd.cut(df.ground_stiffness, K_EDGES, labels=False, include_lowest=True))
    nets = list(df.controller.drop_duplicates())
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(K_EDGES) - 1))
    fig, axes = plt.subplots(len(nets), 3, figsize=(7.0, 2.2 * len(nets)), sharex=True, squeeze=False)
    for r, net in enumerate(nets):
        d = df[df.controller == net]
        for ci, (col, lab) in enumerate(OUTPUTS):
            ax = axes[r, ci]
            for b, colour in enumerate(colours):
                per_seed = (d[d.kbin == b].groupby(["model_seed", "step"])[col].mean()
                              .unstack("model_seed"))
                m, sd = per_seed.mean(axis=1), per_seed.std(axis=1, ddof=1)
                ax.plot(m.index, m.values, color=colour, lw=1.1,
                        label=_bin_labels()[b] if (r == 0 and ci == 0) else None)
                ax.fill_between(m.index, m - sd, m + sd, color=colour, alpha=0.18, lw=0)
            ax.axhline(anchor[ci], color="#777777", ls=":", lw=0.9,
                       label="residual anchor" if (r == 0 and ci == 0) else None)
            ax.set_title(f"{NET_LABEL.get(net, net)}: {lab}")
            ax.grid(alpha=0.3)
            if r == len(nets) - 1:
                ax.set_xlabel("step (5 ms)")
    fig.legend(loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               title="ground k0 bin (training range)", title_fontsize=6.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"actions_time.{ext}", dpi=300)
    plt.close(fig)


def early_means(df):
    e = df[df.step < EARLY]
    return (e.groupby(["controller", "model_seed", "episode_seed"])
              .agg(ground_stiffness=("ground_stiffness", "first"), theta_d=("theta_d", "mean"),
                   K=("K", "mean"), B=("B", "mean"))
              .reset_index())


def plot_early(em, out, anchor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    nets = list(em.controller.drop_duplicates())
    edges = np.logspace(np.log10(5e3), np.log10(1e5), 9)
    centres = np.sqrt(edges[:-1] * edges[1:])
    fig, axes = plt.subplots(len(nets), 3, figsize=(7.0, 2.2 * len(nets)), sharex=True, squeeze=False)
    for r, net in enumerate(nets):
        d = em[em.controller == net]
        for ci, (col, lab) in enumerate(OUTPUTS):
            ax = axes[r, ci]
            for si, (s, ds) in enumerate(d.groupby("model_seed")):
                ax.scatter(ds.ground_stiffness, ds[col], s=3, alpha=0.45, lw=0,
                           color=SEED_COLOURS[si % len(SEED_COLOURS)],
                           label=f"seed {s}" if ci == 0 else None)
            b = np.digitize(d.ground_stiffness, edges[1:-1])
            means = [d[col][b == i].mean() for i in range(len(centres))]
            ax.plot(centres, means, color="black", lw=1.4, marker="o", ms=2.5,
                    label="binned mean" if ci == 0 else None)
            ax.axhline(anchor[ci], color="#777777", ls=":", lw=0.9,
                       label="residual anchor" if ci == 0 else None)
            ax.set_xscale("log")
            ax.set_title(f"{NET_LABEL.get(net, net)}: {lab}")
            ax.grid(alpha=0.3)
            if r == len(nets) - 1:
                ax.set_xlabel("ground base stiffness k0 (N/m)")
        axes[r, 0].legend(loc="best", frameon=False, markerscale=2.5, handlelength=1.5)
    fig.suptitle(f"Mean applied action over steps 0-{EARLY - 1} of each episode "
                 "(one dot per episode, training ground)", fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"actions_early.{ext}", dpi=300)
    plt.close(fig)


def slopes(em):
    rows = []
    lo, hi = em.ground_stiffness <= K_EDGES[1], em.ground_stiffness > K_EDGES[-2]
    for (net, s), d in em.groupby(["controller", "model_seed"]):
        x = np.log10(d.ground_stiffness)
        row = {"controller": net, "model_seed": s, "n_episodes": len(d)}
        for col, _ in OUTPUTS:
            row[f"{col}_slope_per_decade"] = float(np.polyfit(x, d[col], 1)[0])
            row[f"{col}_corr"] = float(np.corrcoef(x, d[col])[0, 1])
            row[f"{col}_softest_bin"] = float(d[col][lo.loc[d.index]].mean())
            row[f"{col}_hardest_bin"] = float(d[col][hi.loc[d.index]].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def outputs(out):
    import json
    df = pd.read_csv(Path(out) / "actions.csv")
    run0 = next(c for c in discover_controllers() if c["kind"] == "network")["run"]
    anchor = json.loads((Path(run0) / "config.json").read_text())["config"]["anchor"]
    plot_time(df, out, anchor)
    em = early_means(df)
    plot_early(em, out, anchor)
    sl = slopes(em)
    sl.to_csv(Path(out) / "actions_slopes.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(sl.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if not a.plot_only:
        run(a.out, eval_seeds(a.episodes), a.workers)
    outputs(a.out)


if __name__ == "__main__":
    main()
