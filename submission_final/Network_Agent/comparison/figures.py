"""Task G: report figures and the cost table, from the CSVs already in results/.

Drafted with Claude Code (Anthropic), for the AI declaration.

One style for all figures: half-page width (3.5 in), 7 pt text, Okabe-Ito colours (colour-blind
safe), one colour per controller everywhere; line style by family: fixed actions dotted, hand
rules dashed, networks solid. Networks are the mean over model seeds with a band (or dots) for
the seed spread. R3 uses the true ground stiffness and is drawn hatched and labelled privileged.

  fig_main   paired Δ return and Δ fall rate vs the reference, 95% bootstrap CIs, train/test
             (source: summary.csv; network bars = seed-averaged per episode, dots = single seeds)
  fig_sweep  return vs ground stiffness (robustness.csv), training range marked on the x axis
  fig_kdip   mean K over steps 0-60 in the softest / hardest training-stiffness bins (actions.csv),
             MLP and LSTM mean ± seed sd, R1's K profile as a step
  cost_table.md  numbers read from the repo (config.json, curve.csv, model.zip, log.txt,
             tuning CSVs); "n/a" where the repo does not record it

    python -m comparison.figures [--out DIR]
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from comparison.harness import RESULTS, TRAINED

COL = {"reference": "#000000", "best_fixed_tuned": "#E69F00", "R1_timed_dip": "#009E73",
       "R2_sensed_dip": "#999999", "R3_oracle_dip": "#56B4E9", "R4_dip_plus_schedule": "#CC79A7",
       "networkA_mlp": "#0072B2", "networkB_lstm": "#D55E00"}
NAME = {"reference": "fixed ref. (-20, 500, 10)", "best_fixed_tuned": "best fixed (tuned)",
        "R1_timed_dip": "R1 timed dip", "R2_sensed_dip": "R2 sensed dip",
        "R3_oracle_dip": "R3 oracle dip (privileged)", "R4_dip_plus_schedule": "R4 dip + schedule",
        "networkA_mlp": "MLP (A)", "networkB_lstm": "LSTM (B)"}
MARK = {"reference": "s", "best_fixed_tuned": "^", "R1_timed_dip": "v", "R2_sensed_dip": "P",
        "R3_oracle_dip": "X", "R4_dip_plus_schedule": "*", "networkA_mlp": "o", "networkB_lstm": "D"}
TRAIN_K = (5e3, 1e5)
W = 3.5


def ls(c):
    return "-" if c.startswith("network") else ("--" if c.startswith("R") else ":")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                         "hatch.linewidth": 0.6})
    return plt


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"{name}.{ext}", dpi=300)


# ---------------------------------------------------------------- fig_main
def fig_main(out):
    plt = _plt()
    s = pd.read_csv(Path(out) / "summary.csv")
    ctrls = ["reference", "best_fixed_tuned", "R1_timed_dip", "R3_oracle_dip", "networkA_mlp", "networkB_lstm"]
    fig, axes = plt.subplots(2, 2, figsize=(W, 3.3), sharey=True)
    y = np.arange(len(ctrls))[::-1]
    for ci, ground in enumerate(("train", "test")):
        g = s[s.ground == ground]
        for ri, (m, scale, xl) in enumerate((("d_return", 1, "Δ return"),
                                             ("d_fall_rate", 100, "Δ fall rate (pts)"))):
            ax = axes[ri, ci]
            ax.axvline(0, color="#000000", lw=0.6)
            for yi, c in zip(y, ctrls):
                r = g[(g.controller == c) & g.row.isin(["fixed", "mean over seeds"])]
                if c == "reference" or r.empty:
                    ax.plot(0, yi, marker=MARK[c], color=COL[c], ms=3.5, ls="none")
                    continue
                r = r.iloc[0]
                v, lo, hi = r[m] * scale, r[f"{m}_lo"] * scale, r[f"{m}_hi"] * scale
                ax.barh(yi, v, height=0.62, color=COL[c] if c != "R3_oracle_dip" else "white",
                        edgecolor=COL[c], hatch="////" if c == "R3_oracle_dip" else None, lw=0.8)
                ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="#333333",
                            elinewidth=0.8, capsize=1.8)
                seeds = g[(g.controller == c) & (g.row == "seed")]
                if len(seeds):
                    ax.plot(seeds[m] * scale, np.full(len(seeds), yi), "o", ms=2.0, mfc="white",
                            mec="#333333", mew=0.5, ls="none", zorder=5)
            ax.grid(axis="x", alpha=0.3)
            ax.set_xlabel(xl)
            if ri == 0:
                ax.set_title("training ground" if ground == "train" else "test ground")
    axes[0, 0].set_yticks(y, [NAME[c] for c in ctrls])
    axes[1, 0].set_yticks(y, [NAME[c] for c in ctrls])
    fig.text(0.99, 0.005, "Δ vs fixed reference, 95% bootstrap CI; dots: single network seeds",
             ha="right", va="bottom", fontsize=5.5, color="#444444")
    fig.tight_layout(pad=0.3, rect=(0, 0.02, 1, 1))
    _save(fig, out, "fig_main")
    plt.close(fig)


# ---------------------------------------------------------------- fig_sweep
def fig_sweep(out):
    plt = _plt()
    s = pd.read_csv(Path(out) / "robustness.csv")
    s = s[s.sweep == "stiffness"]
    ctrls = [c for c in ("reference", "R1_timed_dip", "R3_oracle_dip", "R4_dip_plus_schedule",
                         "networkA_mlp", "networkB_lstm") if c in set(s.controller)]
    fig, ax = plt.subplots(figsize=(W, 3.1))
    ax.axvspan(*TRAIN_K, color="#DDDDDD", alpha=0.6, lw=0, zorder=0)
    for c in ctrls:
        r = s[(s.controller == c) & s.row.isin(["fixed", "rule", "mean over seeds"])].sort_values("level")
        ax.plot(r.level, r.mean_return, color=COL[c], ls=ls(c), marker=MARK[c], ms=3, lw=1.1,
                label=NAME[c], zorder=3)
        if (r.row == "mean over seeds").any():
            sd = r.mean_return_sd.fillna(0)
            ax.fill_between(r.level, r.mean_return - sd, r.mean_return + sd, color=COL[c], alpha=0.18,
                            lw=0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("ground base stiffness k0 (N/m)")
    ax.set_ylabel("mean episode return")
    ax.grid(alpha=0.3)
    ymin = ax.get_ylim()[0]
    # training range on the x axis: a bar just above the axis plus a label
    ax.annotate("", xy=(TRAIN_K[0], 0), xytext=(TRAIN_K[1], 0), xycoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="|-|", lw=0.9, color="#444444", shrinkA=0, shrinkB=0,
                                mutation_scale=3))
    ax.text(np.sqrt(TRAIN_K[0] * TRAIN_K[1]), 0.03, "training range", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6, color="#444444")
    ax.set_ylim(ymin, 101)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False,
              handlelength=2.4, columnspacing=0.9)
    fig.tight_layout(pad=0.3)
    _save(fig, out, "fig_sweep")
    plt.close(fig)


# ---------------------------------------------------------------- fig_kdip
def fig_kdip(out):
    from comparison.actions import K_EDGES
    plt = _plt()
    df = pd.read_csv(Path(out) / "actions.csv", usecols=["controller", "model_seed", "episode_seed",
                                                         "step", "ground_stiffness", "K"])
    df = df[df.step <= 60]
    df["kbin"] = pd.cut(df.ground_stiffness, K_EDGES, labels=False, include_lowest=True)
    r1 = json.loads((Path(out) / "rules_tuned.json").read_text())["timed"]["rule"]
    steps = np.arange(0, 61)
    k_r1 = np.where((steps * 5 >= r1["start_ms"]) & (steps * 5 < r1["end_ms"]), r1["k_dip"], 500.0)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    for ax, b, title in ((axes[0], 0, "softest bin"), (axes[1], len(K_EDGES) - 2, "hardest bin")):
        for c in ("networkA_mlp", "networkB_lstm"):
            d = df[(df.controller == c) & (df.kbin == b)]
            per = d.groupby(["model_seed", "step"]).K.mean().unstack("model_seed")
            m, sd = per.mean(axis=1), per.std(axis=1, ddof=1)
            ax.plot(m.index * 5, m.values, color=COL[c], ls="-", lw=1.1, label=NAME[c])
            ax.fill_between(m.index * 5, m - sd, m + sd, color=COL[c], alpha=0.18, lw=0)
        ax.step(steps * 5, k_r1, where="post", color=COL["R1_timed_dip"], ls="--", lw=1.1,
                label=f"R1 timed dip (K {r1['k_dip']:g}, {r1['start_ms']:g}-{r1['end_ms']:g} ms)")
        lo, hi = K_EDGES[b], K_EDGES[b + 1]
        ax.set_title(f"{title} (k0 {lo / 1e3:.0f}-{hi / 1e3:.0f} kN/m)")
        ax.set_xlabel("time after contact (ms)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 300)
    axes[0].set_ylabel("K (N.m/rad)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6, handlelength=2.2,
               columnspacing=0.8)
    fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
    _save(fig, out, "fig_kdip")
    plt.close(fig)


# ---------------------------------------------------------------- cost table
def _log_times(out):
    """Timestamps of the last occurrence of each marker in log.txt."""
    marks = {}
    pat = re.compile(r"^(?:=== )?(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
    path = Path(out) / "log.txt"
    for line in (path.read_text().splitlines() if path.exists() else []):
        m = pat.match(line)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            for key, needle in (("grid_done", "grid: 80 cells"), ("rules_ref", "reference on tuning seeds"),
                                ("r1", "R1 chosen"), ("r2", "R2_sensed_dip chosen"), ("r3", "R3_oracle_dip chosen"),
                                ("r4_start", "Task F: tune and evaluate R4"), ("r4", "R4 chosen"),
                                ("grid_start", "Task C: grid")):
                if needle in m.group(2):
                    marks[key] = (t, m.group(2))
    return marks


def _dur(marks, a, b):
    if a in marks and b in marks:
        return f"{(marks[b][0] - marks[a][0]).total_seconds() / 60:.1f} min wall (14 workers)"
    return "n/a"


def cost_table(out):
    out = Path(out)
    marks = _log_times(out)
    grid_s = re.search(r"\((\d+) s this run\)", marks.get("grid_done", (None, ""))[1])
    tr = pd.read_csv(out / "tuning_rules.csv")
    n = lambda rule: int(tr[tr.rule == rule].n_episodes.sum())
    nc = lambda rule: int((tr.rule == rule).sum())
    r4 = pd.read_csv(out / "tuning_r4.csv") if (out / "tuning_r4.csv").exists() else None
    grid = pd.read_csv(out / "grid.csv")
    nets = {}
    for net in ("networkA_mlp", "networkB_lstm"):
        runs = sorted(TRAINED.glob(f"{net}/final_s*"))
        cfg = json.loads((runs[0] / "config.json").read_text())["config"]
        mins = [float(pd.read_csv(r / "curve.csv").minutes.iloc[-1]) for r in runs]
        from comparison.harness import load_network
        _, model, _, _, _ = load_network(str(runs[0]))
        ps = list(model.policy.named_parameters())
        tot = sum(p.numel() for _, p in ps)
        crit = sum(p.numel() for nm, p in ps if any(k in nm for k in ("value", "vf", "critic")))
        nets[net] = dict(steps=cfg["total_steps"], seeds=len(runs), mins=mins, tot=tot, actor=tot - crit,
                         frames=cfg["frames"], drop=cfg["drop_sensors"])
    rows = [
        ("fixed ref. (-20, 500, 10)", "3 (θ, K, B)",
         "n/a (chosen from an earlier 45-action grid whose output was not saved; P3_findings §1)",
         "n/a", "none", "no", "yes"),
        ("best fixed (tuned) (-20, 500, 8)", "3 (θ, K, B)",
         f"{len(grid)} cells × {int(grid.n_episodes.iloc[0])} = {int(grid.n_episodes.sum()):,} tuning episodes",
         f"{int(grid_s.group(1))} s wall (14 workers)" if grid_s else "n/a", "none", "no", "yes"),
        ("R1 timed dip", "3 (K_dip, start, end)",
         f"{nc('R1_timed_dip')} cells, {n('R1_timed_dip'):,} tuning episodes",
         _dur(marks, "rules_ref", "r1"), "none (time since contact only)", "yes", "yes"),
        ("R2 sensed dip", "3 (a, b, m) + R1 window",
         f"{nc('R2_sensed_dip')} cells, {n('R2_sensed_dip'):,} (+ R1's for the window)",
         _dur(marks, "r1", "r2"), "vertical GRF (ch 3)", "yes", "yes"),
        ("R3 oracle dip (privileged)", "2 (a, b) + R1 window",
         f"{nc('R3_oracle_dip')} cells, {n('R3_oracle_dip'):,} (+ R1's for the window)",
         _dur(marks, "r2", "r3"), "none; TRUE ground k0 (privileged)", "yes", "yes"),
    ]
    if r4 is not None:
        rows.append(("R4 dip + schedule", "8 (θ_start, T_θ, B_start, B_end, T_B, K_dip, start, end)",
                     f"{len(r4)} cells, {int(r4.n_episodes.sum()):,} (+ R1's for the initial dip)",
                     _dur(marks, "r4_start", "r4"), "none (time since contact only)", "yes", "yes"))
    for net, label in (("networkA_mlp", "MLP (A)"), ("networkB_lstm", "LSTM (B)")):
        d = nets[net]
        sens = f"all 11 channels × {d['frames']} frame{'s' if d['frames'] > 1 else ''}"
        sens += " (8 sensors + previous action)" + ("" if not d["drop"] else f", minus {d['drop']}")
        rows.append((label, f"{d['tot']:,} weights ({d['actor']:,} actor + {d['tot'] - d['actor']:,} critic)",
                     f"{d['steps']:,} env steps per seed × {d['seeds']} seeds; episodes n/a (not logged)",
                     " / ".join(f"{m:.1f}" for m in d["mins"]) + " min per seed (curve.csv, incl. curve evals)",
                     sens, "implicit only (no clock input)", "no"))
    md = ["# Cost table", "",
          "Drafted with Claude Code (Anthropic), for the AI declaration. Generated by "
          "`python -m comparison.figures`; numbers read from the repo (trained_networks/*/config.json, "
          "curve.csv, model.zip; comparison/results/grid.csv, tuning_rules.csv, tuning_r4.csv, log.txt). "
          "\"n/a\" = not recorded in the repo.", "",
          "| controller | parameters | training / tuning data | wall-clock | sensors used | uses contact timing | interpretable |",
          "|---|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "Notes: tuning episodes are on seeds 2000-2099 (training ground), 100 per cell; rule "
           "wall-clock times are from log.txt timestamps on a 16-CPU machine with 14 worker processes at "
           "nice 10. Network times are the `minutes` column of each run's curve.csv at its last row "
           "(single process, 8 vectorised envs), which includes the periodic 50-episode curve "
           "evaluations. Episodes start at heel contact, so rules use time since episode start as "
           "contact timing; the networks get no clock input (the MLP's 10-frame history starts with zero "
           "padding and the LSTM resets at contact, so timing is only implicit).", ""]
    (out / "cost_table.md").write_text("\n".join(md))
    print("\n".join(md))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(RESULTS))
    a = p.parse_args()
    fig_main(a.out)
    fig_sweep(a.out)
    fig_kdip(a.out)
    cost_table(a.out)


if __name__ == "__main__":
    main()
