"""Task A statistics: summary.csv and summary.md from episodes.csv.

Drafted with Claude Code (Anthropic), for the AI declaration.

For each ground and controller:
  - mean return, fall rate, mean toe peak (BW)
  - networks: one row per model seed, then a "mean over seeds" row with the mean and the std
    (ddof=1) of the per-seed means
  - paired differences vs "reference" on the same episode seeds (return, fall rate), with a
    95% percentile bootstrap interval over episodes (10,000 resamples, RNG seed 5418)
For the "mean over seeds" row the paired difference is taken per episode on the seed-averaged
value, so its interval covers episode-to-episode variation only; the seed spread is the std.

    python -m comparison.stats [--in comparison/results/episodes.csv] [--out comparison/results]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparison.harness import RESULTS

N_BOOT = 10_000
BOOT_SEED = 5418
REF = "reference"
ORDER = ["reference", "best_fixed", "best_fixed_tuned", "R1_timed_dip", "R2_sensed_dip", "R3_oracle_dip",
         "R4_dip_plus_schedule"]
RULES = ["R1_timed_dip", "R2_sensed_dip", "R3_oracle_dip", "R4_dip_plus_schedule"]


def vs_r1(df):
    """Paired differences of R2/R3/R4 vs R1 on the same episodes."""
    rows = []
    for ground, g in df.groupby("ground", sort=False):
        if "R1_timed_dip" not in set(g.controller):
            continue
        r1 = g[g.controller == "R1_timed_dip"].set_index("episode_seed")[["return", "fell"]]
        for rule in [r for r in RULES[1:] if r in set(g.controller)]:
            rr = g[g.controller == rule].set_index("episode_seed")[["return", "fell"]]
            rows.append({"ground": ground, "rule": rule, "vs": "R1_timed_dip", **paired(rr, r1)})
    return pd.DataFrame(rows)


def label(c):
    return c + " (privileged)" if "oracle" in c else c


def vs_networks(df):
    """Paired differences of each hand rule vs each network (per episode, seed-averaged network)."""
    rows = []
    for ground, g in df.groupby("ground", sort=False):
        nets = [c for c in g.controller.drop_duplicates() if g[g.controller == c].model_seed.notna().any()]
        for rule in [r for r in RULES if r in set(g.controller)]:
            rr = g[g.controller == rule].set_index("episode_seed")[["return", "fell"]]
            for net in nets:
                avg = g[g.controller == net].groupby("episode_seed")[["return", "fell"]].mean()
                rows.append({"ground": ground, "rule": rule, "network": net, **paired(rr, avg)})
    return pd.DataFrame(rows)


def boot_ci(d):
    """Mean of d and its 95% percentile bootstrap interval (fixed seed, so reproducible)."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(per_ep, ref):
    """per_ep, ref: DataFrames indexed by episode_seed with return/fell. Paired diffs vs ref."""
    common = per_ep.index.intersection(ref.index)
    out = {"n_paired": len(common)}
    for col, name in (("return", "d_return"), ("fell", "d_fall_rate")):
        m, lo, hi = boot_ci(per_ep.loc[common, col] - ref.loc[common, col])
        out.update({name: m, f"{name}_lo": lo, f"{name}_hi": hi})
    return out


def summarise(df):
    rows = []
    for ground, g in df.groupby("ground", sort=False):
        ref = g[g.controller == REF].set_index("episode_seed")[["return", "fell"]]
        for ctrl, c in g.groupby("controller", sort=False):
            seeds = sorted(c.model_seed.dropna().unique())
            base = {"ground": ground, "controller": ctrl}
            if not seeds:                                   # fixed action
                per = c.set_index("episode_seed")
                rows.append({**base, "row": "fixed", "model_seed": "", "n_seeds": 0,
                             "n_episodes": len(c), "mean_return": c["return"].mean(),
                             "fall_rate": c.fell.mean(), "toe_peak_bw": c.toe_peak_bw.mean(),
                             **(paired(per, ref) if ctrl != REF else {})})
                continue
            per_seed = []
            for s in seeds:
                cs = c[c.model_seed == s]
                r = {**base, "row": "seed", "model_seed": int(s), "n_seeds": 1,
                     "n_episodes": len(cs), "mean_return": cs["return"].mean(),
                     "fall_rate": cs.fell.mean(), "toe_peak_bw": cs.toe_peak_bw.mean(),
                     **paired(cs.set_index("episode_seed"), ref)}
                rows.append(r)
                per_seed.append(r)
            ps = pd.DataFrame(per_seed)
            avg = c.groupby("episode_seed")[["return", "fell"]].mean()
            rows.append({**base, "row": "mean over seeds", "model_seed": "", "n_seeds": len(seeds),
                         "n_episodes": len(c), "mean_return": ps.mean_return.mean(),
                         "mean_return_sd": ps.mean_return.std(ddof=1),
                         "fall_rate": ps.fall_rate.mean(), "fall_rate_sd": ps.fall_rate.std(ddof=1),
                         "toe_peak_bw": ps.toe_peak_bw.mean(), "toe_peak_bw_sd": ps.toe_peak_bw.std(ddof=1),
                         **paired(avg, ref)})
    cols = ["ground", "controller", "row", "model_seed", "n_seeds", "n_episodes",
            "mean_return", "mean_return_sd", "fall_rate", "fall_rate_sd", "toe_peak_bw", "toe_peak_bw_sd",
            "n_paired", "d_return", "d_return_lo", "d_return_hi",
            "d_fall_rate", "d_fall_rate_lo", "d_fall_rate_hi"]
    return pd.DataFrame(rows).reindex(columns=cols)


def _pm(m, sd, fmt):
    return fmt.format(m) + ("" if pd.isna(sd) else " ± " + fmt.format(sd))


def _ci(m, lo, hi, scale=1.0, fmt="{:+.2f}"):
    if pd.isna(m):
        return "0 (reference)"
    return f"{fmt.format(m * scale)} [{fmt.format(lo * scale)}, {fmt.format(hi * scale)}]"


def to_markdown(s, n_eps):
    lines = ["# Controller comparison: summary", "",
             "Drafted with Claude Code (Anthropic), for the AI declaration. Generated by "
             "`python -m comparison.stats`; do not edit by hand.", "",
             f"Episodes per controller (per model seed): {n_eps}, seeds 1000-{999 + n_eps}, deterministic. "
             "Fixed reference = (-20 deg, 500, 10); best_fixed = (-20 deg, 500, 6). Network rows marked "
             "\"mean ± sd\" give the mean and the std across model seeds of the per-seed means. "
             "Δ = controller − reference on the same episode seeds, 95% bootstrap interval over episodes "
             f"({N_BOOT:,} resamples, RNG seed {BOOT_SEED}). Δ fall rate is in percentage points.", ""]
    for ground, g in s.groupby("ground", sort=False):
        title = "Training-range ground" if ground == "train" else "Test ground (out-of-range stress)"
        lines += [f"## {title} (`{ground}`)", "",
                  "| controller | seed | mean return | fall rate (%) | toe peak (BW) | Δ return [95%] | Δ fall rate, pts [95%] |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            seed = {"fixed": "-", "seed": str(r.model_seed), "mean over seeds": f"mean ± sd ({int(r.n_seeds)})"}[r.row]
            lines.append(
                f"| {label(r.controller)} | {seed} | {_pm(r.mean_return, r.mean_return_sd, '{:.2f}')} | "
                f"{_pm(100 * r.fall_rate, 100 * r.fall_rate_sd if not pd.isna(r.fall_rate_sd) else np.nan, '{:.1f}')} | "
                f"{_pm(r.toe_peak_bw, r.toe_peak_bw_sd, '{:.3f}')} | "
                f"{_ci(r.d_return, r.d_return_lo, r.d_return_hi)} | "
                f"{_ci(r.d_fall_rate, r.d_fall_rate_lo, r.d_fall_rate_hi, 100, '{:+.1f}')} |")
        lines.append("")
    lines += ["Note: on the test ground the yield force (400-590 N per contact point) is below 1.58 BW "
              "(about 1140 N), so the toe peak is capped by the ground giving way and the impact term is "
              "zero there; toe peaks on `test` are not comparable with `train`.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="inp", nargs="+", default=None,
                   help="episode CSVs (default: results/episodes.csv, plus episodes_tuned.csv if present)")
    p.add_argument("--out", default=None, help="output folder (default: the first input's folder)")
    a = p.parse_args()
    inp = [Path(x) for x in a.inp] if a.inp else [RESULTS / "episodes.csv"]
    for extra in ("episodes_tuned.csv", "episodes_rules.csv", "episodes_r4.csv"):
        if a.inp is None and (RESULTS / extra).exists():
            inp.append(RESULTS / extra)
    out = Path(a.out or inp[0].parent)
    df = pd.concat([pd.read_csv(f) for f in inp], ignore_index=True)
    # Stable order: fixed actions first (old best_fixed next to best_fixed_tuned), then networks.
    first = {c: i for i, c in enumerate(df.controller.drop_duplicates())}
    key = df.controller.map(lambda c: (ORDER.index(c), 0) if c in ORDER else (len(ORDER), first[c]))
    df = df.iloc[sorted(range(len(df)), key=lambda i: key.iat[i])].reset_index(drop=True)
    s = summarise(df)
    s.to_csv(out / "summary.csv", index=False, float_format="%.5g")
    n_eps = int(df.groupby(["ground", "controller", "model_seed"], dropna=False).size().min())
    md = to_markdown(s, n_eps)
    tuned = out / "best_fixed_tuned.json"
    if "best_fixed_tuned" in set(df.controller) and tuned.exists():
        t = json.loads(tuned.read_text())
        a0 = tuple(t["action"])
        md += (f"best_fixed_tuned = ({a0[0]:g} deg, {a0[1]:g}, {a0[2]:g}): chosen by `comparison/tune.py` "
               f"as the highest mean return among grid cells whose fall rate is not above the reference's, "
               f"on tuning seeds {t['tuning_seeds'][0]}-{t['tuning_seeds'][1]} (training ground). "
               "The old best_fixed (-20, 500, 6) row is kept for comparison.\n")
    rules_json = out / "rules_tuned.json"
    if set(RULES) & set(df.controller) and rules_json.exists():
        t = json.loads(rules_json.read_text())
        fmt = lambda r: ", ".join(f"{k}={v:g}" for k, v in r.items() if k != "type")
        md += ("\nHand rules (`comparison/rules.py`): theta_d -20 deg and B 8 fixed, K 500 outside a window after "
               "contact and K_dip inside it, tuned on seeds 2000-2099 (training ground) with fall rate not above "
               "the reference's. "
               f"R1_timed_dip ({fmt(t['timed']['rule'])}): constant dip, no ground information. "
               f"R2_sensed_dip ({fmt(t['sensed']['rule'])}): K_dip = a + b (z - {t['tuning']['Z0']:g}), "
               "z = log10 running-max vertical-GRF rate over the first m frames (causal, sensors only). "
               f"R3_oracle_dip ({fmt(t['oracle']['rule'])}): K_dip = a + b (log10 k0 - {t['tuning']['LK0']:.3f}) "
               "from the TRUE ground stiffness: privileged, an upper bound, not a deployable controller.\n")
        v = vs_networks(df)
        v.to_csv(out / "summary_rules_vs_networks.csv", index=False, float_format="%.5g")
        md += ("\n## Hand rules vs networks\n\nΔ = rule − network (network averaged over its model seeds per "
               "episode), same episode seeds, 95% bootstrap interval over episodes "
               f"({N_BOOT:,} resamples, RNG seed {BOOT_SEED}).\n\n"
               "| ground | rule | network | Δ return [95%] | Δ fall rate, pts [95%] |\n|---|---|---|---|---|\n")
        for _, r in v.iterrows():
            md += (f"| {r.ground} | {label(r.rule)} | {r.network} | {_ci(r.d_return, r.d_return_lo, r.d_return_hi)} | "
                   f"{_ci(r.d_fall_rate, r.d_fall_rate_lo, r.d_fall_rate_hi, 100, '{:+.1f}')} |\n")
    r4_json = out / "r4_tuned.json"
    if "R4_dip_plus_schedule" in set(df.controller) and r4_json.exists():
        r4 = json.loads(r4_json.read_text())["rule"]
        th = ("theta_d constant at -20 deg (tuned theta_start = -20, so no ramp)" if r4["theta_start"] == -20
              else f"theta_d ramping from {r4['theta_start']:g} to -20 deg over {r4['T_theta']:g} steps")
        bb = (f"B constant at {r4['B_start']:g} (tuned B_start = B_end, so no ramp)" if r4["B_start"] == r4["B_end"]
              else f"B ramping from {r4['B_start']:g} to {r4['B_end']:g} over {r4['T_B']:g} steps")
        md += ("\nR4_dip_plus_schedule (`comparison/rules.py --r4`, Task F): R1-style K dip "
               f"(K_dip={r4['k_dip']:g}, {r4['start_ms']:g}-{r4['end_ms']:g} ms), {th}, {bb}; tuned on seeds "
               "2000-2099 with the same fall-rate constraint.\n")
        v1 = vs_r1(df)
        v1.to_csv(out / "summary_rules_vs_r1.csv", index=False, float_format="%.5g")
        md += ("\n## Hand rules vs R1\n\nΔ = rule − R1_timed_dip, same episode seeds, 95% bootstrap interval "
               "over episodes.\n\n| ground | rule | Δ return [95%] | Δ fall rate, pts [95%] |\n|---|---|---|---|\n")
        for _, r in v1.iterrows():
            md += (f"| {r.ground} | {label(r.rule)} | {_ci(r.d_return, r.d_return_lo, r.d_return_hi)} | "
                   f"{_ci(r.d_fall_rate, r.d_fall_rate_lo, r.d_fall_rate_hi, 100, '{:+.1f}')} |\n")
    (out / "summary.md").write_text(md)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
