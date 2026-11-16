"""Hand-written "K dip" controllers, their tuning and their evaluation.

Drafted with Claude Code (Anthropic), for the AI declaration.

All three hold theta_d = -20 deg and B = 8 N.m.s/rad, and K = 500 N.m/rad outside a window
[start_ms, end_ms) after contact (episodes start at heel contact; action t acts during
[5t, 5t + 5) ms). Inside the window K = K_dip, clipped to [5, 500]:
  R1 timed_dip   K_dip constant.                               params: k_dip, start_ms, end_ms
  R2 sensed_dip  K_dip = a + b (z - Z0), z = log10 of the running max of the vertical-GRF rate
                 (channel 3, N/s) over frames 0..min(t, m): causal, from the observation only.
                 Before any rate is available z = Z0 (so it acts like R1 with K_dip = a).
                 Window taken from R1.                         params: a, b, m
  R3 oracle_dip  K_dip = a + b (log10 k0 - LK0), k0 = the TRUE ground stiffness read from the env
                 at reset. PRIVILEGED (not available to a real controller): an upper bound.
                 Window taken from R1.                         params: a, b
Same interface as common.policies.FixedPolicy: reset(), act(obs) -> normalised action.

Tuning (training ground, tuning seeds 2000-2099 only), by grid, constraint: fall rate <= the
reference's on the same seeds; objective: mean return. R1 first, then R2 and R3 with R1's window.
Log of every cell -> results/tuning_rules.csv; chosen parameters -> results/rules_tuned.json.
Evaluation (seeds 1000-1199, train and test ground) -> results/episodes_rules.csv; then
`python -m comparison.stats` and `python -m comparison.robustness` pick the rules up.

    python -m comparison.rules [--episodes 100] [--eval-episodes 200] [--out DIR]
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from comparison.harness import EP_COLUMNS, RESULTS, default_workers, eval_seeds, log, merge_cells, run_jobs
from common.env_setup import N_SENSORS, to_norm_action
from common.evaluate import REFERENCE

THETA, B_FIX, K_HIGH, K_MIN = -20.0, 8.0, 500.0, 5.0
STEP_MS, DT = 5.0, 0.005
GRF = 3
Z0 = 5.1                      # log10 N/s: median of the feature on tuning seeds (sensing.py, reference)
LK0 = math.log10(2.236e4)     # log-centre of the training k0 range (5e3-1e5)
NAMES = {"timed": "R1_timed_dip", "sensed": "R2_sensed_dip", "oracle": "R3_oracle_dip"}
TUNE_SEEDS = list(range(2000, 2100))

R1_GRID = dict(k_dip=[25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
               start_ms=[0, 10, 25, 50], end_ms=[50, 100, 150, 200, 300])
R2_GRID = dict(a=[50, 100, 150, 200, 250, 300, 400], b=[-600, -400, -200, -100, 0, 100, 200], m=[2, 3, 5])
R3_GRID = dict(a=[50, 100, 150, 200, 250, 300, 400], b=[-300, -200, -100, -50, 0, 50, 100])


class DipRule:
    def __init__(self, env, type, start_ms, end_ms, k_dip=None, a=None, b=None, m=3):
        self.env, self.type = env, type
        self.start_ms, self.end_ms = float(start_ms), float(end_ms)
        self.k_dip, self.a, self.b, self.m = k_dip, a, b, int(m)
        self._cache = {}
        self.reset()

    def reset(self):
        self.t, self.prev_grf, self.rate_max = 0, None, -math.inf
        self.k0 = float(self.env.unwrapped.ground.k0[0]) if self.type == "oracle" else None

    def _norm(self, K):
        K = float(np.clip(K, K_MIN, K_HIGH))
        if K not in self._cache:
            self._cache[K] = to_norm_action(self.env, THETA, K, B_FIX)
        return self._cache[K]

    def k_now(self):
        if not (self.start_ms <= self.t * STEP_MS < self.end_ms):
            return K_HIGH
        if self.type == "timed":
            return self.k_dip
        if self.type == "sensed":
            z = math.log10(self.rate_max) if self.rate_max > 0 else Z0
            return self.a + self.b * (z - Z0)
        return self.a + self.b * (math.log10(self.k0) - LK0)

    def act(self, obs):
        grf = float(np.asarray(obs)[-N_SENSORS + GRF])       # newest frame = frame t
        if self.prev_grf is not None and self.t <= self.m:
            self.rate_max = max(self.rate_max, (grf - self.prev_grf) / DT)
        self.prev_grf = grf
        a = self._norm(self.k_now())
        self.t += 1
        return a


def make_rule(ctrl, env):
    return DipRule(env, **ctrl["rule"])


def rule_ctrl(name, rule):
    return {"controller": name, "model_seed": "", "kind": "rule", "rule": rule}


def tuned_controllers(out=RESULTS):
    p = Path(out) / "rules_tuned.json"
    if not p.exists():
        return []
    t = json.loads(p.read_text())
    return [rule_ctrl(NAMES[k], t[k]["rule"]) for k in ("timed", "sensed", "oracle")]


# ---------------------------------------------------------------- tuning
def _tag(rule):
    return "_".join(f"{k}{rule[k]:g}" if isinstance(rule[k], (int, float)) else f"{k}{rule[k]}"
                    for k in sorted(rule) if k != "type")


def _job(out, ctrl, ground, seeds, prefix):
    cid = f"{prefix}__{ground}__{ctrl['controller']}__{_tag(ctrl['rule'])}" if ctrl["kind"] == "rule" \
        else f"{prefix}__{ground}__{ctrl['controller']}"
    return {"id": cid, "ctrl": ctrl, "ground": ground, "seeds": seeds,
            "path": str(Path(out) / "cells" / f"{cid}.csv"), "columns": EP_COLUMNS}


def _score(jobs, seeds):
    rows = []
    for j in jobs:
        d = pd.read_csv(j["path"])
        d = d[d.episode_seed.isin(seeds)]
        rows.append({**{k: v for k, v in j["ctrl"].get("rule", {}).items()},
                     "rule": j["ctrl"]["controller"], "n_episodes": len(d),
                     "mean_return": d["return"].mean(), "fall_rate": d.fell.mean(),
                     "toe_peak_bw": d.toe_peak_bw.mean()})
    return pd.DataFrame(rows)


def _pick(df, ref_fall):
    df = df.copy()
    df["eligible"] = df.fall_rate <= ref_fall + 1e-12
    best = df[df.eligible].sort_values("mean_return", ascending=False).iloc[0]
    df["chosen"] = df.index == best.name
    return df, best


def tune(out, seeds, workers):
    ref_job = _job(out, {"controller": "reference", "model_seed": "", "kind": "fixed",
                         "action": tuple(REFERENCE)}, "train", seeds, "R_tune")
    run_jobs([ref_job], workers)
    ref = _score([ref_job], seeds).iloc[0]
    log(f"reference on tuning seeds: return {ref.mean_return:.2f}, fall {100 * ref.fall_rate:.1f}%")
    logs, chosen = [], {}

    # R1: full grid (end > start)
    r1 = [rule_ctrl(NAMES["timed"], {"type": "timed", "k_dip": k, "start_ms": s, "end_ms": e})
          for k, s, e in itertools.product(*R1_GRID.values()) if e > s]
    j1 = [_job(out, c, "train", seeds, "R_tune") for c in r1]
    run_jobs(j1, workers)
    d1, b1 = _pick(_score(j1, seeds), ref.fall_rate)
    logs.append(d1)
    chosen["timed"] = {"type": "timed", "k_dip": float(b1.k_dip), "start_ms": float(b1.start_ms),
                       "end_ms": float(b1.end_ms)}
    log(f"R1 chosen: {chosen['timed']} return {b1.mean_return:.2f} fall {100 * b1.fall_rate:.1f}%")
    win = {"start_ms": chosen["timed"]["start_ms"], "end_ms": chosen["timed"]["end_ms"]}

    # R2 and R3 with R1's window
    for typ, grid in (("sensed", R2_GRID), ("oracle", R3_GRID)):
        cs = [rule_ctrl(NAMES[typ], {"type": typ, **win, **dict(zip(grid, v))})
              for v in itertools.product(*grid.values())]
        js = [_job(out, c, "train", seeds, "R_tune") for c in cs]
        run_jobs(js, workers)
        d, b = _pick(_score(js, seeds), ref.fall_rate)
        logs.append(d)
        chosen[typ] = {"type": typ, **win, **{k: float(b[k]) for k in grid}}
        if typ == "sensed":
            chosen[typ]["m"] = int(b["m"])
        log(f"{NAMES[typ]} chosen: {chosen[typ]} return {b.mean_return:.2f} fall {100 * b.fall_rate:.1f}%")

    log_df = pd.concat(logs, ignore_index=True)
    log_df.insert(0, "reference_fall_rate", ref.fall_rate)
    log_df.insert(1, "reference_mean_return", ref.mean_return)
    cols = ["rule", "type", "k_dip", "a", "b", "m", "start_ms", "end_ms", "n_episodes", "mean_return",
            "fall_rate", "toe_peak_bw", "eligible", "chosen", "reference_mean_return", "reference_fall_rate"]
    log_df.reindex(columns=cols).to_csv(Path(out) / "tuning_rules.csv", index=False, float_format="%.5g")
    res = {k: {"rule": v, "controller": NAMES[k]} for k, v in chosen.items()}
    res["tuning"] = {"seeds": [seeds[0], seeds[-1]], "ground": "train",
                     "constraint": "fall rate <= reference fall rate on the tuning seeds",
                     "reference_fall_rate": float(ref.fall_rate), "Z0": Z0, "LK0": LK0,
                     "privileged": ["oracle"]}
    (Path(out) / "rules_tuned.json").write_text(json.dumps(res, indent=2))
    return res


def evaluate(out, seeds, workers):
    jobs = [_job(out, c, g, seeds, "R_eval") for g in ("train", "test") for c in tuned_controllers(out)]
    dt = run_jobs(jobs, workers)
    rows = merge_cells(jobs, Path(out) / "episodes_rules.csv", EP_COLUMNS)
    log(f"rules evaluated: {len(rows)} rows -> episodes_rules.csv ({dt:.0f} s)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=100, help="tuning episodes (seeds 2000..)")
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--workers", type=int, default=default_workers())
    a = p.parse_args()
    tune(a.out, TUNE_SEEDS[: a.episodes], a.workers)
    evaluate(a.out, eval_seeds(a.eval_episodes), a.workers)


if __name__ == "__main__":
    main()
