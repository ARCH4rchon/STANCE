"""Step 1 (rules): can the sensors tell the ground stiffness early in stance?

Drafted with Claude Code (Anthropic), for the AI declaration.

The 11 observation channels per frame (common/env_setup.SENSOR_NAMES; common/inputs.py only
selects frames/channels from them):
    0 ankle_angle (encoder)        1 ankle_rate (encoder)       2 ankle_torque (motor current)
    3 vertical_grf (pylon load cell, total normal force, N)     4 pylon_compression (linear sensor)
    5 shank_tilt (IMU)             6 shank_tilt_rate (IMU)      7 vertical_accel (IMU)
    8-10 previous action (theta_d, K, B; memory, not sensors)
There is no toe-force or sinkage sensor: toe force exists only in the env's `info` (reported
below as "info-only" for comparison, never used by a controller) and sinkage is not observed.

Contact time: the env starts every episode with the heel already on the ground (reset places
it there with downward velocity), so frame j is j x 5 ms after contact (frame 0 = reset obs).

The reference controller (-20, 500, 10) is run on tuning seeds 2000-2099, training ground.
For each causal feature and cutoff N (frames 0..N, i.e. information available at N x 5 ms),
the script reports, against log10(true k0): Spearman rho, and R^2 of a one-variable least-squares
line on the feature (and on log10|feature|; the better is reported, with the transform).

    python -m comparison.sensing [--episodes 100] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparison.harness import RESULTS, log
from common.env_setup import N_SENSORS, SENSOR_NAMES, make_env, to_norm_action
from common.evaluate import REFERENCE
from common.policies import FixedPolicy

CUTOFFS = [1, 2, 3, 5, 10, 20, 30]          # frames after contact (x 5 ms)
DT = 0.005


def record(seeds):
    """Per episode: raw sensor frames 0..T (T+1, 11), info toe force per step, true k0."""
    env = make_env("train")
    pol = FixedPolicy(to_norm_action(env, *REFERENCE))
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        frames, toe = [np.asarray(obs).reshape(-1, N_SENSORS)[-1].copy()], []
        k0 = float(env.unwrapped.ground.k0[0])
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(pol.act(obs))
            frames.append(np.asarray(obs).reshape(-1, N_SENSORS)[-1].copy())
            toe.append(float(info["toe_force_bw"]))
            done = term or trunc
        out.append({"seed": int(s), "k0": k0, "frames": np.array(frames), "toe": np.array(toe),
                    "fell": int(term)})
    env.close()
    return out


def features(ep, n):
    """Causal features from frames 0..n. Keys: (name, is_sensor)."""
    f = ep["frames"][: n + 1]
    grf, comp, acc, torque = f[:, 3], f[:, 4], f[:, 7], f[:, 2]
    d = {}
    for c in range(8):
        nm = SENSOR_NAMES[c]
        d[(f"{nm} max", True)] = f[:, c].max()
        d[(f"{nm} min", True)] = f[:, c].min()
        d[(f"{nm} at cutoff", True)] = f[-1, c]
    if n >= 1:
        rate = np.diff(grf) / DT
        d[("grf rate max (N/s)", True)] = rate.max()
        d[("grf rise per pylon compression (N/m)", True)] = (grf.max() - grf[0]) / max(comp.max() - comp[0], 1e-9)
        d[("torque rate max", True)] = (np.diff(torque) / DT).max()
        d[("argmax grf (frames)", True)] = float(np.argmax(grf))
    toe = ep["toe"][: n]
    if len(toe):
        d[("toe force max (info-only, not a sensor)", False)] = toe.max()
    return d


def r2(x, y):
    if np.std(x) == 0:
        return np.nan
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    return 1 - res.var() / y.var()


def spearman(x, y):
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 else np.nan


def analyse(eps):
    y_all = np.log10([e["k0"] for e in eps])
    rows = []
    for n in CUTOFFS:
        ok = [i for i, e in enumerate(eps) if len(e["frames"]) > n]
        feats = [features(eps[i], n) for i in ok]
        y = y_all[ok]
        for key in feats[0]:
            name, is_sensor = key
            x = np.array([f[key] for f in feats], dtype=float)
            r_lin = r2(x, y)
            ax = np.abs(x)
            r_log = r2(np.log10(ax), y) if np.all(ax > 0) else np.nan
            best = "log10|x|" if (not np.isnan(r_log) and r_log > (r_lin if not np.isnan(r_lin) else -1)) else "x"
            rows.append({"cutoff_frames": n, "cutoff_ms": n * 5, "feature": name, "sensor": is_sensor,
                         "n_episodes": len(ok), "spearman": spearman(x, y),
                         "r2": max(v for v in (r_lin, r_log) if not np.isnan(v)) if not (np.isnan(r_lin) and np.isnan(r_log)) else np.nan,
                         "r2_transform": best})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default=str(RESULTS))
    a = p.parse_args()
    seeds = list(range(2000, 2000 + a.episodes))
    eps = record(seeds)
    log(f"recorded {len(eps)} reference episodes (seeds {seeds[0]}-{seeds[-1]}), "
        f"{sum(e['fell'] for e in eps)} falls")
    df = analyse(eps)
    df.to_csv(Path(a.out) / "sensing.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    best = (df[df.sensor].assign(a=lambda d: d.spearman.abs())
              .sort_values(["cutoff_frames", "a"], ascending=[True, False]).groupby("cutoff_frames").head(4))
    print(best.drop(columns="a").to_string(index=False))
    print(df[~df.sensor].to_string(index=False))


if __name__ == "__main__":
    main()
