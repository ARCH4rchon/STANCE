"""Shared machinery for compare.py and robustness.py.

Drafted with Claude Code (Anthropic), for the AI declaration.

Everything runs through the unchanged project code:
    common.env_setup.make_env      builds the environment and applies wrappers
    common.evaluate.run_episodes   plays the episodes (deterministic, one per seed)
    agent.train.make_wrappers      the network's input wrappers plus the residual action
    common.policies                FixedPolicy / SB3Policy / RecurrentSB3Policy
Nothing in Gym/, common/, agent/ or trained_networks/ is modified. The extra wrappers here
(ground stiffness override, sensor noise, per-step recorder) sit around the unchanged env.

A "cell" is one controller (one model seed for networks) on one condition, for a list of
episode seeds. Each finished cell is saved atomically to <out>/cells/<cell id>.csv, so an
interrupted run resumes by skipping cells whose file already holds every requested seed.
"""
import collections
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from agent.train import NETWORKS, make_wrappers
from common.env_setup import N_SENSORS, make_env, to_norm_action
from common.evaluate import EVAL_SEEDS, REFERENCE, run_episodes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # Network_Agent/
TRAINED = ROOT / "trained_networks"
RESULTS = HERE / "results"

# Fixed controllers. best_fixed is (-20, 500, 6); see NOTES.md for why it is kept but flagged.
FIXED = {"reference": tuple(REFERENCE), "best_fixed": (-20.0, 500.0, 6.0)}

# Sensor channels that get noise (0-7). Channels 8-10 are the previous action, not a sensor.
SENSOR_CHANNELS = list(range(8))

EP_COLUMNS = ["controller", "model_seed", "episode_seed", "ground", "return", "fell", "steps",
              "toe_peak_bw", "total_peak_bw", "fall_reason", "impact_sum", "smooth_sum",
              "fall_penalty", "mean_theta_deg", "mean_K", "mean_B", "k0", "noise"]


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- controllers
def discover_controllers():
    """The two fixed actions, then every trained_networks/<network>/final_s*/ run (globbed)."""
    ctrls = [{"controller": k, "model_seed": "", "kind": "fixed", "action": v} for k, v in FIXED.items()]
    for run in sorted(TRAINED.glob("*/final_s*")):
        if not all((run / f).exists() for f in ("model.zip", "vecnormalize.pkl", "config.json")):
            continue
        saved = json.loads((run / "config.json").read_text())
        ctrls.append({"controller": run.parent.name, "model_seed": int(saved["seed"]),
                      "kind": "network", "run": str(run)})
    return ctrls


def ctrl_label(c):
    if c["kind"] == "rule":    # include the parameters, so a re-tuned rule never reuses old cells
        r = c["rule"]
        return c["controller"] + "__" + "_".join(f"{k}{r[k]:g}" for k in sorted(r) if k != "type")
    return c["controller"] if c["kind"] == "fixed" else f"{c['controller']}_s{c['model_seed']}"


_NET_CACHE = {}


def load_network(run_dir):
    """Same steps as demo_agent.load_run (kept here so the viewer import is not needed)."""
    import importlib
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    if run_dir in _NET_CACHE:
        return _NET_CACHE[run_dir]
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    cfg = saved["config"]
    net = importlib.import_module(NETWORKS[saved["network"]])
    wrappers = make_wrappers(cfg, net)
    model = net.ALGO.load(run_dir / "model.zip", device="cpu")
    vn = VecNormalize.load(str(run_dir / "vecnormalize.pkl"),
                           DummyVecEnv([lambda: make_env("train", wrappers)]))
    vn.training = False
    _NET_CACHE[str(run_dir)] = (net, model, vn, wrappers, cfg)
    return _NET_CACHE[str(run_dir)]


def sensor_std():
    """One per-sensor std vector (11 values) shared by every network: the newest-frame std from
    each run's VecNormalize obs_rms, averaged over all runs found. Used for the noise sweep."""
    stds = []
    for c in discover_controllers():
        if c["kind"] != "network":
            continue
        net, model, vn, wrappers, cfg = load_network(c["run"])
        var = np.asarray(vn.obs_rms.var, dtype=np.float64)
        keep = [i for i in range(N_SENSORS) if i not in cfg.get("drop_sensors", [])]
        newest = var.reshape(-1, len(keep))[-1]
        full = np.full(N_SENSORS, np.nan)
        full[keep] = np.sqrt(newest)
        stds.append(full)
    return np.nanmean(np.array(stds), axis=0)


# ---------------------------------------------------------------- wrappers
class GroundStiffness(gym.Wrapper):
    """Sets the ground base stiffness k0 to a fixed value after each reset.

    The env draws the ground in reset() and the reset observation does not depend on k0
    (the GRF channel is 0 until the first step), so overriding it here is equivalent to having
    drawn that k0. The damping c is recomputed with the same formula as GroundModel.sample, so
    the damping ratio zeta keeps its drawn value. Other parameters keep their seeded draws.
    """

    def __init__(self, env, k0):
        super().__init__(env)
        self.k0 = float(k0)

    def reset(self, **kw):
        from stance_env.ground import DAMP_LIMIT
        obs, info = self.env.reset(**kw)
        g = self.env.unwrapped.ground
        g.k0 = np.full(g.n, self.k0)
        g.c = 2.0 * g.zeta * np.sqrt(g.k0 * g.total_mass)
        if g.heel_mass is not None:
            g.c = np.minimum(g.c, DAMP_LIMIT * g.heel_mass / g.timestep)
        return obs, info


class SensorNoise(gym.ObservationWrapper):
    """Gaussian noise on sensor channels 0-7, sigma = level * per-sensor std.

    Each new sensor frame gets noise once and is kept in this wrapper's own history, so a
    reading carries the same noise as it moves through the 10-frame stack (as a real noisy
    sensor would). Zero padding frames at the start of an episode stay zero. The noise
    generator is seeded from the episode seed, so every controller sees the same draws.
    """

    def __init__(self, env, level, std):
        super().__init__(env)
        self.sigma = float(level) * np.asarray(std, dtype=np.float64)
        self.frames = env.observation_space.shape[0] // N_SENSORS
        self.rng = np.random.default_rng(0)
        self.hist = None

    def _noisy(self, frame):
        f = np.array(frame, dtype=np.float64)
        f[SENSOR_CHANNELS] += self.sigma[SENSOR_CHANNELS] * self.rng.standard_normal(len(SENSOR_CHANNELS))
        return f

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.rng = np.random.default_rng([int(seed or 0), 424242])
        grid = np.asarray(obs, dtype=np.float64).reshape(self.frames, N_SENSORS)
        self.hist = collections.deque(list(grid[:-1]) + [self._noisy(grid[-1])], maxlen=self.frames)
        self.clean = np.asarray(obs, dtype=np.float32)   # noise-free obs, for noise_check.py
        return np.concatenate(self.hist).astype(np.float32), info

    def observation(self, obs):
        self.clean = np.asarray(obs, dtype=np.float32)
        newest = np.asarray(obs, dtype=np.float64).reshape(self.frames, N_SENSORS)[-1]
        self.hist.append(self._noisy(newest))
        return np.concatenate(self.hist).astype(np.float32)


class StepRecorder(gym.Wrapper):
    """Outermost wrapper: per episode, sums the impact term (recomputed from info's toe force
    with the env's own W_IMPACT / LAMBDA_BW) and records the ground k0 the env reports."""

    def __init__(self, env):
        super().__init__(env)
        mod = sys.modules[type(env.unwrapped).__module__]
        self.w, self.lam, self.fall = mod.W_IMPACT, mod.LAMBDA_BW, mod.FALL_PENALTY
        self.records = []

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.records.append({"impact": 0.0, "k0": math.nan})
        return out

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        rec = self.records[-1]
        rec["impact"] -= self.w * max(0.0, float(info["toe_force_bw"]) - self.lam) ** 2
        rec["k0"] = float(np.asarray(info.get("k0", [math.nan]))[0])
        return obs, r, term, trunc, info


# ---------------------------------------------------------------- one cell
def run_cell(ctrl, ground, seeds, k0=None, noise=0.0, std=None):
    """Play one cell and return its per-episode rows (EP_COLUMNS).

    ground: "train" or "test" (env id). k0: override the ground stiffness (robustness sweep).
    noise: sensor noise level in units of std (only applied to networks; fixed actions ignore
    the sensors, so the noise would not change them).
    """
    inner = []
    if k0 is not None:
        inner.append(lambda e: GroundStiffness(e, k0))
    if ctrl["kind"] == "network":
        net, model, vn, wrappers, cfg = load_network(ctrl["run"])
        if noise > 0:
            inner.append(lambda e: SensorNoise(e, noise, std))
        env = StepRecorder(make_env(ground, inner + list(wrappers)))
        policy = net.POLICY(model, vn)
    elif ctrl["kind"] == "rule":                       # hand-written rules (comparison/rules.py)
        from comparison.rules import make_rule
        env = StepRecorder(make_env(ground, inner))
        policy = make_rule(ctrl, env)
    else:
        from common.policies import FixedPolicy
        env = StepRecorder(make_env(ground, inner))
        policy = FixedPolicy(to_norm_action(env, *ctrl["action"]))
    eps = run_episodes(policy, env, seeds)
    env.close()
    rows = []
    for e, rec in zip(eps, env.records):
        fell = int(e["terminated"])
        fall_pen = -env.fall * fell
        rows.append({
            "controller": ctrl["controller"], "model_seed": ctrl["model_seed"],
            "episode_seed": e["seed"], "ground": ground, "return": e["ret"], "fell": fell,
            "steps": e["steps"], "toe_peak_bw": e["toe_peak"], "total_peak_bw": e["total_peak"],
            "fall_reason": e["reason"], "impact_sum": rec["impact"],
            # survival is +1 per step, so the smoothness sum is what remains of the return
            "smooth_sum": e["ret"] - e["steps"] - rec["impact"] - fall_pen,
            "fall_penalty": fall_pen, "mean_theta_deg": e["theta"], "mean_K": e["K"],
            "mean_B": e["B"], "k0": rec["k0"], "noise": noise})
    return rows


# ---------------------------------------------------------------- storage / resume
def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: _fmt(r.get(k, "")) for k in columns} for r in rows)
    os.replace(tmp, path)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cell_done(path, seeds):
    if not Path(path).exists():
        return False
    have = {int(r["episode_seed"]) for r in read_csv(path)}
    return set(seeds) <= have


def _work(job):
    """Pool worker: run one cell, save it, return its id and run time."""
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    rows = run_cell(job["ctrl"], job["ground"], job["seeds"], job.get("k0"),
                    job.get("noise", 0.0), job.get("std"))
    for r in rows:
        r.update(job.get("tags", {}))
    write_csv(job["path"], rows, job["columns"])
    return job["id"], time.time() - t0, len(rows)


def run_jobs(jobs, workers):
    """Run the cells not finished yet, in parallel. Returns the total wall time."""
    todo = [j for j in jobs if not cell_done(j["path"], j["seeds"])]
    log(f"{len(jobs)} cells, {len(jobs) - len(todo)} already done, {len(todo)} to run "
        f"on {min(workers, max(len(todo), 1))} workers")
    t0 = time.time()
    if todo:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")      # fresh processes: no shared torch/MuJoCo state
        with ctx.Pool(min(workers, len(todo))) as pool:
            for i, (cid, dt, n) in enumerate(pool.imap_unordered(_work, todo), 1):
                log(f"  [{i}/{len(todo)}] {cid}: {n} episodes in {dt:.0f} s")
    return time.time() - t0


def merge_cells(jobs, out_path, columns):
    """Concatenate the cell files (requested seeds only, in job order) into one CSV."""
    rows = []
    for j in jobs:
        keep = set(j["seeds"])
        rows += [r for r in read_csv(j["path"]) if int(r["episode_seed"]) in keep]
    write_csv(out_path, rows, columns)
    return rows


def default_workers():
    return max(1, (os.cpu_count() or 2) - 2)


def eval_seeds(n):
    if not 1 <= n <= len(EVAL_SEEDS):
        raise SystemExit(f"--episodes must be 1-{len(EVAL_SEEDS)} (seeds 1000-1199)")
    return EVAL_SEEDS[:n]
