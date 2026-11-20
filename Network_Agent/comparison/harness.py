"""Shared pieces for compare.py: the controllers, the ground-stiffness wrapper and a parallel runner."""

import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
 
import gymnasium as gym
import numpy as np
 
from agent.train import NETWORKS, make_wrappers
from common.env_setup import make_env, to_norm_action
from common.evaluate import REFERENCE, run_episodes
from common.policies import FixedPolicy
from comparison.rules import NAMES, R1, R3, DipRule
 
HERE = Path(__file__).resolve().parent
TRAINED = HERE.parent / "trained_networks"
RESULTS = HERE / "results"
 
# Fixed actions (angle in degrees, K, B). best_fixed_tuned was the best of an 80-action grid on seeds 2000-2099.
FIXED = {"reference": tuple(REFERENCE), "best_fixed_tuned": (-20.0, 500.0, 8.0)}
COLUMNS = ["controller", "model_seed", "episode_seed", "ground", "level", "return", "fell", "steps", "toe_peak_bw"]
 
 
def controllers():
    """The fixed actions, the two hand rules, then every trained run found under trained_networks/."""
    ctrls = [{"controller": n, "model_seed": "", "kind": "fixed", "action": a} for n, a in FIXED.items()]
    ctrls += [{"controller": NAMES[k], "model_seed": "", "kind": "rule", "rule": r}
              for k, r in (("timed", R1), ("oracle", R3))]
    for run in sorted(TRAINED.glob("*/final_s*")):
        seed = json.loads((run / "config.json").read_text())["seed"]
        ctrls.append({"controller": run.parent.name, "model_seed": int(seed), "kind": "network", "run": run})
    return ctrls
 
 
_loaded = {}
 
 
def load_network(run):
    """Load one trained run the same way demo.py does (once per process)."""
    if run not in _loaded:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        saved = json.loads((run / "config.json").read_text())
        net = importlib.import_module(NETWORKS[saved["network"]])
        wrappers = make_wrappers(saved["config"], net)
        model = net.ALGO.load(run / "model.zip", device="cpu")
        vn = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: make_env("train", wrappers)]))
        vn.training = False
        _loaded[run] = (net, model, vn, wrappers)
    return _loaded[run]
 
 
class GroundStiffness(gym.Wrapper):
    """Sets the ground base stiffness k0 right after each reset (used for the stiffness sweep).
 
    The reset observation does not depend on k0, so this equals drawing that k0 in the first place.
    The damping is recomputed with the same formula as GroundModel.sample, so the damping ratio keeps
    its drawn value. The other ground parameters keep their seeded draws.
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
 
 
def run_cell(ctrl, ground, seeds, k0=None):
    """Play one controller on the given episode seeds (optionally with the ground stiffness forced to k0)."""
    extra = [lambda e: GroundStiffness(e, k0)] if k0 else []
    if ctrl["kind"] == "network":
        net, model, vn, wrappers = load_network(ctrl["run"])
        env = make_env(ground, extra + list(wrappers))
        policy = net.POLICY(model, vn)
    else:
        env = make_env(ground, extra)
        if ctrl["kind"] == "rule":
            policy = DipRule(env, **ctrl["rule"])
        else:
            policy = FixedPolicy(to_norm_action(env, *ctrl["action"]))
    episodes = run_episodes(policy, env, seeds)
    env.close()
    return [{"controller": ctrl["controller"], "model_seed": ctrl["model_seed"], "episode_seed": e["seed"],
             "ground": ground, "level": k0, "return": e["ret"], "fell": e["terminated"],
             "steps": e["steps"], "toe_peak_bw": e["toe_peak"]} for e in episodes]
 
 
def _work(job):
    import torch
    torch.set_num_threads(1)
    return run_cell(*job)
 
 
def run_all(jobs, workers=None):
    """jobs: list of (controller, ground, seeds, k0). Runs them in parallel; returns all rows, in job order."""
    workers = min(workers or max(1, (os.cpu_count() or 2) - 2), len(jobs))
    with mp.get_context("spawn").Pool(workers) as pool:
        return [row for rows in pool.imap(_work, jobs) for row in rows]
 