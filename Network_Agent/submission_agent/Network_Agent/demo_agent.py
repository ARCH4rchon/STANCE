"""
Demo of the learning agent: load or train a network, then compare it with the best fixed action.

Uses: agent.train (TRAIN, NETWORKS, make_wrappers, make_venv), common.env_setup,
common.evaluate and common.policies.
"""
# Libraries
import argparse
import importlib
import json
import tempfile
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local imports
from agent.train import NETWORKS, TRAIN, make_wrappers, make_venv
from common.env_setup import make_env, to_norm_action
from common.evaluate import EVAL_SEEDS, REFERENCE, run_episodes, summarise
from common.policies import FixedPolicy

ROWS = [("survival_pct", "survival (%)"), ("toe_peak_bw", "toe peak (BW)"),
        ("mean_theta_deg", "mean angle (deg)"), ("mean_K", "mean K (N.m/rad)"),
        ("mean_B", "mean B (N.m.s/rad)"), ("mean_return", "mean return")]


def load_run(run_dir):
    """
    Load a saved run folder: its config, the network's wrappers, the model and the VecNormalize.
    The run folder must contain model.zip, vecnormalize.pkl and config.json.
    """
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    cfg = saved["config"]
    net = importlib.import_module(NETWORKS[saved["network"]])
    wrappers = make_wrappers(cfg, net)
    model = net.ALGO.load(run_dir / "model.zip", device="cpu")
    vn = VecNormalize.load(str(run_dir / "vecnormalize.pkl"),
                           DummyVecEnv([lambda: make_env("train", wrappers)]))
    vn.training = False
    return net, model, vn, wrappers


def train_run(network, steps=None, seed=0):
    """
    Train a network from scratch with the settings of agent/train.py, in a temporary folder.
    Same steps as agent/train.py: merged config, wrappers, normalised parallel environments,
    the PPO model and `model.learn`.
    """
    net = importlib.import_module(NETWORKS[network])
    cfg = {**TRAIN, **net.CONFIG}
    if steps:
        cfg["total_steps"] = steps
    wrappers = make_wrappers(cfg, net)
    run_dir = Path(tempfile.mkdtemp(prefix=f"stance_demo_{net.NAME}_"))
    (run_dir / "config.json").write_text(
        json.dumps({"network": network, "seed": seed, "config": cfg}, indent=2))
    venv = make_venv(cfg, wrappers, seed)
    model = net.ALGO(
        net.POLICY_NAME, venv, policy_kwargs=net.policy_kwargs(cfg),
        n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"], gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"], ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"], seed=seed, device="cpu", verbose=0)
    print(f"training {net.NAME} for {cfg['total_steps']:,} steps; run folder: {run_dir}")
    model.learn(cfg["total_steps"])
    model.save(run_dir / "model.zip")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    venv.training = False
    return net, model, venv, wrappers


def main():
    """Parse the arguments, get a model, print its architecture, evaluate it and the reference."""
    p = argparse.ArgumentParser(description="Demo of the STANCE learning agent.")
    p.add_argument("--network", choices=sorted(NETWORKS), default="mlp", help="choice of network to train")
    p.add_argument("--model", metavar="DIR", help="saved run folder (model.zip, vecnormalize.pkl, config.json)")
    p.add_argument("--episodes", type=int, default=20, help="evaluation episodes (seeds 1000 and up)")
    p.add_argument("--train", action="store_true", help="train from scratch instead of loading a pre-trained model")
    p.add_argument("--steps", type=int, help="with --train: training steps")
    a = p.parse_args()

    # Get a model: load a saved run, or train one.
    if a.model:
        net, model, vn, wrappers = load_run(a.model)
    elif a.train:
        net, model, vn, wrappers = train_run(a.network, a.steps)
    else:
        print("Error, please see README.md.")
        return

    # The network's layers (PyTorch's printout of the policy) and parameter count.
    print(f"\n{net.NAME} layers:\n{model.policy}")
    print(f"parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Evaluate the policy and the best fixed action on the same seeds.
    seeds = EVAL_SEEDS[: a.episodes]
    env = make_env("train", wrappers)
    agent = summarise(run_episodes(net.POLICY(model, vn), env, seeds))
    env.close()
    env = make_env("train")
    reference = summarise(run_episodes(FixedPolicy(to_norm_action(env, *REFERENCE)), env, seeds))
    env.close()

    # Side-by-side table.
    print(f"\n{len(seeds)} episodes, seeds {seeds[0]}-{seeds[-1]}, deterministic policy")
    print(f"{'':20s} {net.NAME:>14s} {'fixed action':>14s}")
    for key, label in ROWS:
        print(f"{label:20s} {agent[key]:14.2f} {reference[key]:14.2f}")
    print(f"(fixed action = {REFERENCE[0]:g} deg, K {REFERENCE[1]:g}, B {REFERENCE[2]:g})")


if __name__ == "__main__":
    main()
