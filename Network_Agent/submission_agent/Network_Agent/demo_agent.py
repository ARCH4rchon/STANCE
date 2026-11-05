"""
Demo of the learning agent: load or train a network, then compare it with the best fixed action.

Uses: agent.train (TRAIN, NETWORKS, make_wrappers, make_venv), common.env_setup,
common.evaluate and common.policies.
After the table, the first --show episodes are replayed in the MuJoCo viewer, the same way
as Gym/demo.py (on macOS run it with mjpython instead of python; --show 0 needs no display).
"""
# Libraries
import argparse
import importlib
import json
import tempfile
import threading
import time
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


def show_episodes(net, model, vn, wrappers, n, slow):
    """
    Replay the first n evaluation episodes (seeds 1000, 1001, ...) in the MuJoCo viewer, with the
    same policy and wrappers as the evaluation, on the training ground. Built like Gym/demo.py.
    """
    import mujoco
    import mujoco.viewer   # imported here so that --show 0 needs no display
    from stance_env.ankle_gym_env import FRAME_SKIP

    env = make_env("train", wrappers)
    policy = net.POLICY(model, vn)
    sim = env.unwrapped
    step_dt = sim.model.opt.timestep * FRAME_SKIP   # 5 ms of simulated time per env step
    seeds = EVAL_SEEDS[:n]
    print(f"\nshowing {len(seeds)} episodes in the MuJoCo viewer ({slow:g}x slow motion)")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Side view that follows the 70 kg body mass (as in Gym/demo.py).
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.body_bid
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 90.0, -8.0
        for i, seed in enumerate(seeds, 1):
            if not viewer.is_running():
                break
            obs, _ = env.reset(seed=int(seed))
            policy.reset()
            with viewer.lock():
                sim.ground.update_visual(sim.model)   # flat ground again for the new episode
            viewer.sync()
            ret, toe, done, term = 0.0, 0.0, False, False
            while not done and viewer.is_running():
                obs, r, term, trunc, info = env.step(policy.act(obs))
                ret += float(r)
                toe = max(toe, float(info["toe_force_bw"]))
                with viewer.lock():
                    sim.ground.update_visual(sim.model)   # sink and dent the ground cells
                viewer.sync()
                time.sleep(step_dt * slow)
                done = term or trunc
            if not done:
                break   # window closed mid-episode
            print(f"episode {i}/{len(seeds)} (seed {seed}): {'fell' if term else 'survived'}, "
                  f"return {ret:.2f}, toe peak {toe:.2f} BW")
            # Keep the final pose on screen for about 1 s.
            end = time.time() + 1.0
            while time.time() < end and viewer.is_running():
                time.sleep(0.05)
    # Leaving the block only asks the viewer to close. Wait for its window thread to finish,
    # otherwise Python's exit can shut GLFW down while that thread is still drawing (segfault).
    for t in threading.enumerate():
        if t.name.endswith("(_launch_internal)"):
            t.join(timeout=5)
    env.close()


def show_count(text):
    """--show: an integer from 0 to 10."""
    n = int(text)
    if not 0 <= n <= 10:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 10")
    return n


def main():
    """Parse the arguments, get a model, print its architecture, evaluate it and the reference."""
    p = argparse.ArgumentParser(description="Demo of the STANCE learning agent.")
    p.add_argument("--network", choices=sorted(NETWORKS), default="mlp", help="choice of network to train")
    p.add_argument("--model", metavar="DIR", help="saved run folder (model.zip, vecnormalize.pkl, config.json)")
    p.add_argument("--episodes", type=int, default=20, help="evaluation episodes (seeds 1000 and up)")
    p.add_argument("--train", action="store_true", help="train from scratch instead of loading a pre-trained model")
    p.add_argument("--steps", type=int, help="with --train: training steps")
    p.add_argument("--show", type=show_count, default=5, metavar="N",
                   help="episodes to replay in the MuJoCo viewer, 0-10 (0 = no viewer)")
    p.add_argument("--slow", type=float, default=10.0, metavar="X", help="viewer slow-motion factor")
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

    # Replay the first episodes live in the MuJoCo viewer.
    if a.show > 0:
        show_episodes(net, model, vn, wrappers, a.show, a.slow)


if __name__ == "__main__":
    main()
