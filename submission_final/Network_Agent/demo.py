"""
Demo of the STANCE learning agent: run the trained networks on the training and test grounds,
next to the best fixed action, and write the results to CSV files.

Drafted with Claude Code (Anthropic), for the AI declaration.

Uses: agent.train (TRAIN, NETWORKS, make_wrappers, make_venv), common.env_setup,
common.evaluate (run_episodes, the same protocol as every reported result) and common.policies.
After the evaluation, the first --show episodes are replayed in the MuJoCo viewer
(on macOS run it with mjpython instead of python; --show 0 needs no display).
--record saves one episode to an .mp4 file without opening a window.
"""
# Libraries
import argparse
import csv
import importlib
import json
import os
import sys
import tempfile
import threading
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local imports
from agent.train import NETWORKS, TRAIN, make_wrappers, make_venv
from common.env_setup import make_env, to_norm_action
from common.evaluate import REFERENCE, run_episodes
from common.policies import FixedPolicy
from stance_env.ankle_gym_env import FRAME_SKIP

TRAINED = Path(__file__).resolve().parent / "trained_networks"
# Model seeds of the saved runs, e.g. final_s0, final_s1, final_s2 -> ["0", "1", "2"]
MODEL_SEEDS = sorted({p.name[len("final_s"):] for p in TRAINED.glob("*/final_s*")}, key=int)


def load_run(run_dir):
    """Load a saved run folder: its config, the network's wrappers, the model and the VecNormalize.
    The run folder must contain model.zip, vecnormalize.pkl and config.json."""
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
    """Train a network from scratch with the settings of agent/train.py, in a temporary folder
    (never inside trained_networks/)."""
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


# --- viewer start ---
def show_episodes(net, model, vn, wrappers, n, slow, first_seed=1000):
    """Replay n episodes (seeds first_seed, first_seed + 1, ...) in the MuJoCo viewer, in slow motion."""
    env = make_env("train", wrappers)
    policy = net.POLICY(model, vn)
    sim = env.unwrapped                                   # the AnkleEnv inside the wrappers
    step_dt = sim.model.opt.timestep * FRAME_SKIP         # 5 ms of simulated time per env step
    print(f"\nshowing {n} episodes in the MuJoCo viewer ({slow:g}x slow motion)")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Side view that follows the body mass, as in Gym/demo.py.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.body_bid
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 90.0, -8.0

        for i, seed in enumerate(range(first_seed, first_seed + n), 1):
            obs, _ = env.reset(seed=seed)
            policy.reset()
            ret, toe, done = 0.0, 0.0, False
            while not done and viewer.is_running():
                obs, r, term, trunc, info = env.step(policy.act(obs))
                ret += r
                toe = max(toe, info["toe_force_bw"])
                with viewer.lock():
                    sim.ground.update_visual(sim.model)   # sink and dent the ground under the foot
                viewer.sync()
                time.sleep(step_dt * slow)
                done = term or trunc
            if not viewer.is_running():
                break                                     # window closed
            print(f"episode {i}/{n} (seed {seed}): {'fell' if term else 'survived'}, "
                  f"return {ret:.2f}, toe peak {toe:.2f} BW")
            time.sleep(1.0)                               # keep the final pose on screen

    # Let the viewer's own thread finish, otherwise Python can crash while it closes.
    for t in threading.enumerate():
        if t.name.endswith("(_launch_internal)"):
            t.join(timeout=5)
    env.close()
# --- viewer end ---


def record_episode(net, model, vn, wrappers, seed, path):
    """Render one episode offscreen (one frame per env step) and save it as an mp4 at 30 fps."""
    import imageio
    env = make_env("train", wrappers)
    policy = net.POLICY(model, vn)
    sim = env.unwrapped
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    # Same tracking side view as the viewer.
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = sim.body_bid
    cam.distance, cam.azimuth, cam.elevation = 2.0, 90.0, -8.0

    frames = []
    obs, _ = env.reset(seed=seed)
    policy.reset()
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        sim.ground.update_visual(sim.model)
        renderer.update_scene(sim.data, camera=cam)
        frames.append(renderer.render())
        done = term or trunc
    renderer.close()
    env.close()
    imageio.mimsave(path, frames, fps=30)


def mean_row(rows, network, model_seed, ground):
    """One summary row: mean return, fall rate (%) and mean toe peak (BW) over the given episodes."""
    return {"network": network, "model_seed": model_seed, "ground": ground,
            "episodes": len(rows),
            "mean_return": round(np.mean([r["return"] for r in rows]), 4),
            "fall_rate_pct": round(100 * np.mean([r["fell"] for r in rows]), 2),
            "toe_peak_bw": round(np.mean([r["toe_peak_bw"] for r in rows]), 4)}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Demo of the STANCE learning agent.")
    p.add_argument("--network", choices=["mlp", "lstm", "both"], default="both", help="network(s) to run")
    p.add_argument("--model-seed", choices=MODEL_SEEDS + ["all"], default="all",
                   help="which trained_networks/<network>/final_s<N> run(s) to load")
    p.add_argument("--episodes", type=int, default=200, help="evaluation episodes per ground")
    p.add_argument("--seed", type=int, default=1000, help="first episode seed (episodes use S, S+1, ...)")
    p.add_argument("--out", default="results/demo_episodes.csv", help="per-episode CSV file")
    p.add_argument("--show", type=int, default=5, metavar="N (0-10) must be a number")
    p.add_argument("--slow", type=float, default=10.0, metavar="X", help="viewer slow-motion factor")
    p.add_argument("--record", metavar="PATH.mp4", help="save one episode (seed --seed) as a video")
    p.add_argument("--train", action="store_true", help="train from scratch instead of loading the trained networks")
    p.add_argument("--steps", type=int, help="with --train: training steps")
    a = p.parse_args()

    networks = ["mlp", "lstm"] if a.network == "both" else [a.network]
    model_seeds = MODEL_SEEDS if a.model_seed == "all" else [a.model_seed]
    seeds = list(range(a.seed, a.seed + a.episodes))

    # Get the models: the saved runs, or freshly trained ones (labelled model seed "new").
    runs = []
    for name in networks:
        if a.train:
            runs.append((name, "new", train_run(name, a.steps)))
        else:
            folder = NETWORKS[name].split(".")[0]          # e.g. networkA_mlp
            for ms in model_seeds:
                runs.append((name, ms, load_run(TRAINED / folder / f"final_s{ms}")))

    # Evaluate every run and the best fixed action on both grounds, on the same seeds.
    print(f"Please wait, evaluating on {len(seeds)} episodes per ground (seeds {seeds[0]}-{seeds[-1]})...")
    episodes = []
    for ground in ["train", "test"]:
        env = make_env(ground)
        eps = run_episodes(FixedPolicy(to_norm_action(env, *REFERENCE)), env, seeds)
        env.close()
        episodes += [("reference", "-", ground, e) for e in eps]
        for name, ms, (net, model, vn, wrappers) in runs:
            env = make_env(ground, wrappers)
            eps = run_episodes(net.POLICY(model, vn), env, seeds)
            env.close()
            episodes += [(name, ms, ground, e) for e in eps]

    rows = [{"network": name, "model_seed": ms, "ground": ground, "episode_seed": e["seed"],
             "return": round(e["ret"], 6), "fell": e["terminated"], "steps": e["steps"],
             "toe_peak_bw": round(e["toe_peak"], 6)}
            for name, ms, ground, e in episodes]

    # Summary: one row per network, model seed and ground, then the mean over model seeds.
    summary = []
    for name in ["reference"] + networks:
        for ground in ["train", "test"]:
            mine = [r for r in rows if r["network"] == name and r["ground"] == ground]
            per_seed = [mean_row([r for r in mine if r["model_seed"] == ms], name, ms, ground)
                        for ms in dict.fromkeys(r["model_seed"] for r in mine)]
            summary += per_seed
            if len(per_seed) > 1:
                summary.append({"network": name, "model_seed": "mean", "ground": ground,
                                "episodes": sum(s["episodes"] for s in per_seed),
                                **{k: round(np.mean([s[k] for s in per_seed]), 4)
                                   for k in ["mean_return", "fall_rate_pct", "toe_peak_bw"]}})

    out = Path(a.out)
    out_summary = out.with_name(out.stem + "_summary" + out.suffix)
    write_csv(out, rows)
    write_csv(out_summary, summary)
    print(f"episodes: {out}\nsummary:  {out_summary}")
    for name in ["reference"] + networks:
        last = {s["ground"]: s for s in summary if s["network"] == name}   # mean row if there is one
        print(f"{name:9s}  train: return {last['train']['mean_return']:6.2f}, falls {last['train']['fall_rate_pct']:5.1f}%"
              f"   test: return {last['test']['mean_return']:6.2f}, falls {last['test']['fall_rate_pct']:5.1f}%")

    # Optional video of one episode. It must never stop the demo.
    name, ms, first = runs[0]
    if a.record:
        try:
            t0 = time.time()
            record_episode(*first, a.seed, a.record)
            print(f"recorded {name} (model seed {ms}), seed {a.seed}, to {a.record} in {time.time() - t0:.1f} s")
        except Exception as err:
            print(f"recording skipped ({type(err).__name__}: {err})")

    # Replay the first episodes live in the MuJoCo viewer.
    if a.show > 0:
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("no display found, skipping the viewer (use --show 0 to hide this message)")
        else:
            show_episodes(*first, a.show, a.slow, a.seed)


if __name__ == "__main__":
    main()
