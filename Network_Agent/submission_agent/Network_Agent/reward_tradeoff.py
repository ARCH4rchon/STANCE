"""How the reward weights change which fixed action is best. OWNER: P3.

Runs the 45 fixed actions (angle x K x B) on the train ground, keeps every step's toe force
and the environment's reward, then re-scores each action offline under each pair
(W_IMPACT, LAMBDA_BW) with agent.reward.impact_penalty, the same formula ReweightedImpact uses:

    reward_new = reward_env + impact_penalty(toe, 2.94, 1.58) - impact_penalty(toe, W, L)

For each pair it takes the action with the highest mean return and plots, against W_IMPACT
(one line per LAMBDA_BW): (a) that action's survival, (b) its toe peak, and (c) its return
gap over the original best fixed action (-20 deg, 500, 10) scored under the same pair.
Raw returns are not plotted: returns under different weights are not on the same scale.

Commands (from Network_Agent/)
    python reward_tradeoff.py            200 episodes per action (seeds 1000-1199)
    python reward_tradeoff.py --quick    20 episodes per action
Output: figures/reward_tradeoff.png and .pdf, and the table behind the figure.
"""
import argparse
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.reward import impact_penalty
from common.env_setup import make_env, to_norm_action
from common.evaluate import EVAL_SEEDS, REFERENCE

GRID = list(itertools.product([-20.0, -10.0, 0.0], [50.0, 125.0, 250.0, 375.0, 500.0], [2.0, 6.0, 10.0]))
W_IMPACTS = [1.0, 2.94, 4.0, 6.0, 8.0, 12.0]
LAMBDAS = [1.0, 1.58, 2.5]
MARKS = {(2.94, 1.58): "original", (6.0, 1.58): "W6"}


# Stage 1: play one fixed action; per episode keep the step toe forces, env rewards and whether it fell
def run_action(args):
    action, seeds = args
    env = make_env("train")
    module = sys.modules[type(env.unwrapped).__module__]
    a = to_norm_action(env, *action)
    episodes = []
    for seed in seeds:
        env.reset(seed=seed)
        toe, rew, done, fell = [], [], False, False
        while not done:
            _, r, term, trunc, info = env.step(a)
            toe.append(float(info["toe_force_bw"])); rew.append(float(r))
            done, fell = term or trunc, term
        episodes.append((np.array(toe), np.array(rew), fell))
    env.close()
    return episodes, (module.W_IMPACT, module.LAMBDA_BW)


# Stage 2: mean return of one action under (w, lam), from its recorded steps
def rescore(episodes, orig, w, lam):
    return float(np.mean([sum(r + impact_penalty(t, *orig) - impact_penalty(t, w, lam) for t, r in zip(toe, rew))
                          for toe, rew, _ in episodes]))


def main():
    p = argparse.ArgumentParser(description="Reward trade-off figure for the 45 fixed actions.")
    p.add_argument("--quick", action="store_true", help="20 episodes per action instead of 200")
    seeds = EVAL_SEEDS[: 20 if p.parse_args().quick else 200]
    start = time.time()

    # Stage 1: run the grid, one process per action
    with Pool() as pool:
        results = pool.map(run_action, [(action, seeds) for action in GRID])
    data, orig = [r[0] for r in results], results[0][1]
    survival = [100 * np.mean([not fell for _, _, fell in eps]) for eps in data]
    toe_peak = [np.mean([toe.max() for toe, _, _ in eps]) for eps in data]
    ref = GRID.index(tuple(float(x) for x in REFERENCE))

    # Stage 2: for each weight pair, the best action and its gap over the reference action
    rows = []
    for lam, w in itertools.product(LAMBDAS, W_IMPACTS):
        returns = [rescore(eps, orig, w, lam) for eps in data]
        best = int(np.argmax(returns))
        rows.append((w, lam, GRID[best], survival[best], toe_peak[best], returns[best] - returns[ref]))

    # Stage 3: print the table behind the figure
    print(f"\n{len(seeds)} episodes per action, seeds {seeds[0]}-{seeds[-1]}, train ground; "
          f"env weights W_IMPACT {orig[0]}, LAMBDA_BW {orig[1]}")
    print(f"{'W_IMPACT':>8s} {'LAMBDA':>6s}  {'best action (deg, K, B)':>24s} {'surv %':>7s} {'toe BW':>7s} {'gap':>7s}")
    for w, lam, act, s, t, g in rows:
        tag = MARKS.get((w, lam), "")
        print(f"{w:8.2f} {lam:6.2f}  {f'{act[0]:g}, {act[1]:g}, {act[2]:g}':>24s} {s:7.1f} {t:7.3f} {g:7.2f}  {tag}")

    # Stage 4: plot (a) survival, (b) toe peak, (c) return gap against W_IMPACT, one line per LAMBDA_BW
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    titles = ["(a) survival of best action (%)", "(b) its toe peak (BW)", "(c) return gap over (-20, 500, 10)"]
    for col, (ax, title) in enumerate(zip(axes, titles)):
        for lam in LAMBDAS:
            pts = [(r[0], r[3 + col]) for r in rows if r[1] == lam]
            ax.plot(*zip(*pts), marker="o", label=f"LAMBDA_BW {lam:g}")
        for (w, lam), name in MARKS.items():
            r = next(r for r in rows if r[:2] == (w, lam))
            ax.plot(w, r[3 + col], marker="*", markersize=14, color="black", zorder=3)
            ax.annotate(name, (w, r[3 + col]), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_xlabel("W_IMPACT"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"reward_tradeoff.{ext}", dpi=150)
    print(f"\nsaved {out}/reward_tradeoff.png and .pdf ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
