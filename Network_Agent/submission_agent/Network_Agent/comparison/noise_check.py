"""Task D: check that sensor noise reaches the policy, after normalisation, in every frame.

Drafted with Claude Code (Anthropic), for the AI declaration.

For seed 0 of each network, one episode (seed 1000, training ground) is played twice:
  noise 0    -> the policy input must equal the clean normalised observation exactly;
  noise 1.0  -> at every step, the input the policy actually receives (VecNormalize applied, as
                in common.policies.SB3Policy._normalise) is compared with the clean observation
                of the same state passed through the same network wrappers and normalisation.
Printed: the normalised input at one step (noise 0 and 1.0 side by side for the newest frame),
the noisy-minus-clean difference for every stacked frame (MLP) / the single frame (LSTM), the
std of that difference per channel over the whole episode (expect about 1 x sensor std divided
by the VecNormalize std for channels 0-7, exactly 0 for the previous-action channels 8-10), and
a check that a frame's noise is carried unchanged as it moves back through the stack.

    python -m comparison.noise_check            -> also saved to results/noise_check.txt
"""
import io
import sys
from contextlib import redirect_stdout

import gymnasium as gym
import numpy as np

from comparison.harness import (RESULTS, SensorNoise, StepRecorder, discover_controllers,
                                load_network, sensor_std)
from common.env_setup import SENSOR_NAMES, make_env

STEP_SHOWN = 12          # after 9 steps all 10 stacked frames are real (no zero padding)
SEED = 1000
np.set_printoptions(precision=3, suppress=True, linewidth=150)


def through_wrappers(env, clean):
    """Apply the observation wrappers between SensorNoise and the outside, in order."""
    chain, e = [], env
    while not isinstance(e, SensorNoise):
        chain.append(e)
        e = e.env
    obs = clean
    for w in reversed(chain):
        if isinstance(w, gym.ObservationWrapper):
            obs = w.observation(obs)
    return obs


def play(ctrl, level, std):
    net, model, vn, wrappers, cfg = load_network(ctrl["run"])
    # Noise wrapper always present (level 0 adds exactly 0), so the clean obs is available.
    env = StepRecorder(make_env("train", [lambda e: SensorNoise(e, level, std)] + list(wrappers)))
    noise_env = env
    while not isinstance(noise_env, SensorNoise):
        noise_env = noise_env.env
    policy = net.POLICY(model, vn)
    obs, _ = env.reset(seed=SEED)
    policy.reset()
    got, clean, done = [], [], False
    while not done:
        got.append(policy._normalise(obs).copy())                  # what the network sees
        clean.append(vn.normalize_obs(through_wrappers(env, noise_env.clean)).copy())
        obs, r, term, trunc, info = env.step(policy.act(obs))
        done = term or trunc
    env.close()
    frames = cfg["frames"]
    keep = [i for i in range(len(SENSOR_NAMES)) if i not in cfg["drop_sensors"]]
    shape = (len(got), frames, len(keep))
    return np.array(got).reshape(shape), np.array(clean).reshape(shape), vn, keep, frames


def report(ctrl, std):
    name = f"{ctrl['controller']} s{ctrl['model_seed']}"
    g0, c0, vn, keep, frames = play(ctrl, 0.0, std)
    g1, c1, _, _, _ = play(ctrl, 1.0, std)
    names = [SENSOR_NAMES[i][:12] for i in keep]
    print(f"\n==================== {name}: {frames} frame(s) x {len(keep)} channels, episode seed {SEED}")
    print(f"noise 0  : max |policy input - clean normalised obs| over the episode = {np.abs(g0 - c0).max():.2e}")
    k = min(STEP_SHOWN, len(g1) - 1)
    print(f"\nnormalised newest frame at step {k} (rows: noise 0 run, noise 1.0 run, clean of the noise-1.0 state)")
    print("  " + " ".join(f"{n:>12s}" for n in names))
    for label, v in (("n=0 ", g0[min(k, len(g0) - 1), -1]), ("n=1 ", g1[k, -1]), ("cln ", c1[k, -1])):
        print(label + " ".join(f"{x:12.3f}" for x in v))
    d = g1 - c1
    print(f"\nnoise 1.0: policy input - clean (normalised units), step {k}, every stacked frame "
          "(row 0 = oldest):")
    print("  " + " ".join(f"{n:>12s}" for n in names))
    for f in range(frames):
        print(f"{f:2d}" + " ".join(f"{x:12.3f}" for x in d[k, f]))
    # Expected std of the normalised noise = 1.0 * sensor std / VecNormalize std of that feature.
    vstd = np.sqrt(vn.obs_rms.var).reshape(frames, len(keep))
    expect = (std[keep] / vstd)
    expect[:, [i for i, c in enumerate(keep) if c >= 8]] = 0.0
    real = d[frames - 1:] if frames > 1 else d       # skip the start-of-episode zero padding
    print(f"\nstd over the episode of (input - clean), steps {frames - 1}-{len(d) - 1}, per channel:")
    print("observed (newest frame) " + " ".join(f"{x:7.3f}" for x in real[:, -1].std(0)))
    print("expected (newest frame) " + " ".join(f"{x:7.3f}" for x in expect[-1]))
    if frames > 1:
        print("observed (oldest frame) " + " ".join(f"{x:7.3f}" for x in real[:, 0].std(0)))
        # noise in frame f at step t must equal noise in frame f-1 at step t+1 (same reading)
        carried = np.abs(d[1:, :-1] * vstd[:-1] - d[:-1, 1:] * vstd[1:]).max()
        print(f"same noise carried back through the stack: max mismatch {carried:.2e} (raw units)")
    zero = np.abs(d[..., [i for i, c in enumerate(keep) if c >= 8]]).max()
    print(f"previous-action channels: max |noise| = {zero:.2e} (should be 0)")


def main():
    std = sensor_std()
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("Task D noise sanity check (python -m comparison.noise_check). Drafted with Claude Code.")
        print("sensor std (raw units, channels 0-10):", np.array2string(std, precision=4))
        seen = set()
        for c in discover_controllers():
            if c["kind"] == "network" and c["controller"] not in seen:
                seen.add(c["controller"])
                report(c, std)
    text = buf.getvalue()
    sys.stdout.write(text)
    (RESULTS / "noise_check.txt").write_text(text)


if __name__ == "__main__":
    main()
