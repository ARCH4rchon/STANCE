# Controller comparison: notes

Drafted with Claude Code (Anthropic), for the AI declaration. Branch `Control_Comparison_Data`.

Evaluation only: nothing was trained. Nothing in `Gym/`, `common/`, `agent/`, `demo_agent.py` or
`trained_networks/` was edited. All episodes go through `common.evaluate.run_episodes` and
`common.env_setup.make_env` unchanged. Networks are loaded with `agent.train.make_wrappers`, the
same way as `demo_agent.load_run`.

## Reproduce (from `submission_agent/Network_Agent/`, conda env `stance`)

```bash
# smoke test (20 episodes, separate folder, not committed)
python -m comparison.compare    --episodes 20 --out comparison/results/smoke
python -m comparison.stats      --in comparison/results/smoke/episodes.csv
python -m comparison.robustness --episodes 20 --out comparison/results/smoke

# full run at nice 10, appends to comparison/results/log.txt; rerun to resume
bash comparison/run_all.sh "$(conda run -n stance which python)"

# or step by step
python -m comparison.compare                  # Task A, 200 episodes -> results/episodes.csv
python -m comparison.stats                    # -> results/summary.csv, results/summary.md
python -m comparison.robustness               # Task B, 100 episodes/cell -> robustness*.csv/.png/.pdf
python -m comparison.robustness --plot-only   # redraw the figure from robustness_episodes.csv
```

If ROS sets `PYTHONPATH`, prefix with `env -u PYTHONPATH` (`run_all.sh` already does).
- **Resume:** each controller × condition cell is saved atomically to `results/cells/` (not committed).
  On a rerun, a cell is skipped once its file holds every requested seed. The merged CSVs are then rebuilt.
- **Parallelism:** 14 worker processes (CPU count − 2) via `--workers`, with one torch thread each.

## Files

| file | content |
|---|---|
| `harness.py` | controller discovery, model loading, extra wrappers, cell runner, resume |
| `compare.py`, `stats.py` | Task A run and statistics |
| `robustness.py` | Task B sweeps and figure |
| `run_all.sh` | full run, nice 10, logged |
| `results/episodes.csv` | Task A, one row per episode (3,200 rows) |
| `results/summary.csv`, `summary.md` | Task A tables |
| `results/robustness_episodes.csv` | Task B, one row per episode (8,800 rows, about 1 MB) |
| `results/robustness.csv` | Task B per cell plus mean ± sd over seeds |
| `results/robustness.png`, `.pdf` | return vs ground stiffness |
| `results/log.txt` | run log |

For the AI declaration: every file in `comparison/` was drafted with Claude Code. The `.csv`, `.png`
and `.pdf` files in `results/` are written by those scripts. They carry no header line because a header
would break CSV readers.

### Per-episode columns

- **Basic outcome:** `controller`, `model_seed` (empty for fixed actions), `episode_seed`, `ground`, `return`, `fell`, `steps`, `toe_peak_bw`.
- **Extra outcome:** `total_peak_bw` and `fall_reason` (`forward`/`back` from the shank tilt, as in `common/evaluate`).
- **Reward components:**
  - The env exposes only the toe force in `info`, not the reward terms themselves.
  - `impact_sum` is recomputed each step from `info["toe_force_bw"]` using the env's own `W_IMPACT` and `LAMBDA_BW`.
  - `fall_penalty` is −25 × fell.
  - `smooth_sum` is the remainder: return − steps − impact − fall.
- **Action and ground:** `mean_theta_deg`, `mean_K`, `mean_B` are the applied action, averaged over the episode. `k0` is the ground stiffness the env reported. `noise` is the sensor-noise level.
- **Task B only:** `sweep` and `level`.

## Controllers and seeds found

The scripts glob `trained_networks/*/final_s*/`, so extra seeds are picked up automatically. Found:

- `reference` = (-20°, 500, 10). This is `common.evaluate.REFERENCE`.
- `best_fixed` = (-20°, 500, 6). **Not confirmed. See flag 1 below.**
- `networkA_mlp` seeds 0, 1, 2 (MLP, 10 frames).
- `networkB_lstm` seeds 0, 1, 2 (LSTM, 1 frame).

Both networks are residual policies around the anchor (-14°, 250.6, 6.38), not around the reference.

**Check of the harness:** for the reference and for two network runs, the first 20 episodes match
`results/reference_train_1000-1199.csv` and `trained_networks/*/episodes_{train,test}.csv` to
within 5e-5 in return (CSV rounding). The harness therefore reproduces the official protocol.

## Run times (16-CPU machine, 14 workers, nice 10)

| step | content | wall time |
|---|---|---|
| Task A | 16 cells × 200 episodes | 54 s |
| Task B | 88 cells × 100 episodes | 124 s |
| smoke runs | 20 episodes | 10 s (A) and 35 s (B) |
| `run_all.sh` | total | 3 min 5 s |

## Task B design

**Stiffness sweep, done without touching `Gym/`.**
- `GroundModel.sample()` draws `k0` inside `reset()`.
- The reset observation does not depend on `k0`, because the ground-force channel is 0 until the first step.
- So `harness.GroundStiffness` sets `ground.k0` right after reset. It recomputes the damping `c` with the same formula as `sample()`: zeta keeps its drawn value, and the `DAMP_LIMIT` cap is kept.
- The other ground parameters (zeta, alpha, f_yield, mu) and the start pose keep their seeded draws from the **training** ranges, so only k0 changes.
- The k0 value the env reported matched the set level in every episode.

| k0 (N/m) | relation to the ranges in `ground.py` |
|---|---|
| 1e3 | beyond the test soft band (half its minimum) |
| 3.2e3 | inside the test soft band (2e3-5e3) |
| 7e3 | training range (5e3-1e5), soft end |
| 2.2e4 | training range, log-centre |
| 7e4 | training range, hard end |
| 1.4e5 | inside the test hard band (1e5-2e5) |
| 3e5 | beyond the test hard band (1.5× its maximum) |
| 6e5 | far beyond (3× the test maximum) |

The `test` ground in Task A also uses different alpha, f_yield and mu ranges. That makes it a harder,
combined stress, so it is not the same thing as the k0-only sweep.

**Sensor-noise sweep.** This was clean, so it was run.
- `harness.SensorNoise` adds Gaussian noise to sensor channels 0-7. Channels 8-10 are the previous action and get none.
- The noise std is 0, 0.05, 0.1 or 0.2 × the sensor's std.
- One per-sensor std vector is used for every network. It is the newest-frame std from each run's `VecNormalize` stats, averaged over the 6 runs. The vector is printed in `log.txt`.
- Each new frame gets its noise once and keeps it as it moves through the 10-frame stack.
- The noise generator is seeded from the episode seed, so every controller gets the same draws.
- The run uses the training ground, 100 episodes per cell.
- Fixed actions ignore the sensors, so they would give identical results at every noise level. They were not rerun.
- The figure shows the stiffness sweep only, as specified. The noise results are in `robustness.csv` (`sweep == "noise"`).

## Skipped or flagged

1. **`best_fixed` = (-20, 500, 6) is not confirmed.**
   - The grid-sweep output was never saved. `results/P3_findings.md` §1 reads "Output missing… no script remains". `reward_tradeoff.py` can rebuild the grid but only prints it, and I did not rerun it.
   - The action is kept as specified.
   - On these seeds it is **not** clearly better than the reference: +0.26 [−0.38, +0.96] on train and −1.24 [−3.08, +0.32] on test. P3_findings §6b also has it at −0.09 vs the reference.
   - It does raise the toe peak (1.835 vs 1.739 BW).
2. **No reward-component columns from the env itself.** The env does not put its reward terms in `info`, so they are recomputed (see above). Exposing them directly would need `ankle_gym_env.step()` to add, for example, `info["r_impact"]`, `info["r_smooth"]` and `info["r_fall"]`. That is a `Gym/` change, so I did not make it. Because `smooth_sum` is a remainder, it cannot be checked on its own. As a sanity check, it is never positive beyond float rounding (largest value 1.4e-13).
3. **Toe peak on `test` is capped.** Test f_yield is 400-590 N per point, below 1.58 BW (about 1141 N). The toe force therefore saturates at f_yield, every controller shows the same 0.704 BW, and the impact term is 0.
4. **Bootstrap intervals for the "mean over seeds" rows** come from the per-episode, seed-averaged difference. They cover episode variation only. The seed spread is reported separately as the sd.

## Surprising results

- **Higher return, more falls.** On the training ground, both networks get a higher return than the reference, but they fall more often:
  - MLP: +4.2 pts [+1.3, +7.2]
  - LSTM: +2.3 pts [+0.5, +4.3]

  The extra return comes from lower toe peaks (impact term), not from survival. Per-seed, MLP s1/s2 and LSTM s1 each fall significantly more than the reference.
- **The test ground favours the reference.**
  - LSTM mean Δ return is −1.46 [−2.82, +0.06], and its fall rate is +6.2 pts [+2.8, +9.5].
  - This comes from seeds 1 and 2, each about +10 pts. LSTM s0 is level with the reference.
  - The MLP is level with the reference (−0.31 [−1.90, +1.57]).
- **Stiffness sweep: the advantage is on hard ground only.**
  - From 7e4 to 3e5 the networks beat both fixed actions. The LSTM leads the better fixed action by +10, +13 and +11 return; the MLP leads by +8, +9.5 and +4. The gap holds into and past the test hard band.
  - On soft ground (1e3-7e3) the networks do slightly worse and fall more. At 1e3: MLP 44%, LSTM 41%, reference 35%.
- **At 6e5 N/m the MLP collapses to the reference level.**
  - Per seed: −3.4, +18.0 and −10.6 (mean 1.3 ± 14.9).
  - (-20, 500, 6) is best there at 19.5, probably because lower damping softens the impact.
  - Falls are not the cause, since every controller falls in ≤ 7% of episodes. Almost all of the loss is the impact term.
- **Sensor noise up to 0.2 std has no measurable effect** on either network. Return and fall-rate changes are well inside the seed sd: MLP 89.29 → 89.12, LSTM 91.13 → 90.93. Together with the w6 "blind" finding in the redesign branch, this suggests the policies use only coarse features of the sensor history.
