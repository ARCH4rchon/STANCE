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
| `results/robustness_episodes.csv` | Task B (and the Task D noise levels), one row per episode (10,600 rows, 1.3 MB) |
| `results/robustness.csv` | Task B per cell plus mean ± sd over seeds |
| `results/robustness.png`, `.pdf` | return vs ground stiffness |
| `results/log.txt` | run log |
| `tune.py` | Task C: fixed-action grid and best_fixed_tuned |
| `results/grid.csv`, `grid_heatmap.png/.pdf` | Task C grid, 80 cells |
| `results/best_fixed_tuned.json` | Task C: the chosen action and the rule used |
| `results/episodes_tuned.csv` | Task C: best_fixed_tuned on seeds 1000-1199, both grounds |
| `noise_check.py`, `results/noise_check.txt` | Task D: noise reaches the policy |
| `actions.py` | Task E |
| `results/actions.csv` | Task E, per step (119,036 rows, 5.5 MB) |
| `results/actions_time.png/.pdf`, `actions_early.png/.pdf` | Task E figures |
| `results/actions_slopes.csv` | Task E: per-seed trends against k0 |
| `run_cde.sh` | Tasks C-E full run, nice 10, logged |

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

1. **`best_fixed` = (-20, 500, 6) is not confirmed.** (Superseded by Task C. The row is kept, and the grid picks (-20, 500, 8).)
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
- **Sensor noise up to 0.2 std has no measurable effect** on either network (see Task D for larger levels). Return and fall-rate changes are well inside the seed sd: MLP 89.29 → 89.12, LSTM 91.13 → 90.93. Together with the w6 "blind" finding in the redesign branch, this suggests the policies use only coarse features of the sensor history.

---

# Tasks C-E (second round, same branch)

Nothing outside `comparison/` was edited, and nothing was trained. Earlier results were not overwritten:
- `episodes.csv` is unchanged.
- `best_fixed_tuned` episodes are written to `episodes_tuned.csv` instead.
- `stats.py` now reads both files. A diff of `summary.md` and `summary.csv` before and after shows only the two added rows and one footnote, so every earlier number is byte-identical.
- `robustness.csv` gains the new noise rows; its earlier rows are unchanged.

## Reproduce

```bash
# smoke (20 episodes, results/smoke/)
python -m comparison.tune --episodes 20 --eval-episodes 20 --out comparison/results/smoke
python -m comparison.stats --in comparison/results/smoke/episodes.csv comparison/results/smoke/episodes_tuned.csv
python -m comparison.robustness --episodes 20 --out comparison/results/smoke
python -m comparison.actions --episodes 20 --out comparison/results/smoke

# full (nice 10, appends to results/log.txt, resumable)
bash comparison/run_cde.sh "$(conda run -n stance which python)"
#   = tune (100 tuning eps/cell, 200 eval eps) -> stats -> noise_check -> robustness (100) -> actions (200)
```

## Run times (16 CPUs, 14 workers, nice 10)

| step | content | wall time |
|---|---|---|
| Task C grid | 80 cells × 100 episodes | 77 s |
| Task C evaluation | best_fixed_tuned, 2 × 200 episodes | 11 s |
| Task D | noise check | 3 s |
| Task D noise sweep | 18 new cells × 100 episodes | 34 s |
| Task E | 6 runs × 200 episodes | 27 s |
| `run_cde.sh` | total | 2 min 43 s |

## Task C: best fixed action

**Grid adjusted to the valid action range.**
- The env limits are θ ∈ [-20, 20]°, K ∈ [5, 500] and B ∈ [0.5, 10].
- The requested values -26°, K 625 and B 12 would be clipped onto an edge and duplicate other cells.
- They were replaced by -17°, K 440 and B 2.
- Grid: θ {-20, -17, -14, -8} × K {250, 375, 440, 500} × B {2, 4, 6, 8, 10} = 80 cells.
- Tuning seeds 2000-2099 (training ground) do not overlap the evaluation seeds 1000-1199.

**Selection rule:** highest mean return among cells whose fall rate is ≤ the reference's fall rate on the tuning seeds (6.0%). Twelve cells qualify.

**Result: `best_fixed_tuned` = (-20°, 500, 8).**
- Tuning return is 89.19 against the reference's 88.90. Fall rate is 6% for both.
- It ranks 1st of all 80 cells even without the fall-rate filter.
- The old (-20, 500, 6) ranks 5th on the tuning seeds (88.39).

On the standard seeds, compared with the reference (summary.md):

| ground | Δ return [95%] | Δ fall, pts [95%] | toe peak (BW) |
|---|---|---|---|
| train | +0.35 [−0.22, +0.97] | −0.5 [−2.5, +1.0] | 1.778 |
| test | −0.76 [−1.89, +0.10] | +1.0 [−1.0, +3.0] | n/a |

The toe peak on train is 1.778 BW, against 1.739 for the reference and 1.835 for (-20, 500, 6). Neither difference vs the reference is significant.

Grid findings:
- The return surface is flat near the top. The best three cells are within 0.3 return of each other, which is well inside the seed-to-seed noise.
- 9 of the top 10 cells sit at θ = −20°, the plantarflexion limit of the action range. Most of the top cells also have K at or near the K limit of 500.
- **The fixed-action optimum is pinned to the corner of the action space.** A better fixed action may exist outside it, but that cannot be tested without changing `ANKLE_RANGE`/`K_RANGE` in `Gym/`.
- The reference (-20, 500, 10) is effectively the best fixed action. The networks' margins in Task A are therefore measured against a near-optimal fixed baseline.

## Task D: noise sanity check

`results/noise_check.txt` has the printout for MLP s0 and LSTM s0 on episode seed 1000. **The noise reaches the policy correctly, so no fix was needed.**
- **Noise 0:** the policy's normalised input equals the clean normalised observation exactly (max difference 0).
- **Noise 1.0:** at every step the input differs from the clean observation of the same state in channels 0-7.
- **Frames reached:** the noise appears in all 10 stacked frames for the MLP and in the single frame for the LSTM.
- **Size after normalisation:** the std of the difference matches the expected 1.0 × sensor std ÷ VecNormalize std, about 0.9-1.06 against an expected 0.95-1.09.
- **The previous-action channels 8-10 get no noise.**
- **Each reading keeps its noise as it moves back through the MLP stack** (mismatch 1e-4, which is float32 rounding).

To print this, `harness.SensorNoise` now also stores the clean observation (`self.clean`). The noisy output is unchanged, and the earlier noise cells were not rerun.

Extended sweep, training ground, 100 episodes per cell, mean ± sd over 3 seeds:

| noise (× sensor std) | 0 | 0.2 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| MLP return | 89.29 ± 0.75 | 89.12 ± 0.53 | 87.86 ± 0.67 | 84.84 ± 1.58 | 78.05 ± 0.36 |
| MLP fall % | 10.7 | 10.7 | 11.7 | 12.7 | 17.3 |
| LSTM return | 91.13 ± 1.17 | 90.93 ± 0.96 | 90.62 ± 0.63 | 89.51 ± 0.35 | 87.01 ± 0.24 |
| LSTM fall % | 9.0 | 9.3 | 9.7 | 10.7 | 11.7 |

Findings:
- **Performance degrades from about 0.5 std.** The earlier "no effect" result held only because 0.2 std is small.
- **The MLP is about 2.5-3× more sensitive than the LSTM.** At 2.0 std it loses 11.2 return against 4.1 for the LSTM, and its fall rate rises by 6.7 pts against 2.7.
- A plausible reason is that the MLP sees 10 independently noisy frames, while the LSTM filters one frame at a time through its memory.
- **At 2.0 std the MLP (78.0) falls well below the fixed reference, while the LSTM (87.0) matches it.** On the same seeds 1000-1099 the reference scores 87.06 return with an 8.0% fall rate. The fixed actions are unaffected by sensor noise.

## Task E: what did the networks learn?

Per-step applied actions (after the residual offset and clipping) for all 6 runs, seeds 1000-1199, training ground. The data and figures are:
- `actions.csv`
- `actions_time.png`: means over the episode in 4 log-k0 bins. The line is the mean over seeds and the band is the sd over seeds.
- `actions_early.png`: the mean over steps 0-19 against k0.
- `actions_slopes.csv`: per-seed least-squares slope against log10 k0.

Late steps average only the episodes still running.

**Summary (5 lines):**
1. **K: yes, both networks lower K on harder ground, in all 6 seeds.** Over steps 0-19 the slope is −42 to −107 N·m/rad per decade of k0 (r = −0.27 to −0.58). The binned mean falls from about 410 to 320 (MLP) and from 450 to 385 (LSTM) between the softest and hardest training stiffness.
2. **B: no, they do not raise B on hard ground.** Slopes have mixed signs (−1.8 to +1.1 per decade), and MLP s1 clearly lowers B. B mainly rises over the episode, from about 6 to 8-9, whatever the stiffness.
3. **The adaptation is confined to weight acceptance.** The K dip on hard ground happens in steps 5-30 (25-150 ms). By step 40, every seed in every bin is back near K 500, and θ falls toward −20° by about step 60. Late stance therefore converges to the reference-like stiff corner.
4. **θ shifts slightly toward less plantarflexion on hard ground** in 5 of 6 seeds (+0.7 to +3.6° per decade). MLP s2 goes the other way.
5. **The direction of the K schedule is consistent across seeds, but its level is not.** LSTM s0 stays near (−19.7°, K ≈ 440-480), almost a fixed action, while the other seeds sit around −13.6° to −17° with early K 275-480 (bin means). So the seeds share a strategy (soften on hard ground, then stiffen) but have found different operating points.

This matches Tasks A and B. The learned gain over fixed actions is a short, stiffness-dependent drop in K at impact. That pays off on hard ground through lower toe peaks, and costs falls on soft ground, where a lower K lets the leg fold.

## Skipped / caveats (C-E)

- **Grid values outside the action range were replaced** (see Task C). That is a documented substitution, not a skip.
- **The fixed-action optimum lies on the θ/K limits.** Searching beyond them would need the action ranges in `Gym/stance_env/ankle_gym_env.py` to be widened. That was not done.
- **`actions.csv` is 5.5 MB.** It was committed as requested. If the repo should stay small, it can be regenerated in about 30 s with `python -m comparison.actions` and added to `.gitignore`.

---

# Hand rules: is it the K dip, or adapting it to the ground? (third round)

Nothing outside `comparison/` was edited, and nothing was trained. Earlier rows were not overwritten. Before/after diffs of `summary.md`, `summary.csv` and `robustness.csv` show only added lines, and `robustness_episodes.csv` keeps its earlier content byte-for-byte as a prefix.
- **New files:** `sensing.py`, `rules.py`, `run_rules.sh`, and in `results/`: `sensing.csv`, `tuning_rules.csv`, `rules_tuned.json`, `episodes_rules.csv`, `summary_rules_vs_networks.csv`.
- **Changed files:**
  - `harness.py`: runs `kind: "rule"` controllers, and rule cell IDs include their parameters.
  - `stats.py`: rule rows, a rules-vs-networks table, and the `(privileged)` label.
  - `robustness.py`: rules in the stiffness sweep; fixed actions are drawn dotted, rules dashed and networks solid.

## Reproduce

```bash
python -m comparison.sensing --episodes 20 --out comparison/results/smoke     # smoke
python -m comparison.rules --episodes 20 --eval-episodes 20 --out comparison/results/smoke
bash comparison/run_rules.sh "$(conda run -n stance which python)"           # full, nice 10, resumable
#   = sensing (100) -> rules: tune on 2000-2099 + evaluate on 1000-1199 -> stats -> robustness (100)
```

## Run times

16 CPUs, 14 workers, nice 10. Total for `run_rules.sh`: 7 min 43 s.

| step | cells × episodes | wall time |
|---|---|---|
| sensing | 100 episodes | 6 s |
| R1 tuning | 209 cells × 100 | 3 min 20 s |
| R2 tuning | 147 cells × 100 | 2 min 28 s |
| R3 tuning | 49 cells × 100 | 51 s |
| evaluation | 6 cells × 200 | 16 s |
| stiffness sweep | 24 new cells × 100 | 28 s |

## Step 1: what can the sensors tell? (`sensing.py`, `results/sensing.csv`)

**The 11 channels.** `common/inputs.py` only selects frames and channels; the names come from `env_setup.SENSOR_NAMES`:

| channel | name | source |
|---|---|---|
| 0 | ankle_angle | encoder |
| 1 | ankle_rate | encoder |
| 2 | ankle_torque | motor current |
| 3 | vertical_grf | pylon load cell, total normal force |
| 4 | pylon_compression | linear sensor |
| 5 | shank_tilt | IMU |
| 6 | shank_tilt_rate | IMU |
| 7 | vertical_accel | IMU |
| 8-10 | previous action | memory, not sensors |

**What is not sensed.** There is no toe-force or sinkage sensor. Toe force exists only in `info`, so it is reported as info-only and never used. Contact happens at the start of every episode, because reset places the heel on the ground.

**Method.** The reference controller ran on tuning seeds 2000-2099 (training ground). For causal features over frames 0..N (N × 5 ms after contact), I computed:
- Spearman ρ against log10(true k0);
- R² of a one-variable line, on the feature or its log, whichever fits better.

**Result: the sensors predict ground stiffness only moderately. No single early feature explains more than about 45% of the variance in log stiffness.**

| available by | best feature | ρ | R² |
|---|---|---|---|
| 5 ms | shank tilt rate | 0.60 | 0.32 |
| 15 ms | max vertical-GRF rate | 0.61 | 0.37 |
| 25 ms | pylon compression (min) | −0.66 | 0.45 |
| 50 ms | max GRF rate | 0.60 | 0.36 |
| 100 ms | max GRF rate | 0.64 | 0.40 |
| 150 ms | pylon compression | +0.66 | 0.44 |

The info-only toe-force peak suggested in the brief is weaker (ρ 0.44-0.52, R² ≤ 0.28), so it would not have helped even if it were sensed.

**Feature chosen for R2: the running max of the vertical-GRF rate.**
- It is useful from 15 ms and stable up to 150 ms.
- It is monotone, so it is causal at every step.
- It is physically meaningful: stiffer ground gives a faster force rise.
- Pylon compression scores slightly higher at 25 ms, but its sign reverses later in stance, so it was not used.

## Step 2: the three rules (`rules.py`)

- **What all three share:**
  - They hold θ_d = −20° and B = 8, and keep K = 500 outside a window after contact.
  - They use the same interface as `FixedPolicy` (`reset()`, `act(obs)`) and output normalised actions.
- **Tuning:**
  - Grid search only (no scipy in the env, so no Nelder-Mead), on seeds 2000-2099, training ground.
  - Constraint: fall rate ≤ the reference's on the same seeds (6%).
  - R1 was tuned first; R2 and R3 then reuse R1's window.
  - All 405 cells are in `tuning_rules.csv`.

| rule | form | grid | chosen | tuning return / fall |
|---|---|---|---|---|
| R1 timed_dip | K = K_dip in [start, end) ms | K_dip {25…500} × start {0, 10, 25, 50} × end {50…300} | K_dip 250, 50-100 ms | 90.49 / 5% |
| R2 sensed_dip | K_dip = a + b (z − 5.1), z = log10 max GRF rate over frames ≤ min(t, m) | a {50…400} × b {−600…200} × m {2, 3, 5} | a 250, b −100, m 5 | 90.72 / 5% |
| R3 oracle_dip (**privileged**) | K_dip = a + b (log10 k0 − 4.35), true k0 | a {50…400} × b {−300…100} | a 300, b −200 | 90.84 / 5% |

On the same tuning seeds the reference scores 88.90 / 6%.

What the chosen parameters mean:
- **R2's slope is too small to matter.** z varies by only about ±0.25, so R2's K_dip spans just 250 ± 25: it is effectively R1.
- **R3 makes a real choice:** K_dip runs from about 430 on the softest training ground down to about 170 on the hardest. It is lower K on harder ground, the same direction as the networks.

## Step 3: evaluation (seeds 1000-1199; `summary.md`, `robustness.csv`/`.png`)

Paired Δ vs the reference [95% bootstrap]:

| controller | train Δ return | train Δ fall, pts | test Δ return | test Δ fall, pts |
|---|---|---|---|---|
| R1 timed_dip | **+1.52 [+0.36, +2.78]** | +1.0 [−1.5, +3.5] | +0.13 [−0.82, +1.26] | +0.5 |
| R2 sensed_dip | **+1.67 [+0.45, +3.00]** | +1.5 [−1.0, +4.0] | +0.03 | +0.5 |
| R3 oracle_dip (privileged) | **+2.46 [+1.22, +3.81]** | +0.0 [−2.0, +2.0] | +0.17 | −0.5 |
| MLP (mean of 3 seeds) | +1.10 [−0.86, +3.24] | **+4.2 [+1.3, +7.2]** | −0.31 | +2.5 |
| LSTM (mean of 3 seeds) | **+2.97 [+1.24, +4.90]** | **+2.3 [+0.5, +4.3]** | −1.46 | **+6.2** |

Direct rule-vs-rule comparisons:
- **R2 vs R1:** +0.15 [−0.15, +0.40] on train and −0.10 on test.
- **R3 vs R1:** **+0.94 [+0.49, +1.47]** on train and +0.04 [−1.04, +0.84] on test.

Rule-vs-network comparisons are in `summary.md` ("Hand rules vs networks").

Stiffness sweep, mean return (fall %):

| k0 (N/m) | reference | R1 | R2 | R3 (priv.) | MLP | LSTM |
|---|---|---|---|---|---|---|
| 1e3 | 85.9 (35) | 84.1 (39) | 84.5 (38) | **86.4 (33)** | 81.1 (44) | 83.3 (41) |
| 3.2e3 | 93.4 (19) | 92.5 (20) | 92.8 (19) | **93.7 (18)** | 89.5 (26) | 91.6 (23) |
| 7e3 | 95.9 (12) | 95.0 (14) | 95.0 (14) | 95.9 (12) | 93.1 (16) | 94.9 (14) |
| 2.2e4 | 93.1 (9) | 92.9 (9) | 93.0 (9) | 92.7 (9) | 92.1 (11) | 94.6 (9) |
| 7e4 | 73.0 (6) | 78.0 (5) | 79.1 (5) | 80.2 (5) | 81.3 (7) | **82.9 (7)** |
| 1.4e5 | 58.7 (5) | 67.1 (4) | 67.6 (4) | 67.5 (5) | 68.4 (6) | **72.3 (5)** |
| 3e5 | 40.4 (4) | 52.9 (4) | 52.7 (4) | 52.3 (5) | 48.4 (6) | **55.0 (5)** |
| 6e5 | 2.6 (2) | **22.1 (3)** | 21.8 (3) | 18.0 (6) | 1.3 (5) | 15.4 (3) |

## Answers

**Does R1 alone recover most of the networks' gain? Yes, for the MLP; about half, for the LSTM.**
- **On the training ground:**
  - R1 gains +1.52 over the reference, more than the MLP's +1.10.
  - R1 vs MLP is +0.41 [−0.85, +1.61], so there is no detectable difference.
  - The LSTM's gain is +2.97. R1 gets about half of it, and R1 vs LSTM is −1.46 [−2.53, −0.42].
  - R1 falls less than both networks: −3.2 pts vs the MLP (significant) and −1.3 vs the LSTM.
- **On hard ground in the sweep (7e4-3e5),** R1 recovers about 50-85% of the LSTM's advantage over the reference. It beats every network at 6e5.
- **On the test ground,** R1 matches the reference. It beats the LSTM by +1.59 return with 5.7 pts fewer falls, both significant.
- **In short:** a fixed 50 ms dip to K 250, starting 50 ms after contact, is the main mechanism. It is simpler and safer than either network.

**Do R2 or R3 add anything on hard ground? R2 does not; R3 adds a little, and only inside the training range.**
- **R2:** the sensed feature carries too little information (R² ≈ 0.36), and its tuned slope is negligible, so R2 performs like R1 everywhere.
- **R3** (true stiffness, privileged) adds +0.94 [+0.49, +1.47] over R1 on train.
  - It helps most at 7e4 (+2.2) and on soft ground.
  - It adds nothing at 1.4e5-3e5, and loses at 6e5, where its linear-in-log-k0 rule extrapolates to too low a K.
  - Even with perfect knowledge, adapting the dip is worth about 1 return point. The remaining gap to the LSTM on 7e4-1.4e5 ground (about 3-5 points) must come from something other than the dip depth: for example its timing or shape, or its θ/B schedule.

**Does anything fix the soft-ground losses? Only the oracle, and only partly.**
- **Every non-privileged controller with a dip is slightly worse than the reference on soft ground.** At 1e3:
  - R1 is −1.8 return, +4 pts falls.
  - The MLP is −4.7 and the LSTM −2.6, with more falls.
  - The dip lets the leg fold on soft ground.
- **R3 matches or slightly beats the reference there** (86.4 vs 85.9, 33% vs 35% falls). On soft ground its rule pushes K_dip up to 500, so it effectively skips the dip.
- **Skipping the dip is the fix, but it needs to know the ground is soft at about 50 ms.** The sensors predict that only weakly, so R2 cannot do it.
- **The falls themselves are not removed by any controller.** About a third of 1e3 episodes fall whatever the controller; they come from the start pose and the yield on very soft ground.

## Caveats

- Tuned on 100 episodes per cell, with selection over 405 cells. Tuning and evaluation gains agree well (R1 +1.59 on tuning seeds vs +1.52 on evaluation seeds), so selection bias looks small.
- Grids, not a continuous optimiser (no scipy in the `stance` env).
- R2 and R3 are restricted to R1's window by design ("same window"). A ground-dependent window is untested.
- R2 was not added to the sensor-noise sweep (not requested), although it reads one sensor.
- R3 is privileged: it reads the true k0 from the env and is an upper bound, not a controller.

---

# Tasks F-G: richer hand schedule (R4) and report figures (fourth round)

Nothing outside `comparison/` was edited, and nothing was trained. Earlier results are untouched:
- `episodes_rules.csv`, `rules_tuned.json` and `tuning_rules.csv` are byte-identical.
- Before/after diffs of `summary.md`, `summary.csv`, `summary_rules_vs_networks.csv` and `robustness.csv` show only added lines.
- `robustness_episodes.csv` keeps its earlier content as an exact prefix.

**New files:**
- `figures.py` and `run_fg.sh`.
- In `results/`: `tuning_r4.csv`, `r4_marginals.csv`, `r4_tuned.json`, `episodes_r4.csv`, `summary_rules_vs_r1.csv`, `fig_main`, `fig_sweep` and `fig_kdip` (each as `.png` and `.pdf`), and `cost_table.md`.

**Changed files:**
- `rules.py`: R4, plus the `--r4` flag.
- `stats.py`: R4 rows and the "Hand rules vs R1" table.
- `robustness.py`: R4 dashed in the sweep.

## Reproduce

```bash
python -m comparison.rules --r4 --episodes 20 --eval-episodes 20 --out comparison/results/smoke  # smoke
python -m comparison.figures --out comparison/results/smoke
bash comparison/run_fg.sh "$(conda run -n stance which python)"   # full: R4 tune+eval -> stats -> sweep -> figures
```

## Run times

16 CPUs, 14 workers, nice 10. Total for `run_fg.sh`: 6 min 21 s.

| step | cells × episodes | wall time |
|---|---|---|
| R4 stage 1 (schedule grid) | 324 × 100 | 5 min 3 s |
| R4 stage 2 (dip re-check) | 45 × 100 | 47 s |
| evaluation | 2 × 200 | 11 s |
| stiffness sweep | 8 new cells × 100 | 10 s |
| figures and cost table | | 4 s |

## Task F: R4 dip_plus_schedule

**Form.** K follows the R1 dip. In addition:
- θ_d ramps linearly from θ_start {−20, −17, −14} to −20 over T_θ ∈ {20, 40, 60, 80} steps.
- B ramps linearly from B_start {4, 6, 8} to B_end {8, 9, 10} over T_B ∈ {20, 40, 80} steps.

**Tuning.** Tuned on seeds 2000-2099 (training ground) with the same constraint as before: fall rate ≤ the reference's 6%.
- **Stage 1:** the 324-cell schedule grid, with R1's dip (K_dip 250, 50-100 ms).
- **Stage 2:** a 45-cell re-check of K_dip {150-350} × start {25, 50, 75} × end {100, 125, 150} with the best schedule. The grid always includes R1's own values, so stage 2 can only keep or improve stage 1. A built-in check enforces this.

**Result: the best schedule is no schedule.**
- The tuning picked θ_start = −20 and B_start = B_end = 8, which is the constant θ and B of R1.
- 12 grid cells are identical to R1, because T_θ and T_B do nothing when there is no ramp. These cells tie at the top with 90.49, exactly R1's tuning return.
- Stage 2 lengthened the window to 50-125 ms (K_dip 250), giving 90.59. That +0.10 is well within the noise.
- So R4 is R1 with a 25 ms longer dip.

Grid marginals (`r4_marginals.csv`: mean of the cell returns at each value, over the 324 stage-1 cells):

| parameter | values → mean return | range | note |
|---|---|---|---|
| **θ_start** | −20: 90.02, −17: 89.71, −14: 89.42 | **0.60** | the only clear effect: a less plantarflexed start costs return and raises falls (5.4% → 7.1% → 8.3%) |
| B_start | 4: 89.46, 6: 89.82, 8: 89.86 | 0.40 | low initial damping hurts slightly |
| B_end | 8: 89.81, 9: 89.69, 10: 89.64 | 0.17 | negligible |
| T_θ | 20-80: 89.62-89.79 | 0.16 | negligible |
| T_B | 20-80: 89.68-89.76 | 0.08 | negligible |

The best cell for every T_θ value is the same 90.49 (the no-ramp tie). **θ_start matters most, and its best value is "no ramp".**

Evaluation on seeds 1000-1199, paired Δ [95% bootstrap]:

| comparison | train Δ return | train Δ fall, pts | test Δ return | test Δ fall, pts |
|---|---|---|---|---|
| R4 − reference | **+1.70 [+0.36, +3.12]** | +1.0 [−1.5, +4.0] | +0.09 [−0.65, +1.02] | +0.5 |
| R4 − R1 | +0.18 [−0.27, +0.62] | +0.0 [−1.5, +1.5] | −0.04 [−0.81, +0.58] | +0.0 |
| R4 − LSTM | **−1.27 [−2.27, −0.30]** | −1.3 [−3.5, +0.5] | **+1.55 [+0.30, +2.71]** | **−5.7 [−8.8, −2.7]** |
| R4 − MLP | +0.59 [−0.48, +1.63] | **−3.2 [−5.3, −1.2]** | +0.41 | −2.0 |

Hard ground in the stiffness sweep (mean return, 100 episodes):

| k0 | reference | R1 | R4 | LSTM | share of the R1→LSTM gap closed by R4 |
|---|---|---|---|---|---|
| 7e4 | 73.0 | 78.0 | 79.0 | 82.9 | ~19% |
| 1.4e5 | 58.7 | 67.1 | 68.1 | 72.3 | ~19% |

**Does R4 close the gap to the LSTM on hard ground (7e4-1.4e5)? No.**
- It recovers about 1 of the 4-5 return points that separate R1 from the LSTM, and even that 1 point comes from the longer dip window, not the θ/B schedule.
- On the training ground R4 is still significantly below the LSTM (−1.27).
- R4 keeps the rules' advantages: fewer falls than the networks, and on the test ground it beats the LSTM (+1.55 return, −5.7 pts falls).

**Which parameter matters most?** θ_start, and it is best at −20, meaning no ramp. The B ramp and both ramp durations barely matter (range 0.08-0.40 return). Linear θ/B schedules of this kind are not where the LSTM's remaining advantage comes from.

**What else could explain the gap.** `fig_kdip` shows how the networks' K schedule differs from R1's:
- **Early K.** Both networks start at about 300 at contact and climb towards 500 over the first 20-50 ms. R1 starts at 500.
- **Ground-dependent dip.** The mid-stance dip appears only on hard ground, where it reaches about 350 (LSTM) or 280 (MLP). R1 dips to 250 on every ground.
- **Recovery.** The networks recover gradually, not as a step.

R3 (oracle depth) and R4 (θ/B schedules) together close little of the gap. The remaining LSTM advantage is therefore more likely in the shape and timing of K within an episode (the early low K, and the smooth recovery) and in when the dip happens on each ground, than in the depth of the dip or the θ/B schedules. R1's tuning did include windows starting at 0 ms, but only as a constant K_dip, which cannot express "start low, then a ground-dependent dip".

## Task G: report figures and cost table (`figures.py`)

**Shared style.** Every figure is half-page width (3.5 in), with 7 pt text and Okabe-Ito colours. Each controller has the same colour and marker in every figure. Line style shows the family: fixed actions dotted, hand rules dashed, networks solid. Network curves are the mean over 3 seeds, with a ±1 sd band for the seed spread. The figures are saved as PDF and PNG.

1. **`fig_main`:** paired Δ return (top) and Δ fall rate (bottom) vs the fixed reference, for the training ground (left) and test ground (right).
   - Bars with 95% bootstrap CIs, for: reference (marker at 0), best_fixed_tuned, R1, R3 (hatched, labelled privileged), MLP and LSTM.
   - The network bars are seed-averaged per episode, and dots show the single seeds.
   - Source: `summary.csv`.
2. **`fig_sweep`:** return vs ground stiffness for the reference, R1, R3, R4, MLP and LSTM.
   - The training range is shaded and marked with a bracket on the x axis.
   - The best-fixed lines are omitted.
   - Source: `robustness.csv`.
3. **`fig_kdip`:** mean K over the first 300 ms (steps 0-60) in the softest (5-11 kN/m) and hardest (47-100 kN/m) training-stiffness bins.
   - MLP and LSTM as mean ± seed sd, with R1's K profile overlaid as a step.
   - Source: `actions.csv`.
4. **`cost_table.md`:** every number is read from the repo, as follows.

   | quantity | source |
   |---|---|
   | network weights | counted from `model.zip`: MLP 221,575 (110,854 actor); LSTM 75,911 (38,406 actor) |
   | training steps | `config.json`: 300,000 env steps per seed |
   | network training time | last `minutes` value in `curve.csv`, which ends at 300,000 steps for every run: MLP 6.5/6.5/7.4 min, LSTM 18.0/17.7/15.1 min, including the periodic curve evaluations |
   | tuning episodes | grid and tuning CSVs |
   | rule tuning times | `log.txt` timestamps |

   Cells marked "n/a":
   - The reference's tuning data and time: its grid was not saved.
   - The networks' training episode counts: only env steps are logged.
