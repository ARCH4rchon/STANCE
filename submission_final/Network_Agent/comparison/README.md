# Controller comparison

Evaluation only: nothing here trains a network. Six controllers are compared on the training ground
and the test (out-of-range) ground, seeds 1000-1199, deterministic, with `common.evaluate.run_episodes`:
the fixed reference (-20 deg, 500, 10), the tuned best fixed action, two hand-written K-dip rules
(R1 timed, R3 oracle) and the two trained networks (MLP and LSTM, 3 seeds each).

## Files

| file | what it does |
|---|---|
| `harness.py` | shared code: finds the controllers, loads networks, ground-stiffness and sensor-noise wrappers, parallel resumable cells |
| `compare.py` | runs the reference and every network seed on both grounds, writes `results/episodes.csv` |
| `tune.py` | grid search for the best fixed action on tuning seeds 2000-2099, evaluates it as `best_fixed_tuned` |
| `rules.py` | the R1 and R3 K-dip rules, their grid tuning on seeds 2000-2099 and their evaluation |
| `stats.py` | summary tables with paired bootstrap 95% intervals vs the reference, rules vs networks, R3 vs R1 |
| `robustness.py` | ground-stiffness sweep (all controllers) and sensor-noise sweep (networks) |
| `actions.py` | records the networks' applied actions per step against ground stiffness |
| `figures.py` | report figures `fig_main`, `fig_sweep`, `fig_kdip` and `cost_table.md` |

Committed in `results/`: `summary.md` / `summary.csv`, `summary_rules_vs_networks.csv`,
`summary_rules_vs_r1.csv`, `robustness.csv`, `fig_main`, `fig_sweep`, `fig_kdip` (pdf + png),
`cost_table.md`, `actions_slopes.csv`; per-episode data in `episodes.csv`, `episodes_rules.csv`,
`episodes_tuned.csv`; tuning results in `grid.csv`, `tuning_rules.csv`, `rules_tuned.json`,
`best_fixed_tuned.json`. A rerun also writes files that are not committed: `robustness_episodes.csv`,
`robustness.png/pdf` (superseded by `fig_sweep`), `grid_heatmap.*`, `actions_early.*`, `actions_time.*`,
`actions.csv` and `log.txt` if you log to it.

## Reproduce

From `submission_final/Network_Agent/`, the folder that holds `demo.py` (prefix with
`env -u PYTHONPATH` if ROS is sourced). Finished cells are cached
in `results/cells/`, so a rerun only plays what is missing.

```bash
python -m comparison.compare                         # reference + networks, both grounds
python -m comparison.tune                            # best fixed action (grid), then its evaluation
python -m comparison.rules                           # tune and evaluate R1 and R3
python -m comparison.robustness                      # stiffness and noise sweeps
python -m comparison.actions && python -m comparison.stats && python -m comparison.figures
```

Add `--episodes 20 --out comparison/results/smoke` to `compare`, `robustness` or `actions` for a quick check.

`results/actions.csv` (5.5 MB, one row per network step) is not committed. Regenerate it with
`python -m comparison.actions` (27 s on 14 workers in the full run); `figures.py` needs it for `fig_kdip`.
`results/cells/` and `results/smoke/` are caches and are not committed either.
`cost_table.md` takes the rule and grid tuning times from `results/log.txt`, which is not committed;
without it a rerun of `figures.py` writes "n/a" in those three cells (the committed table keeps the
times of the original run: grid 77 s, R1 3.3 min, R3 0.8 min, on 14 workers).

## Differences from the full branch (`Control_Comparison_Data`)

This branch keeps only the six controllers above. The full branch also had best_fixed (-20, 500, 6),
R2 (a sensor-driven dip), R4 (a dip plus theta/B schedule), a sensing analysis and a noise sanity
check; their code and outputs are removed here. Every row for the six kept controllers is
numerically identical to the full branch. The full branch's lab notes (`NOTES.md`) and run log
(`results/log.txt`) stay on `Control_Comparison_Data`.

## AI use

Every file in this folder was drafted with Claude Code (Anthropic), for the AI declaration.
Moved onto the rewritten code; imports adapted by Claude Code.
