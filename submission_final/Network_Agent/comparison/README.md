# comparison

Compares six controllers on the same episodes: the reference fixed action, the tuned fixed action, two hand-written stiffness-dip rules (R1, R3) and the two trained networks.

## Contents

| Path | What it is |
|---|---|
| `rules.py` | The hand-written stiffness-dip rules R1 and R3 |
| `harness.py` | The controllers and a parallel runner |
| `compare.py` | Runs every controller on both grounds and on a ground-stiffness sweep |
| `stats.py` | Paired bootstrap comparisons, printed to the terminal |
| `results/` | Our saved results, figures and tuning tables |
| `results/noise_check.txt` | Sensor-noise sanity check: with noise on, the network inputs differ from the clean ones by the expected amount (about 1 sensor std per channel), and the previous-action inputs get no noise. Both seed-0 networks pass |
| `results/robustness.csv`, `robustness_episodes.csv`, `robustness.png`/`.pdf` | Robustness sweeps: both networks with sensor noise of 0-2 sensor standard deviations, and all controllers across ground stiffness (summary and per-episode rows) |

The noise check and robustness files are kept as data only. The scripts that produced them (`noise_check.py`, `robustness.py` and the sensor-noise wrapper in the full `harness.py`) are not part of this package.

## Run

Run these commands from `Network_Agent/` (the folder above this one):

```bash
python -m comparison.compare                        # all controllers, 200 episodes per ground, plus the sweep (about 3 min on 16 cores)
python -m comparison.stats > comparison/results/summary.txt
```

Both write to `comparison/results/` by default. Use `--out DIR` to write somewhere else. For a quick check, run `python -m comparison.compare --episodes 20 --sweep-episodes 10 --out /tmp/cmp`.
