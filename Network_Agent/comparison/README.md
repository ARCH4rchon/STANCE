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

## Run

Run these commands from `Network_Agent/` (the folder above this one):

```bash
python -m comparison.compare                        # all controllers, 200 episodes per ground, plus the sweep (about 3 min on 16 cores)
python -m comparison.stats > comparison/results/summary.txt
```

Both write to `comparison/results/` by default. Use `--out DIR` to write somewhere else. For a quick check, run `python -m comparison.compare --episodes 20 --sweep-episodes 10 --out /tmp/cmp`.
