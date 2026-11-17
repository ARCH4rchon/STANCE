# Additional training runs

Extra runs to check whether longer training or more seeds improve on the submitted models in `trained_networks/` (300k steps, seeds 0-2).

- `seedN_1M`: 1,000,000 training steps (seeds 0, 1, 2)
- `seedN_300k`: 300,000 training steps, the submitted setting (seeds 4, 5)
- `logs/`: training output for each run

All runs use the unchanged `agent/train.py` settings except `--steps` and `--seed`. They were trained in a conda environment with MuJoCo 3.6 and Gymnasium 1.2.3 (the package pins 3.15 and 1.4). No `curve.csv` is written because the submission `train.py` has no learning-curve logger.

## Results

Every model was evaluated in the same environment: 200 deterministic episodes per ground, seeds 1000-1199. `final_sN` = submitted models from `trained_networks/`.

| Network | Steps | Seed | Train return | Train fall % | Test return | Test fall % |
|---|---|---|---|---|---|---|
| MLP | 300k | 0 (final_s0) | 88.57 | 9.5 | 79.32 | 49.0 |
| MLP | 300k | 1 (final_s1) | 89.49 | 13.5 | 78.25 | 52.5 |
| MLP | 300k | 2 (final_s2) | 88.55 | 13.5 | 77.24 | 54.5 |
| MLP | 300k | 4 | 87.32 | 14.0 | 78.18 | 51.5 |
| MLP | 300k | 5 | 87.28 | 16.5 | 71.27 | 66.0 |
| **MLP** | **300k** | **mean (5)** | **88.24** | **13.4** | **76.85** | **54.7** |
| MLP | 1M | 0 | 88.04 | 15.5 | 75.31 | 58.5 |
| MLP | 1M | 1 | 88.80 | 15.0 | 75.38 | 59.5 |
| MLP | 1M | 2 | 85.74 | 12.0 | 78.01 | 49.5 |
| **MLP** | **1M** | **mean (3)** | **87.53** | **14.2** | **76.23** | **55.8** |
| LSTM | 300k | 0 (final_s0) | 89.75 | 8.0 | 79.22 | 47.5 |
| LSTM | 300k | 1 (final_s1) | 91.01 | 11.5 | 76.20 | 59.5 |
| LSTM | 300k | 2 (final_s2) | 91.61 | 10.0 | 76.18 | 59.0 |
| LSTM | 300k | 4 | 91.59 | 9.5 | 77.29 | 51.0 |
| LSTM | 300k | 5 | 89.23 | 11.0 | 79.68 | 47.5 |
| **LSTM** | **300k** | **mean (5)** | **90.64** | **10.0** | **77.71** | **52.9** |
| LSTM | 1M | 0 | 89.44 | 9.0 | 78.31 | 51.0 |
| LSTM | 1M | 1 | 91.23 | 10.5 | 78.65 | 51.0 |
| LSTM | 1M | 2 | 90.88 | 10.5 | 76.12 | 58.5 |
| **LSTM** | **1M** | **mean (3)** | **90.52** | **10.0** | **77.69** | **53.5** |

## Findings

- Training for 1M steps instead of 300k did not help: the LSTM is unchanged within seed noise and the MLP is slightly worse. The submitted learning curves already flatten around 100-150k steps.
- Seed-to-seed spread is large (LSTM test falls 47.5-59.5%, MLP 49-66%), so results should be averaged over seeds.
- Over 5 seeds the LSTM stays ahead of the MLP on every average. On the test ground both still fall more often than the fixed reference action (49.5%).
