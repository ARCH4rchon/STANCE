# trained_networks_extra_seeds

Two extra training seeds (4 and 5) for each network. Together with the submitted seeds 0-2 in `../trained_networks/`, they give 5 seeds per network. They are kept in a separate folder so that `demo.py` and the comparison, which read `trained_networks/`, still use only the submitted models.

## Contents

| Path | What it is |
|---|---|
| `networkA_mlp/final_s4`, `final_s5` | Network A (MLP) trained with seeds 4 and 5 |
| `networkB_lstm/final_s4`, `final_s5` | Network B (LSTM) trained with seeds 4 and 5 |
| `logs/` | The training output of each run |

Each run folder has the same files as in `trained_networks/`: `config.json`, `model.zip` and `vecnormalize.pkl`. The runs use the unchanged `agent/train.py` settings (300,000 steps) apart from the seed. There is no `curve.csv` or `episodes_*.csv` because the training script used does not write them.

These runs were trained with MuJoCo 3.6 and Gymnasium 1.2.3, not the versions pinned in `environment.yml` (3.15 and 1.4). The results below were all evaluated in the pinned environment.

## 5-seed results

Evaluated with `common/evaluate.py`: 200 deterministic episodes per ground (seeds 1000-1199). Mean ± standard deviation over the 5 seeds.

| Network | Train return | Train fall % | Test return | Test fall % |
|---|---|---|---|---|
| MLP (seeds 0, 1, 2, 4, 5) | 88.42 ± 0.88 | 12.8 ± 2.7 | 76.77 ± 3.38 | 55.0 ± 7.3 |
| LSTM (seeds 0, 1, 2, 4, 5) | 90.73 ± 1.18 | 9.9 ± 1.4 | 77.66 ± 1.71 | 53.1 ± 6.2 |
| Reference fixed action | 87.90 | 7.5 | 78.57 | 49.5 |

| Network | Seed | Train return | Train fall % | Test return | Test fall % |
|---|---|---|---|---|---|
| MLP | 0 | 88.83 | 8.5 | 79.31 | 49.0 |
| MLP | 1 | 89.57 | 13.0 | 78.37 | 52.0 |
| MLP | 2 | 88.62 | 13.5 | 77.09 | 55.0 |
| MLP | 4 | 87.53 | 13.0 | 78.20 | 51.5 |
| MLP | 5 | 87.53 | 16.0 | 70.90 | 67.5 |
| LSTM | 0 | 89.76 | 8.0 | 79.20 | 47.5 |
| LSTM | 1 | 91.06 | 11.5 | 76.20 | 59.5 |
| LSTM | 2 | 91.80 | 10.0 | 75.93 | 60.0 |
| LSTM | 4 | 91.79 | 9.0 | 77.30 | 51.0 |
| LSTM | 5 | 89.22 | 11.0 | 79.68 | 47.5 |

## Loading a run

From the folder holding `demo.py`, a run loads the same way as the submitted ones:

```python
import demo
net, model, vecnormalize, wrappers = demo.load_run("Network_Agent/trained_networks_extra_seeds/networkB_lstm/final_s4")
```
