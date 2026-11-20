# Network_Agent

The learning agent: the two networks, the PPO training code, the shared evaluation code and our trained networks. `demo.py` in the folder above uses everything here.

## Contents

| Path | What it is |
|---|---|
| `networkA_mlp/network.py` | Network A: an MLP (PPO) that sees the last 10 sensor frames |
| `networkB_lstm/network.py` | Network B: an LSTM (RecurrentPPO) that sees one frame at a time; our chosen agent |
| `agent/train.py` | Training: the PPO settings and the training command |
| `agent/actions.py` | Residual action: the network outputs an offset from the fixed action (-20 deg, K 500, B 10) |
| `common/` | Shared code: building the environment, the evaluation protocol (seeds 1000-1199), network inputs and policy adapters |
| `trained_networks/` | Our trained networks, 3 seeds each (`final_s0`-`final_s2`) |
| `comparison/` | The controller comparison (see `comparison/README.md`) |

## Run

There are no tests in this folder. Run these commands from `Network_Agent/`:

```bash
python -m agent.train --network lstm                        # train Network B
python -m agent.train --network mlp --seed 1 --steps 4096   # short Network A run, to check it works
```

Each run is saved to `runs/<network>/seed<N>/`.
