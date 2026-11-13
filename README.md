# STANCE Agent Submission (Group 42)

This folder holds:
- `Gym/`: the STANCE environment (`stance_env/`) and the MuJoCo leg model (`model/leg.xml`) from our previous submission
- `Network_Agent/networkA_mlp/`, `Network_Agent/networkB_lstm/`: the two network definitions
- `Network_Agent/agent/`: `train.py` (PPO training and evaluation, with all PPO settings in `TRAIN`), `actions.py` (residual action)
- `Network_Agent/common/`: environment setup, policy adapters and the shared evaluation protocol
- `Network_Agent/trained_networks/`: our trained networks (weights, normalisation statistics, config and learning curve)
- `Network_Agent/demo_agent.py`: loads or trains a network, evaluates it against the best fixed action, and can replay episodes in the MuJoCo viewer
- `Network_Agent/experiments/`: lightweight records (config, learning curve, per-episode results) of the ablation runs, without weights
- `Network_Agent/figures/`: reward-weight sweep figure
- `Reports/`: proposal and the three milestone reports
- `environment.yml`

## Setup

```bash
conda env create -f environment.yml
conda activate stance
cd Network_Agent
```

## Agent demo
```bash
cd Network_Agent # If you haven't already

python demo_agent.py --model trained_networks/networkB_lstm/final_s0 --episodes 200 --show 5 # The chosen network after testing
#or
python demo_agent.py --model trained_networks/networkA_mlp/final_s0 --episodes 200 --show 5 # If you want to try our old MLP setup
                            
python demo_agent.py --train --show 5 # If you want to train your own model
```
- `--model /path/to/model/folder`: allows you to run the demo with the pre-trained network's weights
- `--train`: allows you to train your own model and run it against out demo. (Can take around 17 minutes) 
- `--show N`: after the table, replay the first N test episodes (0-10, default 5) live in the MuJoCo viewer.
- `--slow X`: viewer slow-motion factor (default 10; 1 = real time).

## If you JUST want to train a fresh network:
```bash
cd Network_Agent # If you haven't already

conda env create -f environment.yml # If you havent already
conda activate stance

python -m agent.train --network lstm
python -m agent.train --network mlp # If you want to try it
```

## Notes:

- If ROS is sourced in your shell: run `env -u PYTHONPATH python demo_agent.py`.
- If the environment is not found: set `STANCE_GYM_PATH` to the `Gym/` folder of this package.

## AI Declaration:

We used Claude (Anthropic) to draft a template code for us such that if we worked on it individually, there were keys that made sure our tests remained common across files. Additionally we used it to help debug our code and to flag discrepencies during merges.