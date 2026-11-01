# STANCE Agent Submission (Group 42)

This folder holds:
- `Gym/`: the STANCE environment (`stance_env/`) and the MuJoCo leg model (`model/leg.xml`) from our previous submission
- `Network_Agent/networkA_mlp/`, `Network_Agent/networkB_lstm/`: the two network definitions
- `Network_Agent/agent/`: the learning agent: `train.py` (training and evaluation), `config.py` (settings), `curve.py` (learning curve), `actions.py` (residual action), `reward.py` (optional reweighted impact penalty, off by default)
- `Network_Agent/common/`: environment setup, policy adapters and the shared evaluation protocol
- `Network_Agent/trained_networks/`: our trained networks (weights, normalisation statistics, config and learning curve)
- `Network_Agent/demo.py`: the network demo; `Network_Agent/demo_agent.py`: the agent demo
- `Network_Agent/reward_tradeoff.py` and `Network_Agent/figures/`: the reward trade-off figure
- `environment.yml`

## Setup and running

```bash
conda env create -f environment.yml
conda activate stance
cd Network_Agent
python demo.py
#or 
python3 demo.py
```

## Agent demo
```bash
cd Network_Agent

# If you want to run a pretrained model:
python demo_agent.py --model trained_networks/networkB_lstm/final_s0 --episodes 200 # The chosen network after testing
#or
python demo_agent.py --model trained_networks/networkAmlp/final_s0 --episodes 200 # If you want to try it

# If you want to train your own model:                              
python demo_agent.py --train 

# If you want to understand about our report section on Reward Weights:
python reward_tradeoff.py --quick                                                 
```

## If you want to train a fresh network:
```bash
cd Network_Agent
conda env create -f environment.yml # If you havent already
conda activate stance
python -m agent.train --network mlp --smoke   # pipeline check
python -m agent.train --network mlp
python -m agent.train --network lstm
```

## What the demo shows:

- Inputs and preprocessing: observation shape, frames, sensors and normalisation statistics
- Data flow: the layers and the parameter count of each part
- Outputs: what the three action values mean and their physical ranges
- One real observation: one forward pass with the action and value estimate
- Weights: shape and statistics per layer, with histograms saved to `demo_output/`
- Learning curve: survival and toe peak against training steps, saved to `demo_output/`
- Evaluation: the network next to the best fixed action on the same seeds
- Forward and backward pass: a short rollout, the PPO loss, its gradients, and 10 Adam steps on a copy

It ends with a side-by-side table of both networks and the fixed action.

## Notes:

- If ROS is sourced in your shell: run `env -u PYTHONPATH python demo.py`.
- If the environment is not found: set `STANCE_GYM_PATH` to the `Gym/` folder of this package.

## AI Declreation:

We used Claude (Anthropic) to draft a template code for us such that if we worked on it induvidually, there were keys that made sure our tests remained common across files. Additionally we used it to help debug our code and to flag discrepencies during merges.