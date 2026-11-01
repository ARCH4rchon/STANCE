# STANCE Agent Submission (Group 42)

This folder holds:
- `Gym/`: the STANCE environment (`stance_env/`) and the MuJoCo leg model (`model/leg.xml`) from our previous submission
- `Network_Agent/networkA_mlp/`, `Network_Agent/networkB_lstm/`: the two network definitions
- `Network_Agent/agent/`: `train.py` (training and evaluation), `config.py` (settings), `actions.py` (residual action)
- `Network_Agent/common/`: environment setup, policy adapters and the shared evaluation protocol
- `Network_Agent/trained_networks/`: our trained networks (weights, normalisation statistics, config and learning curve)
- `Network_Agent/demo.py`: `Network_Agent/demo_agent.py`
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

# If you want to run a pretrained model:
python demo_agent.py --model trained_networks/networkB_lstm/final_s0 --episodes 200 # The chosen network after testing
#or
python demo_agent.py --model trained_networks/networkA_mlp/final_s0 --episodes 200 # If you want to try it

# If you want to train your own model:                              
python demo_agent.py --train                                                
```

## If you want to train a fresh network:
```bash
cd Network_Agent # If you haven't already
conda env create -f environment.yml # If you havent already
conda activate stance
python -m agent.train --network lstm
python -m agent.train --network mlp # If you want to try it
```

## Notes:

- If ROS is sourced in your shell: run `env -u PYTHONPATH python demo.py`.
- If the environment is not found: set `STANCE_GYM_PATH` to the `Gym/` folder of this package.

## AI Declreation:

We used Claude (Anthropic) to draft a template code for us such that if we worked on it induvidually, there were keys that made sure our tests remained common across files. Additionally we used it to help debug our code and to flag discrepencies during merges.