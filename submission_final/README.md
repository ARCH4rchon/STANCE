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

## Demo
```bash
cd Network_Agent # If you haven't already

python demo.py # Both networks, all trained seeds, 200 episodes on the training and test ground, then 5 episodes in the viewer
#or
python demo.py --network lstm --model-seed 0 # Only the chosen network after testing (one seed)
python demo.py --network mlp # If you want to try our old MLP setup
python demo.py --episodes 50 --seed 1100 # 50 episodes, seeds 1100-1149
python demo.py --out my_results/run.csv # Write the results somewhere else
python demo.py --record demo.mp4 # Also save one episode as a video
python demo.py --show 0 # No viewer (for machines without a display)

python demo.py --train --show 5 # If you want to train your own model
python demo.py --train --steps 4096 --show 0 # Short training run, just to check it works
```

Every run is compared with the best fixed action (-20 deg, K 500, B 10) on the same seeds. The terminal shows the two file paths and one line per network (mean return and fall rate on both grounds). The files are:
- `PATH` (e.g. `results/demo_episodes.csv`): one row per episode, with columns `network`, `model_seed`, `ground` (train or test), `episode_seed`, `return`, `fell` (1 if the leg fell), `steps` and `toe_peak_bw` (largest toe force, in body weights).
- `PATH` with `_summary` (e.g. `results/demo_episodes_summary.csv`): one row per network, model seed and ground with the mean return, fall rate (%) and mean toe peak (BW); `model_seed` = `mean` is the mean over the model seeds.

## If you JUST want to train a fresh network:
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

We used Claude (Anthropic) to evaluate ideas, merge our induvidual changes between files to avoid discrepencies, debug parts of our code and aid in writing specific sections of our code that have been outlined clearly (please check our code).

The mp4 video file in the submission folder was recorded and edited by us and the run was fully done and mintored by us.
However, Claude (Anthropic) was used to code scrictly the design and visuals (such as adding arrows, colour and a ghost)
No Machine Learning aspect of the project was altered during the creation of this video. No addition changes were made to the agent itself. No data was altered by using Claude (Anthropic). The video itself is NOT AI generated, nor was it AI editied.