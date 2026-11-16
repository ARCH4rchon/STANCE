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
- `--network mlp|lstm|both`: which network to run (default both).
- `--model-seed N|all`: which trained seed to load from `trained_networks/<network>/final_s<N>` (default all).
- `--episodes N`: evaluation episodes on each ground (default 200).
- `--seed S`: first episode seed; the episodes use S, S+1, ... (default 1000).
- `--out PATH`: where to write the results (default `results/demo_episodes.csv`).
- `--show N`: after the evaluation, replay N episodes (0-10, default 5, starting at `--seed`) live in the MuJoCo viewer, using the first selected network and seed. Without a display it prints one line and skips.
- `--slow X`: viewer slow-motion factor (default 10; 1 = real time).
- `--record PATH.mp4`: save one episode (seed `--seed`, first selected network and seed) as a 30 fps video. Headless machines need `MUJOCO_GL=egl` (or `osmesa`), for example `MUJOCO_GL=egl python demo.py --show 0 --record demo.mp4`. If the video can't be made, the demo prints one line and carries on.
- `--train --steps N`: allows you to train your own model and run it against our demo (can take up to around 17 minutes per network with the default steps). It is saved in a temporary folder, never in `trained_networks/`.

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

We used Claude (Anthropic) to draft a template code for us such that if we worked on it induvidually, there were keys that made sure our tests remained common across files. Additionally we used it to help debug our code and to flag discrepencies during merges.