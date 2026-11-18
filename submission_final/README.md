# STANCE Agent Submission (Group 42)

STANCE is a learning agent for a powered prosthetic ankle during weight acceptance on compliant ground whose stiffness is hidden from the agent. Every 5 ms a PPO-trained network reads the ankle's sensor history and sets the ankle's target angle, stiffness K and damping B. We compare two networks, an MLP and an LSTM (our chosen agent), against the best fixed action and against hand-written stiffness-dip rules.

## Main files

| Path | What it is |
|---|---|
| `Gym/` | The STANCE environment (`stance_env/`) and the MuJoCo leg model (`model/leg.xml`) from our previous submission |
| `Network_Agent/demo.py` | The demo: evaluates the trained networks against the best fixed action on both grounds, writes the results to CSV, and can replay or record episodes |
| `Network_Agent/networkA_mlp/network.py` | Network A: MLP (PPO) that sees the last 10 sensor frames |
| `Network_Agent/networkB_lstm/network.py` | Network B: LSTM (RecurrentPPO) that sees one frame at a time and keeps a memory; our chosen agent |
| `Network_Agent/agent/train.py` | Training: the PPO settings (`TRAIN`), the wrapper list and the training command |
| `Network_Agent/agent/actions.py` | Residual action: the network outputs an offset from the fixed action (-20 deg, K 500, B 10) |
| `Network_Agent/common/` | Shared code: `env_setup.py` (finds and builds the environment, converts actions), `evaluate.py` (the evaluation protocol: seeds 1000-1199, the fixed reference action), `inputs.py` (which frames and sensors a network sees), `policies.py` (policy adapters) |
| `Network_Agent/trained_networks/` | Our trained networks, 3 seeds each (`final_s0`-`final_s2`): weights (`model.zip`), normalisation statistics (`vecnormalize.pkl`), config, learning curve and per-episode results |
| `Network_Agent/comparison/` | The controller comparison: `rules.py` (hand-written stiffness-dip rules R1 and R3), `harness.py` (the controllers and a parallel runner), `compare.py` (runs every controller on both grounds and on a ground-stiffness sweep), `stats.py` (paired bootstrap comparisons) |
| `results_new/` | **The comparison results used in our report**: `episodes.csv` (6 controllers x 2 grounds x 200 episodes, 4,000 rows), `sweep.csv` (stiffness sweep), `summary.txt` (the printed comparison) and `results_new_compare.log` (the run's console output) |
| `Network_Agent/comparison/results/` | Earlier outputs kept for reference: `grid.csv` (the fixed-action grid search behind `best_fixed_tuned`), `tuning_rules.csv` (the tuning of R1 and R3), `fig_*.png` and `cost_table.md`. The scripts that made them are not part of this package |
| `environment.yml` | The conda environment |

## Setup

```bash
conda env create -f environment.yml
conda activate stance
cd Network_Agent
```

Main libraries (exact versions, from `environment.yml`):

| Software | Version |
|---|---|
| Python | 3.10 |
| PyTorch (CPU) | 2.14.1 |
| Stable-Baselines3 | 2.9.0 |
| sb3-contrib (RecurrentPPO) | 2.9.0 |
| Gymnasium | 1.4.0 |
| MuJoCo | 3.15.0 |
| NumPy | 2.2.6 |
| pandas (comparison only) | 2.3.3 |
| imageio, imageio-ffmpeg (`--record` only) | 2.38.0, 0.6.0 |

## Demo
```bash
cd Network_Agent # If you haven't already

python demo.py # Both networks, all trained seeds, 200 episodes on the training and test ground, then 5 episodes in the viewer
#or
python demo.py --network lstm --model-seed 0 # Only the chosen network after testing (one seed)
python demo.py --network mlp # If you want to try our old MLP setup
python demo.py --episodes 50 # 50 episodes per ground, seeds 1000-1049
python demo.py --record demo.mp4 # Also save one episode as a video
python demo.py --show 0 # No viewer (for machines without a display)

python demo.py --train --show 5 # If you want to train your own model
python demo.py --train --steps 4096 --show 0 # Short training run, just to check it works
```
`demo.py` takes 7 arguments:
- `--network mlp|lstm|both`: which network to run (default both).
- `--model-seed 0|1|2|all`: which trained seed to load from `trained_networks/<network>/final_s<N>` (default all).
- `--episodes N`: evaluation episodes on each ground, seeds 1000 to 1000+N-1 (default 200, the full evaluation protocol).
- `--show N`: after the evaluation, replay the first N episodes (from seed 1000) live in the MuJoCo viewer at 10x slow motion, using the first selected network and seed (default 5, 0 = no viewer). Without a display it prints one line and skips.
- `--record PATH.mp4`: save one episode (seed 1000, first selected network and seed) as a 30 fps video. Headless machines need `MUJOCO_GL=egl` (or `osmesa`), for example `MUJOCO_GL=egl python demo.py --show 0 --record demo.mp4`. If the video can't be made, the demo prints one line and carries on.
- `--train`: train each selected network from scratch instead of loading ours, then run it against our demo (can take up to around 17 minutes per network with the default 300,000 steps). It is saved in a temporary folder, never in `trained_networks/`, and shows up as model seed `new`.
- `--steps N`: with `--train`, the number of training steps.

Every run is compared with the best fixed action (-20 deg, K 500, B 10) on the same seeds. The terminal shows the two file paths and one line per network (mean return and fall rate on both grounds). The files are written to `results/` inside the folder you run from:
- `results/demo_episodes.csv`: one row per episode, with columns `network`, `model_seed`, `ground` (train or test), `episode_seed`, `return`, `fell` (1 if the leg fell), `steps` and `toe_peak_bw` (largest toe force, in body weights).
- `results/demo_episodes_summary.csv`: one row per network, model seed and ground with the mean return, fall rate (%) and mean toe peak (BW); `model_seed` = `mean` is the mean over the model seeds.

## If you JUST want to train a fresh network:
```bash
cd Network_Agent # If you haven't already

conda env create -f ../environment.yml # If you haven't already
conda activate stance

python -m agent.train --network lstm
python -m agent.train --network mlp # If you want to try it
python -m agent.train --network lstm --seed 1 --steps 4096 # Other seed, short run
```
The run is saved to `Network_Agent/runs/<network>/seed<N>/` (config, model and normalisation statistics), and the script ends with a 200-episode check on the training ground.

## Reproducing the comparison results
```bash
cd Network_Agent # If you haven't already

python -m comparison.compare --out ../results_new # all 6 controllers, both grounds, 200 episodes, plus the stiffness sweep (about 3 minutes on 16 cores)
python -m comparison.stats --out ../results_new > ../results_new/summary.txt
```
This rewrites `results_new/`; our regeneration matched the earlier results to within 5e-5. Without `--out` both scripts use `Network_Agent/comparison/results/`. The run caches finished cells in `results/cells/` (not part of the package).

## Notes:

- If ROS is sourced in your shell: run `env -u PYTHONPATH python demo.py`.
- If the environment is not found: set `STANCE_GYM_PATH` to the `Gym/` folder of this package.

## AI Declaration:

We used Claude (Anthropic), through Claude Code, to evaluate ideas, merge our individual changes between files to avoid discrepancies, debug parts of our code and aid in writing specific sections of our code that have been outlined clearly (please check our code; the commits that Claude helped with carry a `Co-Authored-By: Claude` line).

The video (`final_video.mp4`) was made with scripts drafted by Claude Code: one script played the demo viewer offscreen with our trained networks and saved the frames, and another added the title cards, captions and explanation text and joined the clips. Claude also coded the viewer's design and visuals (such as the arrows, colours and the ghost leg). We chose what the video shows, ran and monitored the scripts, and checked every clip and number in it against our results in `results_new/`.
No Machine Learning aspect of the project was altered during the creation of this video. No additional changes were made to the agent itself. No data was altered by using Claude (Anthropic). The video shows the real simulation running our trained networks; no part of the footage is AI generated.
