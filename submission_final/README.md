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

We used Claude (Anthropic) to evaluate ideas, merge our individual changes between files to avoid discrepancies, debug parts of our code and aid in writing only the very sepficic sections of our code that have been outlined clearly.

The video (`final_video.mp4`) was entirely run, recorded and editited by us. However, Claude was entirely responsible for coding the viewer's design and visuals (such as the arrows, colours and the ghost leg). We are responsible for the quality of work produced in the video but do not claim the visual imporvements made by Claude as our own.

No Machine Learning aspect of the project was altered during the creation of this video. No additional changes were made to the agent itself. No data was altered by using Claude (Anthropic). The video shows the real simulation running our trained networks; no part of the footage is AI generated.
