# STANCE: Stiffness and Torque Adaptation for Non-linear Compliant Environments

**ME5418 Group 42:** Vishal Mangla, Mukul Yadav, Muthaiah Mahadevan

> [!NOTE]
> **Academic disclosure.** This project was completed as coursework for **ME5418 Machine Learning in Robotics** at the
> **National University of Singapore (NUS)**, Department of Mechanical Engineering, College of Design and Engineering,
> in Semester 1 of Academic Year 2026/2027, as part of our **Master of Science (MSc) in Robotics**. The module was
> taught by Professor Guillaume A. Sartoretti. The work was submitted for assessment, and it is shared here for
> reference only. It is not an official NUS publication. If you are taking the same module, follow its academic
> integrity and reuse rules before you use anything from this repository.

STANCE is a learning agent for a powered prosthetic ankle during weight acceptance on compliant ground whose stiffness is hidden from the agent. Every 5 ms a PPO-trained network reads the ankle's sensor history and sets the ankle's target angle, stiffness K and damping B. We compare two networks, an MLP and an LSTM (our chosen agent), against the best fixed action and against hand-written stiffness-dip rules.

## The problem

Most passive lower-limb prostheses use a fixed-stiffness ankle: a prosthetist sets its compliance once, and it stays the same whatever the terrain. That works on flat, rigid, predictable ground. Powered prostheses adjust for walking speed and for different rigid terrains, but no existing control strategy adjusts for compliant ground, where the surface gives way unpredictably underfoot.

More than half of lower-limb prosthesis users fall at least once a year, many during *weight acceptance*: the 100-200 ms after heel strike when full body weight shifts onto the prosthetic leg. In that window the ankle must get its stiffness right immediately. Too stiff, and it slams the impact into the residual limb. Too compliant, and the leg folds. Which failure occurs is decided by the ground: concrete, wet grass, gravel and sand compress, damp and grip differently, and some stay dented afterwards. No commercial sensor can measure ground compliance before contact, so the agent uses only real prosthesis signals (encoder, motor current, load cells and IMU) and must infer the ground from how they change.

## Why reinforcement learning

| Method | Why it fails here | What RL does instead |
| :--- | :--- | :--- |
| **FSM impedance control (clinical standard)** | Stiffness is fixed in advance for a listed set of situations. The surface about to be landed on is not on that list. | Learns a continuous mapping from sensor history to stiffness, with no list of surfaces to enumerate. |
| **Online estimation + adaptive control** | Needs force data, which only appears after impact begins, to estimate stiffness. | Sets stiffness at touchdown from what it learned in training, then refines it as force builds. |
| **Model Predictive Control** | Requires a contact model; no fixed parameter set describes stochastic ground deformation, and solving a contact problem at 200 Hz on a prosthesis is unrealistic. | Moves the optimisation offline into training. Inference is a small, fixed cost per step. |

- **Structural underactuation:** five degrees of freedom, one motor. There is no inverse from a desired body motion to an ankle torque.
- **Non-stationary ground:** ground properties are drawn from continuous ranges every episode and the surface dents permanently, so its effective stiffness depends on how the agent has been loading it. There is nothing fixed to memorise.

## How it works

- **Environment (`Gym/`):** a MuJoCo leg lands on randomised compliant ground at 200 Hz for up to 100 steps. The agent sees the last 10 frames of 11 sensor values. The reward is +1 per step survived, minus a penalty for toe force above 1.58 body weights, minus a smoothness term, and -25 for a fall. `StanceAnkle-v0` is the training ground; `StanceAnkleTest-v0` is a held-out ground with parameters outside the training ranges.
- **Action:** the ankle torque follows an impedance law, `torque = K × (target angle − ankle angle) − B × ankle speed`. The network outputs an offset from a fixed action (-20°, K 500, B 10), not the action itself.
- **Networks:** Network A is an MLP (PPO) that sees the last 10 frames. Network B is an LSTM (RecurrentPPO) that sees one frame at a time and keeps a memory; it is our chosen agent.
- **Evaluation:** every number comes from `Network_Agent/common/evaluate.py`: deterministic actions on episode seeds 1000-1199, compared with the best fixed action on the same seeds, and averaged over 3 training seeds per network.

## Repository layout

| Path | What it is |
|---|---|
| `demo.py` | The demo: evaluates the trained networks against the best fixed action on both grounds, writes the results to CSV, and can replay or record episodes |
| `demo_ui.py` | The improved viewer and video used by `demo.py` (cosmetic only, see [UI improvements](#ui-improvements)) |
| `STANCE_demo.mp4` | Our demo video |
| `environment.yml` | The conda environment |
| `Gym/` | The STANCE environment (`stance_env/`) and the MuJoCo leg model (`model/leg.xml`) |
| `Network_Agent/networkA_mlp/`, `networkB_lstm/` | Network A (MLP) and Network B (LSTM) |
| `Network_Agent/agent/` | Training (`train.py`: PPO settings, wrappers, training command) and the residual action (`actions.py`) |
| `Network_Agent/common/` | Shared code: building the environment, the evaluation protocol, network inputs and policy adapters |
| `Network_Agent/trained_networks/` | Our trained networks, 3 seeds each (`final_s0`-`final_s2`): weights, normalisation statistics, config, learning curve and per-episode results |
| `Network_Agent/comparison/` | The controller comparison (fixed actions, stiffness-dip rules R1 and R3, both networks) and its saved results in `results/`: `episodes.csv`, `sweep.csv`, `summary.txt`, the run log, the grid search, the rule tuning and the figures |
| `Reports/` | Our individual final reports (`Final_Reports/`), the project proposal and three milestone reports (`Milestone_Submissions/`), and design justifications |

Each of `Gym/`, `Network_Agent/` and `Network_Agent/comparison/` has a short README of its own.

## Setup

```bash
conda env create -f environment.yml
conda activate stance
```

## Running the demo

Run from the repository root:

```bash
python demo.py                                # both networks, all trained seeds, 200 episodes per ground, then 5 episodes in the viewer
python demo.py --network lstm --model-seed 0  # only the chosen network, one seed
python demo.py --network mlp                  # only the MLP
python demo.py --episodes 50                  # 50 episodes per ground (seeds 1000-1049)
python demo.py --record demo.mp4              # also save one episode as a video
python demo.py --show 0                       # no viewer (for machines without a display)
python demo.py --train --steps 4096 --show 0  # train a fresh network briefly instead of loading ours
```

The per-episode results go to `results/demo_episodes.csv` and the summary to `results/demo_episodes_summary.csv`, inside the folder you run the demo from.

## Training a network

```bash
cd Network_Agent
python -m agent.train --network lstm
python -m agent.train --network mlp --seed 1 --steps 4096   # another seed, short run
```

The run is saved to `Network_Agent/runs/<network>/seed<N>/` (config, model and normalisation statistics), and the script ends with a 200-episode check on the training ground.

## Reproducing the comparison results

```bash
cd Network_Agent
python -m comparison.compare --out ../results_new   # all 6 controllers, both grounds, 200 episodes, plus the stiffness sweep (about 3 minutes on 16 cores)
python -m comparison.stats --out ../results_new > ../results_new/summary.txt
```

This writes to `results_new/` and leaves our saved results in `Network_Agent/comparison/results/` untouched; our regeneration matched them to within 5e-5. Without `--out`, both scripts use `Network_Agent/comparison/results/`.

## Notes

- If ROS is sourced in your shell, run `env -u PYTHONPATH python demo.py`.
- If the environment is not found, set `STANCE_GYM_PATH` to the `Gym/` folder.

## AI declaration

For the proposal, we used Claude Opus 5 (Anthropic) to evaluate ideas, summarise the research papers used to justify it and simplify technical concepts.

For the code, we used Claude (Anthropic) to evaluate ideas, merge our individual changes between files to avoid discrepancies, debug parts of our code and aid in writing only the very specific sections of our code that have been outlined clearly.

The video (`STANCE_demo.mp4`) was entirely run, recorded and edited by us. However, Claude was entirely responsible for coding the viewer's design and visuals (such as the arrows, colours and the ghost leg). We are responsible for the quality of work produced in the video but do not claim the visual improvements made by Claude as our own. No machine learning aspect of the project was altered during the creation of this video, no changes were made to the agent itself, and no data was altered by using Claude. The video shows the real simulation running our trained networks; no part of the footage is AI generated.

We are completely responsible for the content and quality of the submitted work.
