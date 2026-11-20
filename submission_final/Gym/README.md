# Gym

The STANCE environment: a MuJoCo leg with a powered ankle stepping onto compliant ground whose stiffness changes every episode and is hidden from the agent.

## Contents

| Path | What it is |
|---|---|
| `stance_env/__init__.py` | Registers `StanceAnkle-v0` (training ground) and `StanceAnkleTest-v0` (held-out test ground) |
| `stance_env/ankle_gym_env.py` | The Gymnasium environment: observation (10 frames × 11 sensors), action (target angle, stiffness K, damping B), reward and fall checks |
| `stance_env/ground.py` | The compliant ground model: contact forces and the ground picked each episode |
| `model/leg.xml` | The MuJoCo leg model |

## Run

There are no tests in this folder. To check that the environment loads, run this from `Gym/`:

```bash
python -c "import gymnasium as gym, stance_env; env = gym.make('StanceAnkle-v0'); print(env.reset(seed=0)[0].shape)"
```

It prints `(110,)`. The rest of the package finds this folder automatically.
