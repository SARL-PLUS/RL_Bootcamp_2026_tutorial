---
title: Intermediate session
---

# Session #2 — Intermediate: Continuous Control & the Reality Gap

The intermediate session is about **robustness**: an agent that performs well in
training can fail badly when the world changes. You will train continuous-control
policies on MuJoCo and probe how they transfer to *damaged* variants of the same
robot — the **"Reality Gap."**

!!! abstract "Learning goals"
    - Train continuous-control agents with **PPO/SAC** (Stable-Baselines3).
    - Measure **zero-shot transfer**: train on a healthy robot, test on a
      crippled one.
    - Run a **hyperparameter sweep** (learning rate × batch size) with **Hydra**.
    - Read training curves critically — reward going up is not the whole story.

## The core exercise: Healthy → Crippled Ant

1. Train a PPO agent on the standard MuJoCo **`Ant-v5`**.
2. Evaluate it **unchanged** on "crippled" variants where one or more joints are
   disabled (actions to those joints are forced to zero).
3. Quantify the performance drop, then train **specialist** agents for the hard
   injuries and compare.

```python
import sys; sys.path.insert(0, '.')   # from sessions/02-intermediate/crippled-ant/
from envs import make_ant

healthy  = make_ant(include_cfrc_ext_in_observation=False)
crippled = make_ant(disabled_legs=[0], include_cfrc_ext_in_observation=False)

obs, info = crippled.reset(seed=0)
info["disabled_joints"]   # (0, 1) — front-left hip + ankle get zero torque
```

The runnable package lives in `sessions/02-intermediate/crippled-ant/` — see the
[Crippled Ant usage guide](running.md) for every exercise as a
one-line command. The exercise-to-command mapping in short:

| Exercise | Command |
|---|---|
| 1 PPO on healthy Ant | `python scripts/train.py` |
| 2 SAC comparison | `python scripts/train.py algo=sac` |
| 3 zero-shot transfer | `python scripts/transfer_benchmark.py --run healthy=runs/...` |
| 4 specialist | `python scripts/train.py env.disabled_legs=[0]` |
| 5 domain randomisation | `python scripts/train.py env.n_random_legs_max=1` |
| 6 grid search | `python scripts/train.py -m +experiment=sweep_lr_batch` |

## Hyperparameter sweeps with Hydra

Training is configured by [Hydra](https://hydra.cc): every value in
`conf/config.yaml` and `conf/algo/*.yaml` can be overridden on the command
line, and `-m/--multirun` turns comma-separated values into a grid:

```bash
python scripts/train.py -m \
    algo.learning_rate=1e-4,3e-4,1e-3 \
    algo.batch_size=64,256
```

Each combination writes to its own directory under `runs/multirun/` for
side-by-side comparison in TensorBoard (`tensorboard --logdir runs/`).

## Toolkit

- Environments: MuJoCo **`Ant-v5`** (also Swimmer, Hopper for extensions).
- Algorithms: **PPO**, **SAC** (Stable-Baselines3).
- Verify MuJoCo works:
  [the primer → MuJoCo](https://sarl-plus.github.io/rl-bootcamp-setup/setup/installation/#mujoco).

## Where to go next

- **[The Crippled Ant environment](crippled-ant.md)** — what the injuries are and
  how they are expressed.
- **[Running it](running.md)** — every exercise as a one-line command.
- **[Scripts and tooling](../reference/tooling.md)** — the complete inventory,
  including the evaluation and rendering tools this page does not use.

The session ends by handing over to Session #3: the *"is it really RL?"* litmus
test is measured on `Flight4DEnv`, the environment the
[Advanced session](../advanced/index.md) designs.
