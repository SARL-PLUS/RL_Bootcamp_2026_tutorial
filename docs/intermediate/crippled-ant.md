---
title: Crippled Ant
---

# Crippled Ant — the environment

MuJoCo's `Ant-v5` with joints disabled: a quadruped that has to keep walking
after losing the use of a leg. It is the *reality gap* in miniature — the body
the policy was trained on is not the body it has to control.

This page is what the environment **is**. To train and evaluate on it, see
[Running the Crippled Ant](running.md).

!!! info "Prerequisites"
    A working [`rlbootcamp` environment](../setup.md). All commands assume it is
    active and that you are in the package directory:

    ```bash
    cd "sessions/02-intermediate/crippled-ant"
    ```

---

## Layout

```text
crippled-ant/
├── envs/crippled_ant.py       # CrippledAnt wrapper + make_ant factory
├── scripts/train.py           # Hydra training CLI (PPO / SAC)
├── scripts/evaluate.py        # evaluate a run, optionally under a new injury
├── scripts/render_agent.py    # render any checkpoint .zip to .mp4, any injury
├── scripts/transfer_benchmark.py   # injury-severity benchmark (Ex 3/4)
├── conf/                      # config.yaml, algo/{ppo,sac}.yaml, experiment/
└── tests/                     # contract + smoke tests
```

---

## The wrapper

```python
import sys; sys.path.insert(0, '.')
from envs import CrippledAnt, make_ant
import gymnasium as gym

# explicit joints…
env = CrippledAnt(gym.make("Ant-v5"), disabled_joints=[2, 3])
# …whole legs (a leg = hip + ankle action pair)…
env = make_ant(disabled_legs=[0])
# …or domain randomisation: new random legs every reset, fixed count
env = make_ant(n_random_legs=1)
# …or randomise the SEVERITY too: count sampled uniformly on every reset
env = make_ant(n_random_legs_max=4)   # 0 is in the support -- a healthy Ant
                                      # is part of the training distribution

obs, info = env.reset(seed=0)
info["disabled_joints"]          # verify what is actually disabled
info["n_disabled_legs"]          # how many legs — 0 under n_random_legs_max
```

Ant-v5 action layout — one `(hip, ankle)` pair per leg:

| leg | joints | position |
|---|---|---|
| 0 | 0, 1 | front left |
| 1 | 2, 3 | front right |
| 2 | 4, 5 | back left |
| 3 | 6, 7 | back right |

The wrapper zeroes **torques**, not observations: the policy still senses the
dead leg, it just cannot move it. Injuries are reproducible per seed, and the
active injury is reported in `info["disabled_joints"]` on every reset and step.

---

## The contract

```bash
pytest                # wrapper + config contracts (< 1 s)
pytest -m slow        # end-to-end train/evaluate/benchmark smoke (~2 min)
```

The suite defines the contract any modified injury implementation must keep:
torque zeroing, no caller-side action mutation, unchanged spaces, seed
reproducibility, and honest `info` reporting. If you change the wrapper in
Session #2, this is what tells you whether you broke it.

---

## Next

- [**Running the Crippled Ant**](running.md) — train, evaluate,
  transfer, sweeps.
- [**Session #2 — Intermediate**](index.md) — the exercises
  this environment exists for.
