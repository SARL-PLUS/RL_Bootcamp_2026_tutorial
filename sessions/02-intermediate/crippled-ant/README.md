# Crippled Ant — Continuous Control & the Reality Gap

MuJoCo Ant with disabled joints, for Session #2 of RL Bootcamp 2026.
The core exercise: train on the healthy `Ant-v5`, deploy on a crippled one,
and quantify how badly the policy transfers.

## Structure

```
crippled-ant/
├── envs/
│   └── crippled_ant.py        # CrippledAnt wrapper + make_ant factory
├── scripts/
│   ├── train.py               # Hydra CLI: PPO/SAC, healthy or crippled
│   ├── evaluate.py            # evaluate a run, optionally under a new injury
│   ├── render_agent.py        # render any checkpoint .zip to .mp4, any injury
│   └── transfer_benchmark.py  # Ex 3/4: injury-severity benchmark
├── conf/
│   ├── config.yaml            # env + training defaults
│   ├── algo/{ppo,sac}.yaml    # per-algorithm hyperparameters
│   └── experiment/sweep_lr_batch.yaml   # Ex 6 grid search
└── tests/                     # contract + smoke tests (run: pytest)
```

## Quick start

```bash
conda activate rlbootcamp
```

The `rlbootcamp` environment covers everything here — see the
[setup guide](https://sarl-plus.github.io/rl-bootcamp-setup/setup/installation/) if you have not created it yet.

```python
import sys; sys.path.insert(0, '.')
from envs import make_ant

env = make_ant(disabled_legs=[0], include_cfrc_ext_in_observation=False)
obs, info = env.reset(seed=0)
print(info["disabled_joints"])   # (0, 1) — front-left hip + ankle
```

The wrapper zeroes the *torques* of disabled joints — the observation is
untouched. A robot with a dead motor still has its encoders; the policy must
notice the leg no longer responds.

## The exercises, as commands

| Exercise | Command |
|---|---|
| 3.1 PPO on healthy Ant | `python scripts/train.py` |
| 3.2 SAC comparison | `python scripts/train.py algo=sac` |
| 4.1 zero-shot transfer | `python scripts/transfer_benchmark.py --run healthy=runs/PPO_Ant-v5/<ts>` |
| 4.2 specialist | `python scripts/train.py env.disabled_legs=[0]`, then benchmark both runs |
| 4.3 domain randomisation | `python scripts/train.py env.n_random_legs=1` |
| 5.1 hyperparameter grid | `python scripts/train.py -m +experiment=sweep_lr_batch` |

Monitor any run with `tensorboard --logdir runs/`.

## Watch a policy

`render_agent.py` takes **any** checkpoint `.zip` (best, final, or periodic),
rebuilds its training environment from the run's saved config (VecNormalize
stats included), and records `.mp4`s — optionally under an injury the policy
never trained on:

```bash
python scripts/render_agent.py runs/PPO_Ant-v5/<ts>/best_model/best_model.zip
python scripts/render_agent.py <zip> --disabled-legs 0     # healthy policy, injured ant
python scripts/render_agent.py <zip> --n-random-legs 2 --episodes 3 --stochastic
```

Videos land in `<run>/videos/`, named after the checkpoint + injury + mode.

## Injury modes

- `disabled_joints=[2, 3]` — exact joint indices (see table in `envs/crippled_ant.py`).
- `disabled_legs=[0]` — whole legs; each leg is a (hip, ankle) action pair.
- `n_random_legs=1` — new random legs every reset (domain randomisation).

## Normalisation notes (QA!)

- PPO runs use `VecNormalize` (obs + reward). The statistics are saved to
  `vecnormalize.pkl` and **must** be loaded at evaluation time —
  `scripts/evaluate.py` does this for you. Evaluating a PPO policy without
  its normalisation stats silently produces garbage scores.
- SAC runs train *without* VecNormalize: its replay buffer would mix
  transitions normalised with different statistics.

## Tests

```bash
pytest                # wrapper + config contract tests (fast)
pytest -m slow        # end-to-end train/evaluate/benchmark smoke tests (~2 min)
```

If you modify the wrapper or configs, keep this suite green — it is the
definition of "working" used throughout the session.
