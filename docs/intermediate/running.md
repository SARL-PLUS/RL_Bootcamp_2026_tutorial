---
title: Crippled Ant — running
---

# Crippled Ant — training, evaluation & transfer

!!! info "Prerequisites"
    A working [`rlbootcamp` environment](../setup.md), active, and you are in
    the package directory:

    ```bash
    cd "sessions/02-intermediate/crippled-ant"
    ```

    New to the environment? Read
    [what it is](crippled-ant.md) first.

---

## Training

```bash
python scripts/train.py                         # PPO, healthy Ant (Ex 1)
python scripts/train.py algo=sac                # SAC (Ex 2)
python scripts/train.py env.disabled_legs=[0]   # specialist (Ex 4)
python scripts/train.py env.n_random_legs=1     # domain randomisation, fixed severity (Ex 5)
python scripts/train.py env.n_random_legs_max=1 # domain randomisation over severity too
```

Runs land in `runs/<ALGO>_Ant-v5/<timestamp>/` with TensorBoard logs (`tb/`),
the best checkpoint (`best_model/`), periodic checkpoints, `final_model.zip`
and — for PPO — `vecnormalize.pkl`.

!!! warning "VecNormalize is part of the model"
    PPO trains with normalised observations and rewards. The saved
    `vecnormalize.pkl` **must** be loaded at evaluation time, otherwise the
    policy sees observations on a different scale and scores garbage.
    `scripts/evaluate.py`, `scripts/render_agent.py` and
    `scripts/transfer_benchmark.py` handle this automatically. SAC
    deliberately trains without VecNormalize (off-policy replay and moving
    statistics don't mix).

!!! danger "…and the statistics must match the *checkpoint*, not just the run"
    Normalisation statistics **move during training**, so every checkpoint
    needs the snapshot that was in force when it was saved. SB3's
    `EvalCallback` writes `best_model.zip` but **not** the statistics behind
    its score — pair it with the end-of-training `vecnormalize.pkl` and the
    policy silently sees the wrong scale.

    We measured it on the 3M-step Ant: the same policy scored
    **2570 ± 55** with matched statistics, but **1964 ± 738** (worst episode
    286 — it falls over) when mispaired. Not a crash; just quietly worse and
    wildly inconsistent.

    Our `train.py` now saves `best_model/vecnormalize.pkl` alongside every new
    best model, and the scripts warn loudly when only a fallback exists (older
    runs). Rule of thumb: **`--model final` is always a matched pair**;
    periodic checkpoints carry their own `rl_model_vecnormalize_<n>_steps.pkl`.

---

## Evaluation and transfer

```bash
# evaluate on the training environment
python scripts/evaluate.py --run runs/PPO_Ant-v5/<ts>

# zero-shot transfer: healthy-trained policy on a crippled Ant
python scripts/evaluate.py --run runs/PPO_Ant-v5/<ts> --disabled-legs 0

# sample the policy instead of taking the argmax action
python scripts/evaluate.py --run runs/PPO_Ant-v5/<ts> --stochastic

# record videos
python scripts/evaluate.py --run runs/PPO_Ant-v5/<ts> --render
```

**Watch any checkpoint** — `render_agent.py` takes a checkpoint `.zip`
directly (best, final, or a periodic `checkpoints/rl_model_*_steps.zip`),
rebuilds its training env + VecNormalize stats, and records `.mp4`s. The
injury flags let you watch a healthy-trained policy cope (or not) with a
crippled body — transferability and robustness by eye:

```bash
python scripts/render_agent.py runs/PPO_Ant-v5/<ts>/best_model/best_model.zip
python scripts/render_agent.py <zip> --disabled-legs 0          # zero-shot injury
python scripts/render_agent.py <zip> --n-random-legs 2 --episodes 3
python scripts/render_agent.py <zip> --stochastic --seed 7
```

The **transfer benchmark** (Exercises 3/4) evaluates checkpoints across
healthy, every 1-leg and every 2-leg injury with identical episode seeds, and
writes a CSV + bar chart:

```bash
python scripts/transfer_benchmark.py \
    --run zero-shot=runs/PPO_Ant-v5/<healthy_ts> \
    --run specialist=runs/PPO_Ant-v5/<crippled_ts> \
    --episodes 20
```

---

## Hyperparameter sweeps (Ex 6)

```bash
python scripts/train.py -m +experiment=sweep_lr_batch
# or spell the grid out yourself:
python scripts/train.py -m algo.learning_rate=1e-4,3e-4,1e-3 algo.batch_size=64,256
```

---

Next: [Session #2 — Intermediate →](index.md)
