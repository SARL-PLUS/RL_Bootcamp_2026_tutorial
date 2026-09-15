---
title: AirTraffic — training
---

# AirTraffic — training an agent

`scripts/train_4d.py` trains PPO (Stable-Baselines3) on `Flight4DEnv`: a
warm-up + cosine learning-rate schedule, periodic checkpoints, and a KPI
callback that evaluates on the actual scored objectives — never on the
training curve, which contains shaping the leaderboard does not.

!!! info "Prerequisites"
    A working [`rlbootcamp` environment](../setup.md), active, and you are in
    the package directory:

    ```bash
    cd "sessions/03-advanced/airtraffic"
    ```

    New to the environment? Read
    [what it is](airtraffic.md) first.

---

## Train

```bash
OMP_NUM_THREADS=1 python scripts/train_4d.py --steps 1500000 --out runs/mine
```

`OMP_NUM_THREADS=1` is not optional if you run more than one job at once. Torch
otherwise opens ~46 threads per process; two such processes on 16 cores
measured **31–39 fps** against **428 fps** pinned — an 11× difference that
looks exactly like a slow environment and is not.

!!! tip "Want a fast sanity run?"
    ```bash
    OMP_NUM_THREADS=1 python scripts/train_4d.py --steps 60000 --out runs/smoke
    ```
    Finishes in about a minute on a laptop CPU — enough to confirm checkpoints,
    logging and the eval callback all work before committing to a long run.

## The settings, and why each one is there

Everything is a flag, not a code change — the exercise is choosing between
options that already exist, not implementing them.

| Flag | Default | Meaning |
|---|---|---|
| `--steps` | `1_500_000` | total environment steps |
| `--out` | *(required-ish)* | run directory: `<out>/tb/`, `<out>/checkpoints/`, `<out>/final_model.zip` |
| `--seed` | `123` | |
| `--n-envs` | `8` | parallel environments. More is not always faster — 16 measured *slower* than 8 here, because the bottleneck is the policy forward/backward pass, not environment stepping |
| `--n-steps` / `--n-epochs` / `--batch-size` | `256` / `10` / `256` | PPO rollout length, update epochs, minibatch size |
| `--lr` / `--final-lr` / `--warmup-frac` | `3e-4` / `3e-5` / `0.01` | warm-up-then-cosine learning-rate schedule |
| `--gamma` | `0.99` | discount |
| `--ent-coef` | `0.01` | entropy bonus, ignored if `--auto-entropy` is set |
| `--target-kl` | `0.03` | PPO's trust-region cutoff |
| `--hidden` | `128` | width of the autoregressive policy's shared encoder |
| `--idle-bias` | `2.5` | `p(idle) = e^b/(e^b + n_flights)`; `2.5` gives ≈0.71 at initialisation |
| `--hold-steps` / `--w-safe` / `--w-clearance` | environment defaults | override the environment's own reward/timing constants; leave unset unless you are deliberately re-tuning them |
| `--eval-freq` / `--eval-episodes` | `250_000` / `20` | how often, and on how many episodes, the KPI callback scores the policy |
| `--subproc` | off | `SubprocVecEnv` instead of `DummyVecEnv` — measured ~1% faster here, not worth the extra process overhead unless your environment is heavier than this one |
| `--checkpoint-every` | `500_000` | timesteps between checkpoints (`0` disables). A 4M-step run is 75+ minutes; without this, an interruption at 3.9M steps loses everything |
| `--curriculum` / `--curriculum-n-start` / `--curriculum-frac` | off / `2` / `0.2` | start on thin traffic (`n` pinned at `--curriculum-n-start`) and widen the per-episode draw of `n` to the environment's own `n_range` over `--curriculum-frac` of training. `n + m` is fixed either way, so the observation and action spaces never change |
| `--auto-entropy` / `--auto-entropy-target` | off / `0.35` | tune `ent_coef` so the policy's entropy tracks a target fraction of `ln(n_actions)`, instead of holding `--ent-coef` fixed. A *fixed* coefficient has failed in both directions on this environment: too high inflates the clearance rate until the budget truncates the episode, too low collapses the policy onto idling |

!!! tip "What was deliberately left out"
    There is one action space, one observation, one policy architecture — no
    `--policy-head` or `--action-set` choice, and no observation frame
    stacking. The first-generation environment offered a menu across all three,
    and the menu was the problem: every combination was a different experiment
    and none of them were comparable. `Flight4DEnv`'s hard masking and
    auto-reverting clearances make most of that menu unnecessary rather than
    merely hidden.

## The reference run

```bash
OMP_NUM_THREADS=1 python scripts/train_4d.py \
    --steps 4000000 --out runs/ref --target-kl 0.05 \
    --curriculum --curriculum-frac 0.2 \
    --auto-entropy --auto-entropy-target 0.35 \
    --checkpoint-every 500000
```

~80 minutes on 16 CPU cores at ~1000 fps. This is what produced
`checkpoints/ppo_4M_curriculum.zip` and the row the board and both decks quote.

## Where results go

```text
runs/<out>/
├── tb/                     # TensorBoard logs  ->  tensorboard --logdir runs/
├── checkpoints/
│   └── v4d_<N>_steps.zip   # periodic, every --checkpoint-every timesteps
└── final_model.zip         # end of training
```

## Watch it learn

```bash
tensorboard --logdir runs/
# open http://localhost:6006
```

Do not trust `rollout/ep_rew_mean` alone. Two runs in this project's history
were called "learning" from a rising `ep_len_mean` while the policy had in fact
collapsed to inaction — only the `[eval]` lines the KPI callback prints, scored
on `congestion_total`, `exit_miss`, `clearances` and `collided`, are comparable
to a baseline. Both deterministic and stochastic modes are always reported:
`idle_bias` makes idling the argmax at initialisation, so a deterministic
rollout of an untrained policy is byte-identical to doing nothing, and three
runs here were written off as collapsed on exactly that evidence.

---

Next: [Evaluation & rendering →](evaluation.md)
