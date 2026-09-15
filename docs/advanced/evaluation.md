---
title: AirTraffic — evaluation
---

# AirTraffic — evaluating an agent

`scripts/score_4d.py` is the one tool you need. It rolls agents over a fixed set
of held-out seeds, prints the KPIs the competition ranks on, and puts the
baselines in the same table for free.

!!! info "Prerequisites"
    The `rlbootcamp` environment active, and run from inside
    `sessions/03-advanced/airtraffic/`.

---

## Score

```bash
conda activate rlbootcamp
cd sessions/03-advanced/airtraffic

python scripts/score_4d.py --agent noop --agent random --agent rule-based \
    --agent runs/mine/final_model.zip
```

| Flag | Default | Meaning |
|---|---|---|
| `--agent` | *(repeatable)* | `noop`, `random`, `rule-based`, or a path to a `.zip` |
| `--seeds` | 100 held-out | how many seeds to score, held fixed across agents |

The ranking is **lexicographic, failures first**. That ordering is load-bearing:
an episode that ends in a mid-air stops accumulating conflict-steps, so ranking
on congestion alone would make crashing out a strategy.

Both scoring modes are always reported. `idle_bias` makes idling the argmax at
initialisation, so a *deterministic* rollout of a weak policy is byte-identical
to doing nothing — three runs in this project were written off as collapsed on
exactly that evidence and all three were still learning.

---


## Common recipes

??? example "Run the do-nothing baseline and report KPIs"
    ```python
    import sys; sys.path.insert(0, ".")
    from envs import Flight4DEnv

    env = Flight4DEnv()                     # n + m = 5, n ~ U(2, 4)
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(0)   # 0 idles
        done = term or trunc
    print("congestion:", info["congestion_total"], "exit_miss:", info["exit_miss"])
    ```

??? example "Load a trained model and act greedily"
    ```python
    import sys; sys.path.insert(0, ".")
    from stable_baselines3 import PPO
    from envs import Flight4DEnv

    env = Flight4DEnv()
    model = PPO.load("runs/mine/final_model.zip", device="cpu")
    obs, _ = env.reset(seed=1)
    action, _ = model.predict(obs, deterministic=True)
    ```

??? example "Only choose among legal actions"
    ```python
    import numpy as np
    legal = np.flatnonzero(env.action_masks())
    ```
    The same mask is carried inside the observation from `env.mask_offset`
    onward, which is how the policy sees it.

??? example "Change the price of a clearance"
    ```python
    env = Flight4DEnv(w_safe=3.0, w_clearance=0.30)
    ```
    Or at training time: `--w-safe 3.0 --w-clearance 0.30`. Only the *ratio*
    matters: `w_clearance` is the price of one clearance against one
    severity-1 infringement.

---

## Rendering episodes to video

```bash
python scripts/render_4d.py --agent noop --agent random --agent rule-based \
    --agent checkpoints/ppo_4M_curriculum.zip --seed 900000
```

| Flag | Default | Meaning |
|---|---|---|
| `--agent` | *(repeatable)* | `noop`, `random`, `rule-based`, a scripted demo (`split`, `macro`, `speed`, or one clearance: `fl_inc`, `fl_dec`, `fl_inc2_t`, `fl_dec2_t`, `spd_dn_t`, `spd_up_t`, `resume`), or a path to a `.zip` |
| `--seed` | `900000` | the first held-out seed; draws `n=3, m=2` at the defaults |
| `--n-range LO HI` | env default | pin the converging-traffic draw, e.g. `2 2` |
| `--n-flights N` | `5` | sector size `n + m`. `--n-flights 2 --n-range 2 2` puts two aircraft on screen and nothing else — the per-action demos on [the environment page](airtraffic.md#the-action-vocabulary-one-clip-at-a-time) |
| `--stochastic` | off | sample actions instead of taking the argmax |
| `--ext` | `mp4` | `gif` for the handbook, `mp4` for the decks |
| `--fps` / `--out` | `5` / `renders/` | |

`Flight4DEnv` itself declares no render modes, so `render_4d.py` drives it from
the outside: step the environment, record what happened, then replay through
`CollisionCourseSimulator`'s shared geometry. The two KPI bars beside the plot
are labelled underneath rather than up their y-axes, so the labels never sit
over the traffic or over each other. Every proximity ring is dashed
while a pair is still legally separated and solid once it is not, sized and
weighted by severity, with a fading trail so a resolved conflict stays visible
after the aircraft have moved on. Videos land in `renders/` (gitignored);
`slides/scripts/make_media.py` stages the posters and filmstrips the decks
actually embed.

---

## Where to go next

- **Is it really RL?** Compare against the
  [rule-based baseline](baseline.md) before believing any number.
- **Reference:** exact spaces, reward maths, and public methods →
  [Environment API](../reference/environment-api.md).
- **Design:** what you are allowed to change →
  [Advanced session](index.md).
