---
title: Advanced session
---

# Session #3 — Advanced: Custom MDP Design & the Flight Challenge

In the advanced session you stop being a *user* of an environment and become its
**designer**. Using the [AirTraffic / `Flight4DEnv`](airtraffic.md)
codebase, you craft the observation space, action space and reward function, then
compete on a shared leaderboard.

!!! abstract "Learning goals"
    - Design an end-to-end **MDP**: observations, actions, reward, termination.
    - Balance **safety vs. efficiency** in the reward function.
    - Apply the **"Is it really RL?"** litmus test against a rule-based baseline.
    - Understand why **action masking** is built into the environment and the
      default policy, not an optional add-on.

## The challenge

`Flight4DEnv` simulates straight-line aircraft converging on a hotspot. Your job
is to resolve the conflicts with as little intervention as possible, while
keeping every aircraft on its 4D exit slot.

### Leaderboard objectives

Your agent is ranked on three objectives in **strict priority order** — a safer
submission always outranks a faster or cheaper one:

1. **Safety** — loss-of-separation events. Zero is the bar.
2. **Timeliness** — distance from each aircraft's original **4D exit point**.
   A clearance is a detour, not a new destination: slow an aircraft down and it
   must speed back up, step it down a level and it must climb back. This is what
   stops "slow everything down" from winning.
3. **Efficiency** — clearances issued, then how many distinct aircraft you touched.

We score with **our** KPI function on **our** secret seeds, never with your
reward. Shape your reward however you like; it moves the leaderboard only
through the behaviour it produces.

**→ [Full competition rules, submission format and scoring](competition.md)**

Start from [training](training.md)
and [evaluation](evaluation.md), then
iterate on the MDP — [the design sprint](design-sprint.md) is the menu of
changes worth trying.

## The "Is it really RL?" litmus test

The bar is `Priority4DController`, and it is worth reading before you train
anything — **[how the rule-based controller works](../reference/rule-based-controller.md)**
documents its algorithm, the arithmetic behind `SEP_LEVELS = 2`, and why it
does *not* use the two-level altitude macro even though that looks like the
obvious shortcut.

Before celebrating a policy, compare it against the hand-coded baseline **and**
against `noop`/`random` — `scripts/score_4d.py` puts all four in one table. If a
simple rule beats your trained agent, the agent has failed QA — the task either
doesn't need RL, or your reward/observation design is hiding the signal. This
mindset runs through the whole bootcamp.

## Action masking is not optional here

`Flight4DEnv` carries a legality mask on every step (`action_masks()`), and the
mask is also folded into the observation itself, from `env.mask_offset`
onward — not a side channel, and not something you opt into. The default
policy (below) reads it and masks the logits directly; there is no
`MaskablePPO`/`sb3-contrib` dependency to install, because the masking already
lives inside the environment and the policy, not bolted on around stock PPO.

If your design changes what counts as legal — [Tracks A, B and
D](design-sprint.md) all can — you extend `action_masks()`. You are not choosing
whether to mask, only what.

## The default policy is order-invariant, autoregressive, and masked

You do not have to build this — `policies.AutoregressivePolicy` ships as the
default for `scripts/train_4d.py`, and there is no second architecture to pick
between. It matters enough to understand before you change anything else.

Traffic is a **set**, not a vector. A flat `MlpPolicy` gives flight 0's
clearance its own weights, separate from flight 4's, so "an aircraft converging
on the hotspot should descend" has to be learned once *per slot*. Swap two
aircraft and a flat policy sees an unrelated observation even though the
situation is physically identical.

`AutoregressivePolicy` shares one per-aircraft MLP across flights, pools it
symmetrically for traffic context, then picks in two stages: **idle, or
*which* aircraft** (a `1 + n_flights`-way choice — an aircraft with no legal
clearance is unselectable), then *which clearance for that aircraft*, with
stage two reading the chosen aircraft's own embedding and that aircraft's row
of the mask. The environment only ever sees the flat index, `0` or
`1 + 7·i + c`. Reordering the flights reorders the outputs
identically — permutation equivariance holds **by construction**, not by
training. Illegal clearances are masked at the logits from the environment's
own mask, which drives invalid actions to **exactly zero** — `Flight4DEnv`
counts them (`n_invalids`) precisely so a broken policy is loud rather than
silent, not because a trained one is expected to produce any.

Your job is the **observation and the reward**. The architecture ships as the
default because it is not the interesting variable here — the interface was.

## If your learning curve is flat, read this first

The environment's own defaults already dodge the failure mode that used to cost
a day here: `idle_bias=2.5` biases initialisation toward idling
(`p(idle) = e^b/(e^b + n_flights)` ≈ 0.71 at the default), and **at most one
aircraft is commanded per step** — there is no way for a fresh policy to spend
its whole clearance budget in the first dozen steps the way a per-aircraft
action vector could. The specific pathology this section used to describe
cannot happen by construction.

What can still go wrong is the same failure in a different shape: a *fixed*
entropy coefficient pushes the policy toward one of two collapses. Too high,
and the clearance rate inflates until `max_clearances` truncates the episode
before it reaches the horizon. Too low, and the policy collapses onto idling
and never explores past it. Neither failure is about the reward — both are
about how much the policy is exploring, which `--ent-coef` only guesses at.

!!! tip "The one-line fix"
    ```bash
    python scripts/train_4d.py --auto-entropy --auto-entropy-target 0.35
    ```

    Tunes `ent_coef` in log space so the policy's entropy tracks a target
    fraction of `ln(n_actions)`, instead of holding a fixed guess for the whole
    run. This is the environment's own answer to "my curve looks flat" —
    ported from a prior project that failed in both directions before landing
    on this.

!!! warning "The general lesson, which is worth more than the fix"
    Before tuning a reward, check what the policy is actually doing at
    initialisation, and what the environment lets it experience. `idle_bias`
    and single-command actions were both design decisions that head off a
    known failure mode; if you change either, the failure mode is not
    guaranteed to still be off the table.

## Framework note

The training stack is **Stable-Baselines3 / PPO** throughout — `scripts/train_4d.py`,
the rule-based baseline, and the competition harness all build on it.

## What the session actually looks like

Two decks. **`advanced-intro`** is ninety minutes: ten on why air traffic
control, and why a deliberately simplified version of it (straight lines,
obedient pilots, altitude that costs nothing — with *timeliness* kept as the
one secondary objective); thirty on the simulator and the MDP, with [every
clearance shown on its own](airtraffic.md#the-action-vocabulary-one-clip-at-a-time);
twenty on whether you need RL at all, with the
[rule-based controller](../reference/rule-based-controller.md) introduced
step by step; fifteen on what is wrong with the current solution and where you
could take it. **`advanced-challenge`** is twenty to thirty minutes of
mechanics — [the harness, the submission, the board](kaggle.md) — and
introduces nothing new. Then the room works.

- **[The design sprint](design-sprint.md)** — the four tracks you can pick from,
  the format, and what you hand in.
- **[The competition](competition.md)** — what your change has to move, and how
  it is measured.
- **[The Kaggle leaderboard](kaggle.md)** — getting a number on the board.

Everything the older `advanced` deck lectured through is still there, in
`advanced-intro`'s labelled appendix sections a presenter can jump to during
questions.
