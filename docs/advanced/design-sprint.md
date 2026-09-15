---
title: The design sprint
---

# The design sprint — what you actually change

The Advanced session is a 30-minute tour followed by **the sprint**: you pick one
design change to `Flight4DEnv`, implement it, train against it, and find out
whether it moved the leaderboard. This page is the menu.

!!! abstract "The one rule that shapes everything else"
    The **physics are frozen** and the **score is ours**. You may redesign the
    observation, the action space, the reward, the network and the training
    regime. You may not change how aircraft move or what counts as a loss of
    separation — see [what you may not
    change](competition.md#how-your-submission-is-scored).

    So every track below is a bet about the same underlying problem: *can the
    agent see it, can it say it, and is it paid for getting it right?*

## Format

| Time | What |
|---|---|
| 15 min | Pick a track. Read the part of `envs/flight_4d.py` it touches. |
| 30 min | Implement it. You write the core logic; the surrounding code exists. |
| 15 min | Train a fixed budget (200k steps is enough to separate "broken" from "promising") and score it against the baselines. |
| 10 min | Present: what changed, what improved, what broke. |

Work in pairs. One person drives, one person keeps the baseline honest — the
second job is the one people skip and then regret.

!!! warning "200k steps does not tell you whether a design is good"
    It tells you whether it is *alive*. The reference run is 4M steps and 80
    minutes. A sprint result that looks flat may be a design that needed
    twenty times the budget, and a sprint result that looks great may be a
    policy that found something cheap. Say which one you think you have.

---

## Before you pick: the six ways MDP design goes wrong

Every one of these has happened in this repository's own history, and the deck's
*eleven mistakes* appendix walks through the arithmetic that caught each.

| Pitfall | What it looks like | Consequence |
|---|---|---|
| **Reward hacking** | The agent maximises the reward in a way you did not intend | High reward, useless behaviour |
| **Observation aliasing** | Two genuinely different situations look identical to the agent | The policy cannot separate them, so it averages them |
| **Action mismatch** | Actions too coarse to express the fix, or too fine to search | Slow learning, or an unreachable optimum |
| **Sparse reward** | The signal only arrives at the end — or never | A bonus that never fires is a constant, not a gradient |
| **Dense but misleading** | Shaping that points at a local optimum | Converges confidently to the wrong behaviour |
| **Mis-specified γ** | Too low is short-sighted; too high is slow and unstable | Either way the credit does not reach the decision that mattered |

!!! tip "The cheapest diagnostic in the whole session"
    Log how often each reward component is **non-zero**. A term that never fires
    is not shaping anything, and you will otherwise spend a training run
    discovering that.

---

## Track A — the observation

**Today:** each aircraft contributes a fixed feature block, squashed into
`[-1, 1]`, with the legality mask appended from `env.mask_offset` onward. See the
[full feature list](../reference/environment-api.md#observation-space).

Ideas worth trying:

- **Relative geometry.** Replace (or augment) absolute positions with offsets to
  the conflict partner: Δx, Δy, Δaltitude. Translation-invariant, and the thing
  the agent actually has to reason about.
- **Time to closest point of approach.** A derived feature giving the agent a
  planning horizon it currently has to infer. `CollisionCourseSimulator` already
  computes conflict geometry — see
  [`predict_conflicts`](../reference/environment-api.md#predict_conflicts).
- **Time-to-exit and slack.** How much of its 4D budget each aircraft still has.
  The score is half timeliness; the observation barely mentions it.
- **Pairwise / graph structure.** Aircraft as nodes, conflicts as edges. The
  policy is already permutation-equivariant, so this is a smaller step than it
  sounds — and a bigger implementation job than 30 minutes allows.

!!! danger "The trap in this track"
    Adding a feature changes the observation shape, which means **every existing
    checkpoint stops loading**. Start from scratch and say so in your writeup;
    do not spend the sprint debugging a shape mismatch.

## Track B — the action space

**Today:** `Discrete(1 + 7 * n_flights)`. Index 0 idles; every other index names
one aircraft and one of seven clearances, so **at most one aircraft is commanded
per step**. See the [clearance
vocabulary](../reference/environment-api.md#action-space).

Ideas worth trying:

- **A different vocabulary.** Three-level altitude macros, a longer or shorter
  `hold_steps`, an asymmetric speed pulse. Cheap to try, and the auto-revert
  machinery already exists.
- **A permanent speed clearance.** Every speed command today is a symmetric
  pulse precisely so it cannot wreck the 4D slot. Removing that guardrail makes
  the timeliness objective genuinely hard — which may be the interesting version.
- **Multi-aircraft commands.** Let one step re-task two aircraft. Field note #5
  (in the deck) is exactly this: on the hardest scenarios the *interface* was the
  binding constraint, not the policy.

!!! warning "Widening the action space is not free"
    Every index you add is one more thing to explore, and the clearance budget
    (`max_clearances`) does not grow with it. Measure the clearance count, not
    only the safety number.

## Track C — the reward

**Today:** a potential-based safety term, a per-clearance charge, and a mid-air
penalty that scales with the steps left unflown. The full arithmetic is in
[reward design](../reference/reward-design.md).

Ideas worth trying:

- **Reward the worst case, not the average.** The reward averages congestion
  across the sector, so an agent can look excellent while one unlucky pair sits
  in permanent conflict. Field note #2: adding an explicit worst-case term broke
  a ceiling that seven runs could not.
- **Curriculum shaping.** Dense early, annealed toward sparse. `anneal.py` in the
  reference solution is a worked example.
- **Constraint separation.** Price safety and efficiency separately rather than
  summing them — Lagrangian-style, with the multiplier tuned during training.
- **Counterfactual credit.** Pay for a clearance that provably resolved a
  conflict; charge for one that did not.

!!! tip "Size the terms before you spend a run"
    ```bash
    python scripts/calibrate_4d.py
    ```
    The first weights on this environment were chosen by taste and were wrong by
    a factor of 86 in a way that was knowable in thirty seconds. Do not repeat
    that.

## Track D — the dynamics *(training only)*

**Today:** straight-line flight, deterministic, no wind, no sensor noise.

You may add **any** of these to your *training* environment:

- heading drift, ±σ per step;
- a constant or time-varying wind field;
- Gaussian noise on the observed positions and velocities;
- aircraft entering and leaving the sector mid-episode.

!!! danger "None of it travels to the scored run"
    The leaderboard runs **our** simulator. Domain randomisation is a legitimate
    and often excellent robustness strategy, and it is scored purely through the
    policy it produces. If your sector has different weather than your
    neighbour's, the ranking between you means nothing — which is why the physics
    are frozen.

    This makes Track D the one track whose success is measured *entirely*
    out-of-distribution. That is either the point or the problem, depending on
    your hypothesis.

---

## What you do **not** have to build

Two design decisions that used to be exercises now ship as defaults. Read them
before you change anything underneath them — the
[session guide](index.md#action-masking-is-not-optional-here) covers both.

- **Action masking** is in the environment (`action_masks()`) *and* folded into
  the observation, and the default policy masks its logits from it. There is no
  `sb3-contrib` / `MaskablePPO` dependency, because the masking is not bolted on
  around stock PPO. If your design changes what counts as legal — Tracks A, B
  and D all can — you extend `action_masks()`. You are choosing *what* to mask,
  never *whether*.
- **Permutation equivariance** is in `policies.AutoregressivePolicy`: one shared
  per-aircraft encoder, symmetric pooling, then *which aircraft* and *which
  clearance* in two stages. Reorder the flights and the outputs reorder
  identically, by construction rather than by training.

---

## Deliverables

| Deliverable | What it is |
|---|---|
| **Design note**, one page | Which track, what you changed, what you expected to happen |
| **The changed files** | Your `envs/`, and `policy/` if you wrote any — we run your code, not a guess at it |
| **A scored run** | `scripts/score_4d.py` against `noop`, `random` and `rule-based`, on the same seeds |
| **A submission** | [`submission.csv` for the public board](kaggle.md) — upload anytime, as often as you like; and [the zip for the private one](competition.md#what-you-submit) — one upload, to the shared Drive folder, by the sprint deadline |
| **Five minutes** | What worked, what did not, what you would do with another day |

The design note is graded on the hypothesis, not the result. A track that failed
for a reason you can state is worth more here than a number that went up and you
cannot explain.

!!! info "Two submissions, two rhythms"
    Kaggle is self-service and live all session — build, check, upload, repeat,
    as much as you want. It is where you publish a result and see it sit next
    to the brick and the rule-based baseline in public.

    The zip is the opposite: one delivery, once, at the close, and it is
    **not** scored on Kaggle at all — we run it ourselves, offline, each zip
    in its own process, because a team's code can change the environment
    itself and a shared sandbox can't run twenty different copies of `envs/`
    safely. That is the run that decides the trophies. Validate it yourself
    first — `competition.ingest --zip ... --contract`,
    [see the rules](competition.md#what-you-submit) — so the one upload that
    decides the trophies is not also the first time your submission has
    actually been scored.

## Related

- [Competition rules and scoring](competition.md) — what the change has to move.
- [The Kaggle leaderboard](kaggle.md) — how to get a number on the board.
- [Environment API](../reference/environment-api.md) — the exact spaces you are editing.
- [Reward design](../reference/reward-design.md) — the maths Track C rewrites.
- [Scripts and tooling](../reference/tooling.md) — every command in the repo.
