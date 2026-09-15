---
title: Rule-based baseline
---

# Approaching `Flight4DEnv` programmatically — the rule-based baseline

*"Prove you need RL."* Before training anything, the litmus test asks whether a
hand-written controller already solves the problem. This page explains how to
attack [`Flight4DEnv`](../reference/environment-api.md) with plain code, and
documents the shipped baseline that RL agents must beat.

!!! abstract "Why a hand-written controller ships at all"
    Not because it is good. Because without one, *"the learner beats do-nothing
    and a random legal policy"* is the strongest claim anyone can make — and that
    is a very low bar. The brick never touches an aircraft and the random policy
    crashes out of a third of its episodes. `Priority4DController` gives the
    leaderboard something **competent** to sit above.

## 1. The four ingredients

Any programmatic controller needs exactly four pieces of knowledge, all
derivable from the environment's public interface.

**① Decode the observation.** The flat vector is normalised; invert it back to
physical units. `agents.rule_based_4d.decode(obs, n_flights, max_steps,
hold_steps)` returns per-aircraft `level` (flight-level index), `severity`,
`hold_left`, `t_conflict` (seconds), `d_alt` (levels off plan) and `x, y`
(metres), plus the episode `step`. The controller sees exactly what the RL agent
sees; no peeking into the simulator.

**② Read the legality mask, do not re-derive it.** The mask is part of the
observation — `obs[env.mask_offset:] > 0.5` — so a controller never has to work
out for itself whether a climb fits under the ceiling or whether a speed change
would make the exit slot unreachable. Propose an action, check the mask, fall
through to the next candidate. Invalid clearances then cost you nothing because
you never issue one.

**③ Know the conflict rule.** Two aircraft are *congested* — counted every step
it persists — iff **both**:

- `|Δalt| < ALT_MIN_SEP` (24, i.e. two 12-unit flight levels), **and**
- horizontal distance ≤ **1000 m**.

Altitudes live on 10 discrete levels (290–398). **Two** levels of separation is
safety; one is not, and that single fact drives most of the design.

**④ Know what you are ranked on.** Failures first, then congestion, then
`exit_miss` (the 4D exit gate), then clearances, then distinct aircraft touched.
Every one of those is in the `info` dict — see the
[API reference](../reference/environment-api.md#the-info-dict). A controller that
optimises the wrong column is easy to write and hard to notice.

**⑤ Exploit the scenario structure.** All converging traffic enters *at the same
altitude*; background traffic already crosses at other levels. With 10 levels and
two-level separation, the conflict-free slots are base, ±2, ±4.

## 2. The shipped baseline

`agents.Priority4DController` implements priority-based altitude-slot assignment
with an explicit recovery gate. The summary below is the shape of it;
**[the rule-based controller](../reference/rule-based-controller.md)** documents
the algorithm in full, including the two bugs it was measured into and the four
things it deliberately does not do.

1. **Assign** each converging aircraft its own conflict-free slot — earliest
   predicted arrival keeps its level, later arrivals move progressively further.
   The slot list is built once from the shared entry level, so no two aircraft
   are ever assigned the same slot.
2. **Steer, one level per step** — the environment commands at most one aircraft
   per step, so the most urgent gets the clearance.
3. **Come home when clear** — once an aircraft is `RECOVERY_SEP_FACTOR` hotspot
   radii from every other converging flight, one `RESUME` restores its level and
   speed. A clearance is a detour, not a new destination.

!!! warning "It does not use the two-level macro, and that is a measured decision"
    `FL_INC2_T` looks strictly better — two levels, one clearance, comes home by
    itself — and it is **worse than a random legal policy**. The macro reverts on
    a fixed timer the controller does not control, so if the traffic has not
    cleared it drops the aircraft straight back onto it. Read
    [the full diagnosis](../reference/rule-based-controller.md#why-it-does-not-use-the-two-level-macro)
    before you design your own action set; it is the single most transferable
    result on these pages.

Both controllers expose the SB3 `predict(obs, deterministic=...)` interface, so
heuristics, checkpoints and competition submissions run through the same
evaluation loop.

## 3. Run the litmus test

```bash
cd sessions/03-advanced/airtraffic
conda activate rlbootcamp

# the three reference agents, 100 frozen seeds
python scripts/score_4d.py --agent noop --agent random --agent rule-based

# put your own checkpoint on the same table
python scripts/score_4d.py --agent noop --agent rule-based \
    --agent runs/v4d_s123/final_model.zip
```

| Flag | Default | Meaning |
|---|---|---|
| `--agent` | `noop random` | `noop`, `random`, `rule-based`, or a path to an SB3 `.zip`. Repeatable |
| `--seeds` | `100` | how many episodes per agent |
| `--seed0` | `900_000` | first seed — held away from the training callback's `100_000` block |
| `--hold-steps` | `5` | must match what the checkpoint was trained on |

The table is ranked **failures → congestion → `exit_miss` → clearances →
touched**, the same lexicographic order the competition uses, and it prints both
scoring modes for a trained checkpoint.

!!! danger "Congestion is not comparable across rows with different failure counts"
    An episode that ends in a mid-air accumulates fewer conflict-steps **purely by
    existing for less time**. Read the congestion column against the `failed`
    column, never alone — otherwise crashing out looks like a safety strategy,
    and a policy will find that out before you do.

!!! warning "Score both modes on anything trained with an idle bias"
    `--idle-bias` makes idling the argmax at initialisation, so a *deterministic*
    rollout of a weak policy is byte-identical to doing nothing. Three runs in
    this project were written off as collapsed on exactly that evidence and all
    three were still learning. `score_4d.py` reports `[det]` and `[sto]` rows for
    every checkpoint for this reason.

!!! note "Where the reference numbers live"
    The current reference board — do-nothing, random-legal, rule-based and the
    trained policy over 100 seeds — is `slides/data/air_4d_board.csv`, and it is
    regenerated whenever the training reference run is. It is deliberately **not**
    transcribed onto this page: a number copied into prose is a number that goes
    stale silently. Run `scripts/score_4d.py` and read your own table.

### Why 100 seeds and not 20

The training callback runs 20 episodes so it can run often. Twenty episodes could
not rank two agents in this project: on a 20-episode callback one run led its
control on both failures and congestion, and on 50 frozen seeds **the ranking
inverted**. The callback exists to tell you a run is alive, not which run is
better.

!!! tip "A cautionary tale from the baseline itself"
    The first version of this controller reached for the two-level macro, which
    is one clearance instead of two and reverts for free. It lost to a random
    legal policy. The second version leaked slots at the altitude rails and lost
    8 of its 10 remaining failures to pairs sitting exactly where they had been
    told to sit. If a rule-based controller can fall into these traps twice, so
    can your reward function.

## 4. Exercise: beat the baseline

- **With RL:** train with `scripts/train_4d.py` (see
  [training](training.md)) and put your checkpoint on the table above.
  Beating do-nothing is not the exercise — do-nothing fails most of its episodes.
  Beating the controller is.

    !!! tip "Two ways to lose, and the diagnosis is different"
        **1. The policy idles.** `clearances` near zero and congestion equal to
        the do-nothing row. The entropy is gone; try `--auto-entropy`, and check
        the `[sto]` row before concluding anything.

        **2. The policy thrashes.** `clearances` in the thousands and episodes
        truncating early on the budget. It is buying safety with clearances it
        cannot afford, and `exit_miss` will be worse than random's. That is a
        weighting problem — run `scripts/calibrate_4d.py` before touching the
        hyperparameters.

- **With code:** the controller ignores speed clearances entirely, uses one global
  recovery radius, never re-plans mid-episode and ignores the clearance budget.
  Each is a straightforward improvement, and `tests/test_rule_based_4d.py`
  defines what any controller must keep true.

## Related

- [How the controller works](../reference/rule-based-controller.md) — the
  algorithm, line by line.
- [`Flight4DEnv` API reference](../reference/environment-api.md) — the
  observation you are decoding and the mask you are obeying.
- [Training](training.md) · [Evaluation](evaluation.md)
- [The Flight Challenge](competition.md) — the ranking this table
  imitates.
