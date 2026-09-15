---
title: The rule-based controller
---

# `Priority4DController` — how the bar is set

This is the agent every submission is measured against. It is a couple of hundred
lines of ordinary Python, it sees **exactly** the observation a neural policy
sees, and it exists so that "my agent beats do-nothing and a random legal policy"
is not the strongest claim anyone can make. That is a very low bar — a random
legal policy crashes out of a third of its episodes.

This page documents what the controller does and why each decision is there. For
*how to run it*, see [the litmus test](../advanced/baseline.md). For the
environment it operates on, see the
[`Flight4DEnv` API reference](environment-api.md).

!!! info "Where it lives"
    `sessions/03-advanced/airtraffic/agents/rule_based_4d.py` —
    `Priority4DController`, plus `Noop4DController` (the brick) and the `decode()`
    helper. Both are re-exported from `agents`.

```python
Priority4DController(n_flights=5, max_steps=50, hold_steps=5, severity_gate=0.05)
```

| Argument | Meaning |
|---|---|
| `n_flights`, `max_steps`, `hold_steps` | **must match the environment** — see the warning below |
| `severity_gate` | ignore predicted conflicts weaker than this |

It exposes SB3's `predict(obs, deterministic=...)`, so the same scoring loop runs
it, a PPO checkpoint and a competition submission interchangeably.

!!! warning "It has to be told the sector's shape"
    None of `n_flights`, `max_steps` or `hold_steps` is recoverable from the
    observation. Its length pins down `n_flights` only once you already know
    `N_FEATS`, and the horizon is needed to invert `t_conflict`. Get them wrong
    and every quantity the controller computes is off by a constant factor,
    **silently**, because the geometry still looks plausible. The competition
    harness passes the frozen scenario in explicitly for exactly this reason.

## What it is allowed to know

Nothing privileged. No peeking at the simulator, no conflict list handed to it
from the side. It receives the same flat observation vector as a neural policy
and inverts the normalisation itself:

```python
d = decode(obs, n_flights, max_steps, hold_steps)
# -> level, severity, hold_left, t_conflict, d_alt, step, x, y  (physical units)
```

The environment squashes two of these through `tanh` and one through
`t/(t + horizon)`. `decode()` inverts all three exactly rather than approximating
them — a controller that guesses at its own inputs is not a baseline, it is a
second source of error. `t_conflict` gets particular care: `q == 1` means *no
conflict predicted*, which is a genuine infinity rather than a very large number,
and is inverted explicitly instead of dividing by an epsilon and hoping.

The legality mask comes from the observation too:

```python
mask = obs[self.mask_offset:] > 0.5
```

Every action the controller proposes is checked against it, so it issues no
invalid clearances by construction.

## Why it does *not* use the two-level macro

This is the most interesting thing on the page, and it is worth reading before
you design your own action set.

### First, what the macro actually is

`Flight4DEnv` gives you two ways to change altitude, and the difference is the
whole argument:

| Clearance | What it does | Cost |
|---|---|---|
| `FL_INC` / `FL_DEC` | move **one** level, and **stay there** until told otherwise | 1 clearance |
| `FL_INC2_T` / `FL_DEC2_T` | move **two** levels, then **automatically revert** after `hold_steps` | 1 clearance — **the revert is free** |

The `_T` means *temporary*. That auto-revert is deliberate and it is not
charged: `_advance_holds()` treats it as "the second half of an instruction
already issued and already paid for."

### Why it looks like a free lunch

Three reasons, and all three are true:

1. **One level is not a fix.** `DALT` is 12 against an `ALT_MIN_SEP` of 24, so a
   single `FL_INC` leaves the pair still in conflict. Two levels is the
   *minimum effective* vertical intervention — the environment's own module
   docstring says exactly that.
2. **It is a third of the price.** By hand the same manoeuvre is `FL_INC` +
   `FL_INC` + `RESUME` = **three** clearances. The macro is **one**. Clearances
   are both a reward term and a hard episode-ending budget.
3. **It serves the 4D objective for free.** Coming home is otherwise a separate
   decision you have to remember to take; the macro bakes it in.

So it looks strictly dominant. It is not, and the reason is worth more than the
result.

### Why it loses anyway

**The return time is fixed the moment you issue it, and can never be
renegotiated.**

The revert fires inside `env.step()`, *before* the next observation. The
aircraft is already back at its original level by the time the controller can
see anything or act. You committed, at issue time, to a return time you had no
way of knowing was right.

And here it is systematically wrong. All the converging traffic enters
**co-level**, heading for one hotspot — that is *sustained* proximity, and it
outlasts `hold_steps`. The macro is built for a *transient* encounter: something
passes, you step aside, you come back. In this geometry, "come back on a timer"
means "come back into the traffic."

Measured, it is **worse than a random legal policy**: 20 failures in 40 episodes
against random's 34 in 100 — **50% against 34%**. The diagnosis over those 20:

| Failure mode | Count |
|---|---:|
| displaced, reverted, **hit** | 12 |
| never commanded at all | 8 |
| hit **while still displaced** | **0** |

Read those three rows together. Zero failures happened while an aircraft was
displaced, so the macro's *outbound* half works perfectly — the return leg is
the entire problem. And the 8 that were never commanded are the budget in
action: only **one aircraft may be commanded per step**, so clearances spent on
macros that will need re-issuing are clearances some other aircraft never gets.

Nor can you fix it by aiming better. Sweeping the **just-in-time lead** — how
many steps before the predicted conflict the macro is issued, so its hold window
straddles closest approach — from 1 to 6 steps never got below 50% failures.
That is what rules out "you just used it wrong."

!!! abstract "The general lesson"
    An auto-reverting macro bundles **two** decisions — *go* and *come home* —
    into one instruction, and prices them as one clearance. That is a bargain
    when the right return time is knowable at issue time, and a trap when it
    depends on how the episode develops.

    So this controller uses permanent level changes for lasting separation and
    an explicit `RESUME` once the aircraft is genuinely clear. It pays three
    clearances instead of one to keep the *when* in its own hands. The recovery
    gate is the whole point.

!!! tip "This is not an argument against the macro"
    A learned policy that can predict the right return time from the geometry
    can use it well — and it is the cheapest resolving action in the game. The
    argument is only that a **fixed-timer heuristic** cannot decide *when*
    adaptively. If your agent can, the macro is yours to exploit.

## The strategy

### 1. Assign slots, once, at reset

All converging traffic enters **co-level**. With `ALT_MIN_SEP` two levels wide,
the conflict-free slots are base, ±2, ±4 (and ±6 if the band allows). Earliest
predicted arrival keeps its level — it "has priority" — and later arrivals move
progressively further.

Converging traffic is identified by *having a predicted conflict at all*:
background flights already cross at other levels and never appear in the list.
That makes the controller indifferent to `n`, which varies per episode and is not
in the observation.

!!! danger "The bug worth knowing about: clipping collapses slots"
    The obvious implementation — `base[i] + offset[k]`, clipped into the band —
    is wrong near a rail. Clipping collapses +2 and +4 onto the same level and
    quietly assigns two aircraft the **same slot**. Measured, that was 8 of 10
    remaining failures, every one of them a pair sitting exactly where it had
    been told to sit.

    The fix is to build the slot list **once**, from the shared entry level,
    keeping only offsets that actually fit, and hand out distinct entries.

### 2. Steer, one level per step

The environment commands at most one aircraft per step, so the controller picks
the most urgent — soonest predicted conflict first — and moves it one level
toward its slot with `FL_INC` / `FL_DEC`. If that action is masked, it tries the
next aircraft.

### 3. Come home when genuinely clear

Once an off-plan aircraft is at least `RECOVERY_SEP_FACTOR` hotspot radii from
**every other converging flight**, a single `RESUME` restores its level and speed.

Distance from the other flights, not from the hotspot: they all share an entry
level, so it is separation from *them* that decides whether coming home is safe.
Keying on the hotspot over-waits the pairs that diverge quickly.

The slot is then retired — otherwise `_steer` sees a gap the instant the aircraft
is home and flies it straight back out, the controller undoing its own recovery
one clearance at a time. `d_alt` gives the deviation, so the entry level is
recoverable from the observation without remembering it.

**When several aircraft are clear at once, only one `RESUME` can go out this
step** — so something has to break the tie. It is assignment order, not flight
index: the same earliest-`t_conflict`-first ordering computed once at reset,
stored, and re-used here. Flight index carries no information about the
problem; the priority the controller already committed to does.

### `RECOVERY_SEP_FACTOR` is the tuning knob

It is a straight **safety-against-timeliness trade**, and the docstring in
`rule_based_4d.py` carries the measured sweep over 40 seeds.

![Trade-off curve of failed episodes against exit_miss across recovery radii,
with 7.0 marked and the random-legal reference lines](../assets/figures/fig_4d_recovery_sweep.png)

Read it down-and-left: both axes are costs, so the frontier bends away from the
origin and every setting buys one column by spending another. The dashed lines
are random-legal. **7.0 is the last point that sits inside both of them** —
push further out and `exit_miss` crosses to the wrong side, which is the moment
the 4D objective stops being served at all. recovering early
buys a better `exit_miss` and costs failures; recovering late (or never) buys
safety and parks aircraft off-level. The shipped value, `7.0`, is the last row of
that sweep that beats a random legal policy on **every** column — an `exit_miss`
worse than random's would mean the 4D objective is no longer being served at all.

!!! tip "This is the trade your agent has to win"
    A learned policy is not restricted to one global recovery radius. It can
    decide *per aircraft*, using the conflict geometry it can see, and it can use
    the speed pulse — which the controller ignores entirely. That is where the
    headroom is.

## `Noop4DController`

Do nothing, ever. The floor every other number is read against, and a genuinely
useful one here: the brick is *perfectly* on time and *perfectly* efficient, and
still loses badly on safety.

## What it deliberately does not do

Four open weaknesses, each a straightforward improvement and each a reasonable
place to start an agent design:

- **It never issues a speed clearance.** The entire velocity axis — and with it
  the schedule-neutral pulse — is unused.
- **One global recovery radius** for every aircraft in every geometry.
- **Slots are assigned once, at reset**, and never re-planned as the episode
  develops.
- **It ignores the clearance budget.** It does not ration, and it does not know
  how much of the episode is left.

`tests/test_rule_based_4d.py` defines what any controller must keep true — valid
actions only, and a real improvement over doing nothing.

## Related

- [Run the litmus test](../advanced/baseline.md) — the commands, and what the
  columns mean.
- [`Flight4DEnv` API reference](environment-api.md) — the observation it decodes
  and the mask it obeys.
- [The Flight Challenge](../advanced/competition.md) — how the same KPIs become a
  ranking.
