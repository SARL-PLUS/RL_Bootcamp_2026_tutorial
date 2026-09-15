---
title: Reward design
---

# Reward design

Everything on this page is **measured**, not asserted. The numbers come from
`Flight4DEnv` at its defaults — `n_flights=5`, `n ~ U(2,4)`, `max_steps=50`,
`hold_steps=5` — discounted at `gamma=0.99` and averaged over 40 seeds.
Reproduce any of them with:

```bash
cd sessions/03-advanced/airtraffic
python scripts/calibrate_4d.py --seeds 40 \
    --agent noop --agent random --agent rule-based
```

!!! abstract "The one-paragraph version"
    The reward is **three direct per-step costs** — predicted separation
    severity, 4D exit deviation, and a flat charge per clearance — plus a
    terminal mid-air charge sized by the steps the episode will now never fly.
    That is the whole thing. It is dominated by safety for every policy anyone
    has run on it, which is the intended ordering: safety is the objective that
    responds to control, and the other two exist to stop you buying it too
    cheaply.

---

## The reward, in full

```python
reward = -( w_safe * severity                              # safety
          + (w_safe / n_flights) * sum(deviation)          # timeliness
          + w_clearance * issued )                         # action
if terminated:
    reward -= collision_rate * w_safe * (max_steps - step_idx)
```

| Term | Form | Purpose |
|---|---|---|
| **safety** | `w_safe x` total predicted infringement severity | The objective. Ranks first, carries the largest weight. |
| **timeliness** | `w_safe / n_flights` per aircraft, each deviation in `[0, 1]` | The 4D exit gate. Closes the "solve it by wandering off" hole. |
| **action** | `w_clearance`, once, for any non-NOOP clearance | Commands are not free. A *transition* cost — a property of the action, not the state. |
| **mid-air** | `collision_rate x w_safe` per remaining step, terminal | Two co-level aircraft within 100 m. Scaled by steps unflown, so an early crash is not cheaper than a late one. |

There is no survival bonus and no terminal bonus of any kind. `max_clearances`
is a truncation condition and an observation feature, never a payout.

### Severity is not a hand-made proximity function

`severity = exp(-d_cpa / R) * exp(-d_alt / ALT_MIN_SEP)`, taken straight from the
simulator's own conflict definition, and pairs already at or beyond
`ALT_MIN_SEP` are not in the list at all.

The consequence is worth knowing before you touch anything: **the vertical
credit falls out of the physics rather than being chosen.** One flight level
removes `1 - exp(-0.5) = 39%` of the severity; the second removes the remaining
**61%** by deleting the conflict outright. An earlier version used
`(1 - d_alt / ALT_MIN_SEP)`, which paid a flat 50% for a first level that
resolves nothing — and the trained policy duly bought exactly that, on every
aircraft, at step 1, and stopped.

![Severity against vertical separation, showing the 39% credit for the first
flight level and the deletion of the conflict at ALT_MIN_SEP](../assets/figures/fig_4d_severity_credit.png)

The drop at `ALT_MIN_SEP` is a **deletion, not a decay** — the pair leaves the
conflict list. That is why one clearance is never a fix.

### Timeliness charges the *projected endpoint*, not the detour

This one is subtle and it is the fix for a real defect in the first-generation
environment, where the timeliness term returned the deviation **right now** and
was charged every step — so the reward integrated the whole detour. Over 100
seeds the reward penalised 9.963 while the KPI scored 0.039: the reward was
saying *never leave your plan* while the ranking said *leave it freely, but come
back*. Every policy duly learned not to deviate rather than to recover.

`Flight4DEnv` does not fix this by moving the charge to the horizon. It fixes it
by changing the quantity: `_exit_deviation()` returns **where the aircraft will
cross, if it holds its current speed** — a projection, not an instantaneous
displacement. A detour that is subsequently recovered projects back to its
target, so it costs nothing in the limit; only a deviation the aircraft cannot
recover from stays on the bill.

!!! note "The two regimes, and why conflating them was a bug"
    Before an aircraft reaches the boundary its deviation is a **projection**.
    After it has crossed, the deviation is a **fact** — the recorded crossing
    time — and must stop moving. Clamping `s_left` at zero and projecting anyway
    pinned the estimate to "now", so every aircraft that had already exited
    drifted later and later, and an untouched, undisturbed aircraft accumulated
    a phantom delay it had no way to cause. Pinned by
    `test_untouched_aircraft_have_zero_exit_deviation`.

### The scaling *is* the design statement

The absolute weights barely matter; the ratios are the argument:

| Quantity | Cost |
|---|---:|
| one severity-1 infringement, per step | `w_safe` = **3.00** |
| the **whole fleet** at maximum 4D deviation, per step | **3.00** (equal, by design) |
| one clearance | `w_clearance` = **0.15** = `w_safe / 20` |

So a single pair in full conflict outweighs every aircraft in the sector being
as late and as far off-level as it is possible to be. That makes the priority
explicit and checkable rather than a matter of taste.

![The three reward weights: safety and timeliness equal at 3.00, one clearance
at 0.15](../assets/figures/fig_4d_design_ratios.png)

---

## Where the return actually comes from

Discounted contribution of each term over a whole episode, 40 seeds. **Costs are
shown positive** — each is subtracted from the return, so smaller is better. The
mid-air charge is excluded; the `failed` column is where it lands.

| agent | safety | timeliness | action | total | failed | clearances | `exit_miss` |
|---|---:|---:|---:|---:|---:|---:|---:|
| do-nothing | 127.51 | 0.00 | 0.00 | **127.51** | 40/40 | 0.0 | 0.000 |
| random-legal | 74.99 | 5.51 | 2.84 | **83.34** | 22/40 | 22.7 | 0.590 |
| rule-based | 36.80 | 9.35 | 1.11 | **47.25** | 7/40 | 7.9 | 0.595 |
| PPO, 4M curriculum | 32.83 | 7.81 | 2.88 | **43.52** | 3/40 | 22.3 | 0.164 |

!!! note "These are seeds 0-39"
    `calibrate_4d.py` measures on the low seed block. `score_4d.py` ranks on the
    held-out block from 900000, where do-nothing fails 96 of 100 rather than 40
    of 40. Both are consistent; do not compare a number from one against a
    number from the other.

![Stacked decomposition of each policy's discounted episode cost into safety,
timeliness and action terms](../assets/figures/fig_4d_reward_decomposition.png)

Four things fall out of this table.

**The brick is the worst policy on the board, by a factor of nearly three.** It
is *perfectly* on time and *perfectly* efficient — its timeliness and action
bills are exactly zero, by construction — and it still loses, because it dies in
every episode and severity accumulates until it does. This is the single most
useful row here: it demonstrates that two of the three objectives can be
maxed by doing nothing at all, so any claim to have "optimised the reward" has
to say *which term*.

**Safety dominates every policy's bill**: 90% of random's total, 78% of the
controller's, 75% of PPO's. That is deliberate. It is the term that responds to
control, and the ranking puts separation first.

**The controller buys safety with lateness.** It pays *more* timeliness than
random (9.35 against 5.51) for less than half the safety cost, and its
`exit_miss` (0.595) is no better than random's (0.590). That is not a defect —
it is the `RECOVERY_SEP_FACTOR` trade, made deliberately and documented on the
[controller page](rule-based-controller.md). It is also exactly the headroom a
learned policy is supposed to take.

**PPO wins both objectives and pays for it in clearances**: better safety *and*
3.6x better `exit_miss` than the controller, for 2.8x the aircraft-commands
(22.3 against 7.9). Whether that is a good trade is the leaderboard's question,
not the reward's.

---

## Why these are direct costs and not potentials

This is the design decision most worth understanding before you change anything.

The reward could have been written as **potential-based shaping**,
`F(s,s') = gamma * phi(s') - phi(s)`, which
[Ng, Harada & Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/ml99-shaping.pdf)
proved leaves the optimal policy unchanged. It is not.

The reason is that the usual motivation for shaping does not apply here.
**Predicted severity is already dense**: a conflict twenty steps out is in
today's sum, because `_severity()` sums over `predict_conflicts(horizon)` rather
than over conflicts happening right now. There is no sparse-reward problem to
solve, so there is nothing for shaping to buy.

What it would cost is clarity. A direct cost says what we actually mean —
*this state is bad, by this much, now* — and the weights above are then a
readable statement of priority.

!!! warning "The price of that choice, stated plainly"
    A direct cost **can** move the optimum. Potential-based shaping provably
    cannot. So the weights on this page are not merely a statement about how
    fast the problem is learned; they are part of what "solved" means. If you
    retune `w_safe` or `w_clearance`, you are changing the answer, not just the
    search — and you should re-check the ranking, not just the return.

---

## Adding shaping yourself

Shaping is not used, but it is a legitimate thing to add — it is **Track C** of
the design sprint. If you do, this is what the environment already knows about
it.

Two potentials fit the problem naturally:

- **separation** — `phi = -` (summed pairwise proximity over co-altitude traffic)
- **timeliness** — `psi = -` (exit deviation)

Because they cannot move the optimum, the shaping *scale* is the safest
parameter in the reward to turn up. Measured on the first clearance out of a
do-nothing policy, on the retired first-generation environment:

| shaping scale | mean Δ return | fraction that help | 90th pct |
|---:|---:|---:|---:|
| 5 | −0.953 | 9.5% | −0.11 |
| 15 | −0.806 | 21.8% | +0.98 |
| **30** | **−0.633** | **26.1%** | **+2.19** |
| 60 | −0.285 | 30.9% | +5.01 |

The mean stays negative throughout, and it should — a randomly chosen clearance
*is* a bad idea. What changes is the **top decile**: at the lowest setting even
the best 10% of clearances were neutral-to-negative, so there was nothing for
the gradient to climb.

!!! danger "Both factors of the potential must be smooth"
    A potential that merely asks *"are these two within `ALT_MIN_SEP`?"* is a
    **step function in altitude**. `ALT_MIN_SEP` is 24 and one clearance moves
    an aircraft by `DALT = 12`, so the first `FL_INC` takes a pair from 0 to 12
    — still inside the threshold, still counted, shaping reward exactly
    **zero**. The agent only gets paid on the second climb, which is precisely
    the step it had no reason to reach.

    Both constants are unchanged in `Flight4DEnv`, so this trap is still live.

---

## Three results that survive the environment they were measured on

These come from a 276k-injection probe on the retired first-generation
environment. The *arguments* are algebraic or structural and carry over; the
numbers are historical.

**Scale cannot create gradient where a term is exactly zero.** Raising the
congestion weight until one event outweighs the worst timeliness bill moved the
help rate from 3.0% to 3.0% and **doubled the variance**. Most single clearances
change no congestion event at all in the step they are issued.

**Decomposing a scalar reward per aircraft is a no-op.**
`Σᵢ ½ Σ_{pairs ∋ i} c = Σ_pairs c` — same scalar, same gradient. It buys
something only if you stop re-adding (per-head advantages in a custom loss) or
change the functional form. The `max`-per-aircraft form measured *worse* than
the pairwise sum, because `max` discards every improvement except to an
aircraft's single worst conflict.

**Signal and reachability are different problems.** A dense potential lifted the
help rate from 3% to 21% by saying something useful on the ~90% of injections
where the KPI did not move at all. The two-level macro lifted *every* reward
variant, because it made a resolving move reachable in one clearance instead of
two. **The potential supplies signal; the macro supplies reachability.** Neither
substitutes for the other — and note that the macro's usefulness here is about
the *outbound* half only; see
[why the controller does not use it](rule-based-controller.md).

---

## The mid-air charge has to beat the cost of flying on

Every term is a cost, so **ending the episode early stops the meter**. If the
terminal charge is smaller than the cost of continuing, the optimal policy is to
crash immediately.

`calibrate_4d.py` checks this on every run:

```text
mid-air charge at step 0                  450.00
... per remaining step                      9.00
WORST cost of flying on, over all states  170.55
```

The mean is not the test. The distribution is skewed and the agent only has to
find **one** state where dying is cheaper, so the check is against the worst
case over all visited states — currently 450 against 170.55, a comfortable
margin. Re-run it after any weight change.

---

## Why the reward is not the whole story

Two failures on this environment look like reward problems and are not.

**The episode can end before most of the conflicts do.** A fresh near-uniform
policy exhausts its clearance budget long before the horizon, experiences a
fraction of the conflicts, and never reaches the states where the interesting
decisions live. Shaping changes the value of states you visit; it can do nothing
about states you never reach. See
[`--noop-bias`](../advanced/training.md).

**A zeroed term is an unconstrained dimension.** Annealing the clearance cost to
exactly zero makes the return indifferent to how many clearances are issued —
and an entropy bonus will expand into any dimension the return does not push
back on. Measured: entropy 4.43 → 9.34 nats, episode length 50 → 31.9, straight
back into the truncation failure. If you want that curriculum, anneal from a
small floor rather than from zero.

---

## Related

- [Environment API](environment-api.md#reward) — the exact terms and constants.
- [The rule-based controller](rule-based-controller.md) — what the measured
  baseline row above is actually doing.
- [The Flight Challenge](../advanced/competition.md) — what you are ranked on,
  which is not the return.
- [Training](../advanced/training.md) — flags, and the flat-curve
  checklist.
