---
title: Environment API
---

# `Flight4DEnv` — API Reference

Exact spaces, reward maths and public surface of the Session #3 environment and
the simulator underneath it.

!!! info "Source of truth"
    `sessions/03-advanced/airtraffic/envs/flight_4d.py`. Everything on this page
    is read off that file. When the two disagree, the code is right and this page
    is the bug.

`Flight4DEnv` is not a subclass of anything. The first-generation environment
grew by accretion and its reward became hard to reason about end to end;
`Flight4DEnv` was written fresh, and the only thing carried across is the
**kinematics** (`CollisionCourseSimulator`), so scenarios stay comparable.

---

## `Flight4DEnv`

```python
Flight4DEnv(n_flights=5, n_range=(2, 4), max_steps=50, hold_steps=5,
            w_safe=3.0, w_clearance=0.15, collision_rate=3.0,
            gamma=0.99, max_clearances=40, seed=None)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `n_flights` | int | `5` | the whole sector, `n + m`. Fixes the observation and action shapes |
| `n_range` | `(lo, hi)` | `(2, 4)` | `n`, the converging traffic, is drawn uniformly from this **each reset**; the rest is background |
| `max_steps` | int | `50` | horizon, in steps of `DT = 2 s` — so 100 s end to end |
| `hold_steps` | int | `5` | how long a temporary manoeuvre holds before it reverts |
| `w_safe` | float | `3.0` | weight on predicted infringement severity |
| `w_clearance` | float | `0.15` | flat cost of issuing one clearance |
| `collision_rate` | float | `3.0` | terminal mid-air charge, **per remaining step**, in units of `w_safe` |
| `gamma` | float | `0.99` | recorded on the env so the reward scaling can be reasoned about; PPO is told the same value |
| `max_clearances` | int | `40` | clearance budget; spending it truncates the episode |
| `seed` | int / None | `None` | stored; the seed that actually matters is the one passed to `reset()` |

!!! note "`n + m = n_flights`, always"
    The sector size never changes, only the **split**. That is deliberate:
    difficulty can vary episode to episode (and across a curriculum) while the
    observation and action spaces cannot, so one checkpoint loads against every
    density. `set_n_range(lo, hi)` narrows or widens the draw and takes effect on
    the next `reset()`; it refuses `lo > hi` or `hi > n_flights`.

### Derived attributes

| Attribute | Value at defaults | Meaning |
|---|---|---|
| `n_actions` | `36` | `1 + 7 · n_flights` |
| `mask_offset` | `67` | where the legality mask starts inside the observation |
| `n`, `m` | drawn per episode | converging / background aircraft, `n + m = n_flights` |
| `DT` | `2` | seconds per step |
| `DALT` | `12` | one flight level |
| `ALT_MIN_SEP` | `24` | vertical separation minimum — **two** flight levels |
| `DVEL` | `35` | one speed increment, m/s |
| `vel_min`, `vel_max` | `200`, `515` | speed rails, m/s |
| `AIRSPACE` | `10000` | half-width of the monitored box, m |
| `D_SEP` | `1000.0` | horizontal loss-of-separation radius, m |
| `COLLISION_RADIUS` | `100.0` | mid-air radius, m |

Altitudes live on `N_ALTS = 10` discrete levels, **290–398**. With
`ALT_MIN_SEP = 24 = 2 · DALT`, *one* level of separation resolves nothing; two
does. That threshold shapes both the clearance vocabulary and the mask.

---

## Action space

```python
Discrete(1 + 7 * n_flights)      # 36 at the defaults
```

**Index 0 idles.** Index `1 + 7·i + c` issues clearance `c` to aircraft `i`.
**At most one aircraft is commanded per step** — there is no simultaneous
command vector, and this is closer to how a controller actually works: the
clearance budget bites, and every step is a choice of *who* as well as *what*.

| `c` | Name | Effect |
|---:|---|---|
| 0 | `FL_DEC` | descend one flight level — **permanent** |
| 1 | `FL_INC` | climb one flight level — **permanent** |
| 2 | `FL_DEC2_T` | descend **two** levels, hold `hold_steps`, then return to the entry level |
| 3 | `FL_INC2_T` | climb **two** levels, hold `hold_steps`, then return |
| 4 | `SPD_DN_T` | −`DVEL`, then +`DVEL`, then nominal — a **symmetric pulse** |
| 5 | `SPD_UP_T` | +`DVEL`, then −`DVEL`, then nominal |
| 6 | `RESUME` | restore the entry level *and* the entry speed in one instruction |

Names and the count are exported as `CLEARANCE_NAMES` and `N_CLEARANCES`.

!!! abstract "Why every disturbing clearance auto-reverts"
    A speed clearance shifts an aircraft's whole temporal trajectory, so meeting
    the 4D exit slot afterwards needs an exact compensating clearance later —
    which fights the "minimise clearances" objective. An agent asked to discover
    a matched `+dv … −dv` pair by independent sampling at each step is being
    asked for a `p²` coincidence, and it will not find one.

    So the recovery is **baked into the instruction**. `SPD_UP_T` is "+dv, hold,
    then come home" as *one* clearance, charged *once*: executing the second half
    is compliance with an instruction already issued, the same contract a
    standing "resume own navigation" has.

    The speed pulse is genuinely schedule-neutral. Reverting to the nominal
    *speed* would not restore the *schedule* — an aircraft that ran fast for
    `hold_steps` and then went back to plan is permanently early. The equal and
    opposite half pays that back. (The counter is `2·hold_steps + 1`, not
    `2·hold_steps`, because the tick happens inside the same `step()` that issues
    the clearance; `2·hold_steps` leaves exactly one `DVEL·DT` of along-track
    debt that nothing ever repays.)

!!! note "Why two levels, not one"
    `DALT` is 12 and `ALT_MIN_SEP` is 24, so a single altitude clearance moves a
    co-level pair from 0 to 12 and leaves it **in conflict**. The two-level macro
    is the minimum effective vertical intervention. The permanent single-level
    commands are still there because "put it there and leave it" is a different
    instruction from "step aside and come back" — which of the two a policy
    prefers is a result, not an assumption.

### `action_masks()`

```python
mask = env.action_masks()        # np.ndarray, dtype=bool, shape (n_actions,)
```

Index 0 is always `True`. Three families of illegality:

**Rails.** A climb at the ceiling, a descent at the floor, `+dv` at `v_max`,
`−dv` at `v_min`. The two-level macro needs **two** levels of headroom, not one:
booking a climb that stops halfway is exactly the half-manoeuvre it exists to
remove.

**Inertness.** `RESUME` on an aircraft already on plan, and *any* clearance to an
aircraft mid-manoeuvre (`hold_left > 0` or `pulse_left > 0`). Re-commanding a busy
aircraft is how a policy spends budget without changing anything.

**Kinematic dead ends.** A speed clearance is illegal if, after it, the remaining
distance could not be covered in the aircraft's remaining *scheduled* time even
at `v_max` — or could not be stretched to it even at `v_min`. Past that point the
4D slot is unreachable and no later clearance recovers it, so the action is
removed rather than punished.

!!! warning "The schedule is per aircraft, not the episode clock"
    The remaining time is measured against **that aircraft's own** exit time
    (`t_target[i] − t_now`), not against `max_steps`. Most aircraft leave the
    sector well before step 50, so using the episode horizon overstates the time
    they have left and makes every speed clearance look as though it would arrive
    early. Measured, that mistake made speed clearances legal on 13–16% of
    `(aircraft, step)` pairs — silently deleting the entire velocity axis, which
    is the one the 4D tension is about.

An action that is masked out and issued anyway is **not** charged a penalty: it
is a no-op and increments `info["invalids"]`. An invalid action is not something
to learn to avoid, it is something to remove from the action space; the counter
exists so a broken policy is loud rather than silent.

---

## Observation space

```python
Box(low=-1, high=1,
    shape=(N_FEATS * n_flights + N_GLOBALS + n_actions,), dtype=float32)
# 13·5 + 2 + 36 = 103 at the defaults
```

Every value is in `[-1, 1]`. Three blocks, in this order:

| Block | Size | Contents |
|---|---:|---|
| **aircraft** | `13 · n_flights` | one 13-tuple per aircraft, in `FEATURE_NAMES` order |
| **globals** | `2` | episode clock, clearance budget remaining |
| **mask** | `n_actions` | `action_masks()` as floats — see below |

### Per-aircraft features (`FEATURE_NAMES`)

| # | Name | Encoding |
|:--:|---|---|
| 0 | `x` | position / `AIRSPACE`, clipped |
| 1 | `y` | position / `AIRSPACE`, clipped |
| 2 | `alt` | normalised level, rescaled to `[-1, 1]` |
| 3 | `vel` | normalised speed, rescaled to `[-1, 1]` |
| 4 | `sin_hdg` | heading unit vector, x component |
| 5 | `cos_hdg` | heading unit vector, y component |
| 6 | `dt_exit` | `tanh(dt / (hold_steps·DT))` — projected exit time minus target exit time |
| 7 | `d_alt` | `tanh(d_alt / 2)` — vertical deviation from the entry level, in flight levels |
| 8 | `n_clearances` | this aircraft's clearance count, saturating at 6 |
| 9 | `hold_left` | steps until an active temporary manoeuvre reverts |
| 10 | `severity` | worst **predicted** conflict this aircraft is in, `[0, 1]` rescaled |
| 11 | `t_conflict` | squashed time until it; reads `+1` when none is predicted |
| 12 | `active` | `1` if this slot holds a real aircraft |

Then the two globals: the episode clock `step_idx / max_steps`, and
`1 − n_clearances / max_clearances`, both rescaled to `[-1, 1]`. Both are
observable because the environment **ends the episode on each of them** — an
agent asked to ration a resource it has no reading of is being set up to fail.

!!! abstract "One frame, not a stack"
    Headings are never commanded, so absent a clearance every aircraft's future
    is a straight line determined by its own `(x, y, alt, v, heading)`. Pairwise
    closing geometry is therefore *derivable* from a single frame, and the state
    is Markov once the pending-manoeuvre countdown (`hold_left`) is included.
    That is why there is no `k_frames`, no history ring and no recurrence here.

!!! abstract "`dt_exit` is in the observation because the reward is about it"
    The third objective is the 4D exit slot, and `dt_exit` is the quantity that
    objective is actually about. A cost the agent cannot see is a cost it cannot
    act on.

**Features 10–11 are forward-looking on purpose.** They come from
`sim.predict_conflicts()`, which solves closest approach in closed form, not from
"who is too close right now" — by the time the latter fires, the conflict has
happened. Severity is `exp(−d_cpa/R) · exp(−d_alt/ALT_MIN_SEP)`, and pairs
already at or beyond `ALT_MIN_SEP` are not in the list at all.

### The mask is inside the observation

The last `n_actions` values are `action_masks()` cast to float. This is not
redundancy, and it is not a convenience:

!!! danger "A mask passed by side channel silently corrupts PPO"
    SB3's rollout buffer stores **observations and nothing else**. A mask handed
    to the policy any other way is present when the action is *chosen* and absent
    when its log-probability is *recomputed at update time* — so PPO compares two
    different distributions and optimises a ratio nobody intended. Carrying the
    mask in the observation makes that impossible by construction.

    It also means the policy never has to re-derive physics from normalised
    features: the environment owns the legality rules and says so.

`env.mask_offset` is where the block starts. A controller reading the observation
recovers the mask with `obs[env.mask_offset:] > 0.5`.

---

## Reward {#reward}

Three terms, and nothing else. All are costs; the reward is their negation.

| Term | Form | Weight |
|---|---|---|
| **safety** | total predicted infringement severity, summed over conflicts | `w_safe` (3.0) |
| **4D adherence** | per-aircraft deviation in `[0, 1]`, summed over the fleet | `w_safe / n_flights` each |
| **action** | `1` if a legal clearance was issued this step, else `0` | `w_clearance` (0.15) |

```text
reward = −( w_safe · severity_total
          + (w_safe / n_flights) · Σ deviation_i
          + w_clearance · issued )
```

Plus one terminal charge on a mid-air:

```text
collision_charge = collision_rate · w_safe · (max_steps − step_idx)
```

!!! abstract "The scaling between the first two terms is the design statement"
    **The whole fleet at maximum 4D deviation costs exactly one severity-1
    infringement.** So safety strictly dominates whenever more than one pair is
    in trouble, and the priority is explicit and checkable rather than a matter
    of taste. One clearance costs `w_safe / 20`.

    An untouched aircraft has zero deviation by construction, so background
    traffic is free.

!!! note "Why the mid-air charge is per remaining step"
    A flat lump made a late mid-air disproportionately expensive and an early one
    disproportionately cheap. Charging the steps *not flown* completes every
    episode to the same horizon, so returns are comparable across episodes and
    the critic does not have to fit that spread on top of everything else. The
    rate is expressed in units of `w_safe`, so there is no separate constant to
    keep in sync.

!!! warning "These are direct costs, not potential-based shaping"
    Predicted severity is *already* dense — a conflict twenty steps out is in
    today's sum — so it does not need shaping to be learnable, and a direct cost
    says what we actually mean. The trade is that a direct cost **can** move the
    optimum, where potential-based shaping provably cannot. That is a deliberate
    choice, and it is the reason the weights are worth arguing about.
    `scripts/calibrate_4d.py` sizes the terms against each other before you spend
    a training run on them.

!!! danger "What a badly sized clearance cost does, measured"
    At `w_clearance = 0.35` against `w_safe = 30`, a fully-intruding pair was
    worth 30 in the safety term, paid immediately, against 0.35 for a clearance —
    **86:1 in favour of acting**. There was no brake: the trained policy went to
    26 clearances an episode and scored **worse than a random legal policy** on
    the 4D objective. Read `scripts/calibrate_4d.py` before changing a weight.

---

## Termination

| Condition | Flag | Trigger |
|---|---|---|
| **Mid-air** | `terminated` | two aircraft within `COLLISION_RADIUS` (100 m) and less than one flight level apart |
| Horizon reached | `truncated` | `step_idx >= max_steps` |
| Clearance budget spent | `truncated` | `n_clearances >= max_clearances` |

**A mid-air is the only genuine terminal state**: the thing the environment
exists to prevent has happened, and no amount of tidy flying afterwards would
undo it, so its value is zero. The other two are time limits — the sector ran out
of episode, or the controller ran out of allowance — and those states still have
a future worth bootstrapping from.

!!! warning "A mid-air is not the same as loss of separation"
    Congestion is counted at **1000 m horizontally and `ALT_MIN_SEP` (two flight
    levels) vertically**. That is a procedural violation: you are charged for it
    every step it persists, and you fly out of it. A mid-air is **100 m at less
    than one flight level** — the aeroplanes have met, and it ends the episode.

---

## The `info` dict

`step()` and `reset()` return the same keys every call, straight from `_info()`.
The [contract test suite](../advanced/competition.md#how-your-submission-is-scored)
asserts each of them, so a modified environment that drops one cannot be scored.

| Key | Type | Meaning |
|---|---|---|
| `n` | int | converging aircraft this episode |
| `m` | int | background aircraft this episode (`n + m = n_flights`) |
| `congestion_events` | int | loss-of-separation pairs **this step** — the safety KPI |
| `congestion_total` | int | the same, cumulative over the episode |
| `clearances` | int | cumulative legal clearances issued |
| `invalids` | int | cumulative masked-out actions that were issued anyway |
| `aircraft_involved` | int | distinct aircraft that received any clearance |
| `collided` | bool | a mid-air happened |
| `dt_exit` | ndarray `(n_flights,)` | per-aircraft temporal deviation, **seconds** |
| `d_alt_exit` | ndarray `(n_flights,)` | per-aircraft vertical deviation, **flight levels** |
| `exit_miss` | float | **the scored 4D quantity** — see below |

### `exit_miss`

```text
exit_miss = mean over aircraft of ( |dt_exit| / (hold_steps · DT) + |d_alt_exit| )
```

Seconds late plus levels off, per aircraft, averaged over the sector. The divisor
is what puts the two on a common scale: **one hold's worth of delay weighs the
same as one flight level off plan**. Lower is better and **zero is achievable** —
an aircraft you never touched is at zero.

!!! note "Two regimes, and conflating them was a real bug"
    *Before* an aircraft reaches the boundary its deviation is a **projection**:
    how late it will be if it holds its current speed. *After* it has crossed,
    the deviation is a **fact** — the recorded crossing time — and must stop
    moving. Clamping the remaining distance at zero and projecting anyway pinned
    the estimate to "now", so every aircraft that had already exited drifted later
    and later and an untouched aircraft accumulated a phantom delay it had no way
    to cause. `tests/test_flight_4d.py::test_untouched_aircraft_have_zero_exit_deviation`
    is the regression.

    Crossing times are interpolated *within* the step, not rounded to the step
    boundary: at 200–515 m/s one step is up to a kilometre, and rounding would
    put a quantisation floor under the very quantity the 4D objective scores.

---

## Key methods & attributes

| Member | Description |
|---|---|
| `reset(*, seed=None, options=None)` | Standard Gymnasium reset. Draws `n`, builds a fresh simulator, computes the 4D gate. Returns `(obs, info)`. |
| `step(action)` | Takes an `int` (or a 1-element array). Returns `(obs, reward, terminated, truncated, info)`. |
| `action_masks()` | Bool mask over the flat action space. Also carried in the observation from `mask_offset`. |
| `set_n_range(lo, hi)` | Narrow or widen the per-episode draw of `n`. Takes effect on the next `reset()`. |
| `collision_charge()` | The terminal mid-air cost at the current step. |
| `mask_offset` | Index where the mask block starts in the observation. |
| `n_actions` | `1 + N_CLEARANCES · n_flights`. |
| `n_clearances`, `n_invalids` | Episode counters. |
| `t_target`, `t_exit_actual` | Scheduled and recorded exit times, per aircraft (`NaN` until the aircraft leaves the box). |
| `hold_left`, `pulse_left` | Per-aircraft countdowns for the two kinds of temporary manoeuvre. |
| `sim` | The underlying `CollisionCourseSimulator`. |

Module-level constants worth importing: `N_CLEARANCES`, `CLEARANCE_NAMES`,
`FEATURE_NAMES`, `N_FEATS`, `N_GLOBALS`, and the clearance indices
`FL_DEC, FL_INC, FL_DEC2_T, FL_INC2_T, SPD_DN_T, SPD_UP_T, RESUME`.

Importing `envs` registers the id `Flight4DEnv-v0`.

---

## `CollisionCourseSimulator`

```python
CollisionCourseSimulator(n, m, airspace_limit=10000, init_flights=True, rng=None)
```

The physics engine, and the **only** thing shared with the first-generation
environment. Spawns `n` aircraft converging on a shared point plus `m` random
aircraft, advances their kinematics, and detects congestion.

| Member | Description |
|---|---|
| `flights` | list of `Flight` objects in the airspace. |
| `advance_into_airspace(max_shift=200)` | see below. |
| `find_congestion(flights)` | who is too close **right now**. |
| `predict_conflicts(horizon, flights=None)` | `Conflict` records for pairs that **will** breach separation within `horizon` if nothing changes. |
| `advance_clock()` | step every flight one tick of `DT`. |
| `norm_alt` / `norm_vel` / `denorm_*` | the scalings the observation uses. |
| `animate(save_path)` | renders the whole airspace evolving to an `.mp4`. |

### `advance_into_airspace` {#advance-into-airspace}

Scenario construction rewinds every flight until it is *outside* the airspace,
which guarantees each one flies across the sector rather than starting midway
through it. Left there, the episode opens on an empty screen.
`advance_into_airspace()` time-shifts the whole scenario forward until every
flight is inside the box. The shift is **uniform** — every flight advances
together — so relative geometry, closing speeds and arrival order are untouched;
only the origin of the clock moves.

### `predict_conflicts`

Returns one `Conflict` per predicted conflicting pair, carrying `t_enter`,
`t_cpa`, `duration`, `d_cpa`, `d_alt`, `center`, `radius` and `severity`. Absent
a clearance every track is a straight line, so `|r₀ + Δv·t| = hotspot_limit`
solves in closed form per pair: the roots give the conflict window and the vertex
gives closest approach. No rollout is needed. Pairs already separated by more
than `ALT_MIN_SEP` are skipped outright, since vertical separation is constant
without a clearance.

!!! abstract "Severity is the simulator's, not a hand-made proximity function"
    `severity = exp(−d_cpa/R) · exp(−d_alt/ALT_MIN_SEP)`, so the vertical credit
    falls out of the physics instead of being chosen: **one** flight level removes
    `1 − exp(−0.5) = 39%` of the severity, and the **second** removes the
    remaining 61% by deleting the conflict outright. An earlier version used
    `(1 − d_alt/ALT_MIN_SEP)`, which paid a flat 50% for a first level that
    resolves nothing — and the trained policy duly bought exactly that, on every
    aircraft, at step 1, and stopped.

---

## `Flight`

Single-aircraft kinematics (stdlib only). Holds position, velocity, heading and
altitude; `advance_clock(dt)` steps it one tick, `change_altitude(delta)` and
`change_velocity(delta)` are how clearances land. `get_heading_unit_vectors()`
returns `(sin, cos)` of the heading — features 4–5 of the observation.
`get_5_tuple()` returns `(x, y, alt, vel, hdg)`; `get_5_tuple_dict()` returns the
same keyed by `Flight.TUPLE5KEYS`.

---

## Related

- [The AirTraffic environment](../advanced/airtraffic.md) — what it is, and
  how to step it by hand.
- [The rule-based controller](rule-based-controller.md) — the bar, built on this
  API and nothing else.
- [The Flight Challenge](../advanced/competition.md) — how these KPIs become a
  ranking.
