"""Kaggle custom evaluation metric for the Flight Challenge. **Lower is better.**

Upload this file as the competition's custom metric. It is deliberately
self-contained: pandas, numpy and the standard library only — no gymnasium, no
matplotlib, no torch, and no import of the bootcamp package, none of which exist
in the metric sandbox.

How it works
------------
The participant submits the action taken at every (seed, step): one integer, the
environment's flat clearance index. The solution file carries the **initial state
of every aircraft** for every scored seed. This metric re-flies each episode from
that initial state, applying the submitted actions, and computes the KPIs itself.
There is no number in the submission to inflate: a trace that was not produced by
a good policy does not replay like one.

Shipping the scenarios in the solution file — rather than regenerating them from
the seed — is what keeps this file honest. Reproducing the scenario generator
here would mean a second implementation of it, free to drift from the real one
between now and September. The initial states are exported from the real
environment, and ``test_kaggle_metric.py`` asserts this replay agrees with it
exactly on random action traces.

What is necessarily mirrored here
---------------------------------
Everything the environment does *after* reset, and nothing else:

* the legality mask — rails, busy aircraft, and the kinematic dead ends;
* clearance application, including the ``hold_steps`` auto-revert of the
  two-level altitude macro and the symmetric speed pulse;
* straight-line motion, congestion counting and mid-air detection;
* the 4D exit gate: projected crossing time against the scheduled one, plus the
  vertical deviation.

Deliberately absent: the reward, the observation, and conflict *prediction*. The
score is built from the info dict, so none of the three can move a leaderboard
position and re-deriving them would be three more chances to drift.

Ranking
-------
The competition ranks lexicographically on
``(failed, congestion, exit_miss, clearances)``. Kaggle sorts one float, so the
four are packed positionally into a single number: each component is clamped to a
documented bound and weighted by the product of all lower-priority ranges, so a
difference in a higher-priority objective always dominates any possible
difference below it. The packed value stays under 2**53 and is therefore exact in
float64.

The full competition key has a fifth component — distinct aircraft touched —
which does **not** fit the exact-packing budget. It breaks ties only on the
private leaderboard. Two submissions identical on all four public components tie
here; they will not tie at the ceremony.
"""
import math
import re

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    """Message shown to the participant on a bad submission."""


# --- Frozen physics, mirroring CollisionCourseSimulator -------------------
DT = 2
DALT = 12
ALT_MIN = 290
N_ALTS = 10
ALT_MIN_SEP = DALT * 2
DVEL = 35          # keep in sync with CollisionCourseSimulator.DVEL
VEL_MIN = 200
N_VELS = 10
AIRSPACE_LIMIT = 10000
#: Loss of separation: horizontal radius of the hotspot, 1000 m.
D_SEP = AIRSPACE_LIMIT * 0.1

ALT_MAX = ALT_MIN + (N_ALTS - 1) * DALT
VEL_MAX = VEL_MIN + (N_VELS - 1) * DVEL

# --- Frozen scenario, mirroring competition.config.ScenarioConfig ----------
MAX_STEPS = 50
HOLD_STEPS = 5
#: Truncates the episode. Flight4DEnv's own default; an episode that hits it is
#: short, and a short episode is scored as a failure.
MAX_CLEARANCES = 40
#: Two co-level aircraft this close have hit each other. A tenth of the hotspot
#: radius — loss of separation is 1000 m and is a violation you fly out of; this
#: is a mid-air, and it ENDS the episode. Keep in sync with
#: ``Flight4DEnv.COLLISION_RADIUS``.
COLLISION_RADIUS = 100.0

# --- The clearance vocabulary, mirroring envs.flight_4d -------------------
FL_DEC, FL_INC, FL_DEC2_T, FL_INC2_T, SPD_DN_T, SPD_UP_T, RESUME = range(7)
N_CLEARANCES = 7

# --- Ranking ------------------------------------------------------------
EXIT_MISS_BUCKET = 0.005

# Clamps, chosen from the worst case each quantity can actually reach, with
# headroom. Their product must stay under 2**53 for the packing to be exact.
BOUND_FAILED = 100          # episodes
BOUND_CONGESTION = 50_000   # <= pairs per step * steps * episodes
BOUND_EXIT_MISS = 4_000     # buckets; an exit miss of 20 gate-units, saturating
BOUND_CLEARANCES = 20_000   # the env truncates well before this


def _weights():
    w_clearances = 1
    w_exit_miss = w_clearances * (BOUND_CLEARANCES + 1)
    w_congestion = w_exit_miss * (BOUND_EXIT_MISS + 1)
    w_failed = w_congestion * (BOUND_CONGESTION + 1)
    span = w_failed * (BOUND_FAILED + 1)
    assert span < 2 ** 53, "packed score would lose precision in float64"
    return w_failed, w_congestion, w_exit_miss, w_clearances


def pack(failed: int, congestion: int, exit_miss_mean: float,
         clearances: int) -> float:
    """Collapse the lexicographic key into one exactly-representable float."""
    w_f, w_g, w_t, w_c = _weights()
    f = min(max(int(failed), 0), BOUND_FAILED)
    g = min(max(int(congestion), 0), BOUND_CONGESTION)
    t = min(max(int(round(exit_miss_mean / EXIT_MISS_BUCKET)), 0), BOUND_EXIT_MISS)
    c = min(max(int(clearances), 0), BOUND_CLEARANCES)
    return float(f * w_f + g * w_g + t * w_t + c * w_c)


def unpack(value: float) -> dict:
    """Invert :func:`pack` — read a raw leaderboard number back as KPIs.

    ``pack`` is a mixed-radix encoding: each component gets its own digit range,
    sized so it can never bleed into the one above it. That makes this an exact
    inverse, not an approximation — paste in whatever Kaggle shows and get the
    four components back:

    >>> pack(10, 176, 0.12875161496894288, 2089)
    40026885486301.0
    >>> unpack(40026885486301.0)
    {'failed': 10, 'congestion': 176, 'exit_miss_bucket': 26, 'clearances': 2089}

    ``exit_miss_bucket`` is ``exit_miss_mean`` rounded to the nearest
    ``EXIT_MISS_BUCKET`` (0.005) and clamped to ``BOUND_EXIT_MISS`` — multiply by
    ``EXIT_MISS_BUCKET`` for an approximate mean; the rounding only goes one way,
    so this cannot recover the exact value ``pack`` was given.

    A component reading exactly its ``BOUND_*`` constant may have been *clamped*
    there, not measured: ``pack`` saturates instead of overflowing into the digit
    above it, so an agent that failed 500 episodes and one that failed exactly
    100 (``BOUND_FAILED``) are indistinguishable from the packed score alone.
    """
    w_f, w_g, w_t, w_c = _weights()
    n = int(round(value))
    failed, n = divmod(n, w_f)
    congestion, n = divmod(n, w_g)
    exit_miss_bucket, n = divmod(n, w_t)
    clearances, _ = divmod(n, w_c)
    return {"failed": failed, "congestion": congestion,
            "exit_miss_bucket": exit_miss_bucket, "clearances": clearances}


# --- Replay -------------------------------------------------------------

def _distance_to_boundary(x: float, y: float, ux: float, uy: float) -> float:
    """How far this aircraft still has to fly before leaving the box.

    Straight line from its position along its heading, intersected with the
    airspace square. Headings are never commanded, so this distance is fixed for
    the whole episode and the 4D gate reduces to it plus the time it takes at
    the entry speed.
    """
    ts = []
    for pos, u in ((x, ux), (y, uy)):
        if abs(u) < 1e-12:
            continue
        for bound in (-AIRSPACE_LIMIT, AIRSPACE_LIMIT):
            t = (bound - pos) / u
            if t > 0:
                ts.append(t)
    return float(min(ts)) if ts else float(2 * AIRSPACE_LIMIT)


class _Episode:
    """One re-flown episode. A transcription of ``Flight4DEnv``, not a model of it.

    Written as a scalar loop rather than vectorised numpy on purpose: the test
    that guards this file asserts *exact* agreement with the environment, and
    the surest way to get that is to do the same arithmetic in the same order.
    """

    def __init__(self, state0):
        state0 = np.asarray(state0, dtype=np.float64)
        self.F = int(state0.shape[0])
        self.x = [float(v) for v in state0[:, 0]]
        self.y = [float(v) for v in state0[:, 1]]
        self.alt = [float(v) for v in state0[:, 2]]
        self.vel = [float(v) for v in state0[:, 3]]
        hdg = [float(v) for v in state0[:, 4]]
        self.ux = [math.cos(math.radians(h)) for h in hdg]
        self.uy = [math.sin(math.radians(h)) for h in hdg]

        self.alt0 = list(self.alt)
        self.v0 = list(self.vel)
        self.s_total = [_distance_to_boundary(self.x[i], self.y[i],
                                              self.ux[i], self.uy[i])
                        for i in range(self.F)]
        self.t_target = [self.s_total[i] / max(self.v0[i], 1e-9)
                         for i in range(self.F)]
        self.s_flown = [0.0] * self.F
        self.t_exit_actual = [math.nan] * self.F

        self.step_idx = 0
        self.n_clearances = 0
        self.n_invalids = 0
        self.touched = [False] * self.F
        self.hold_left = [0] * self.F
        self.pulse_left = [0] * self.F
        self.pulse_sign = [0] * self.F
        self.collided = False
        self.congestion_total = 0

    # -- legality ---------------------------------------------------------
    def action_masks(self):
        mask = [False] * (1 + N_CLEARANCES * self.F)
        mask[0] = True
        d2 = 2 * DALT
        t_now = self.step_idx * DT

        for i in range(self.F):
            base = 1 + N_CLEARANCES * i
            if self.hold_left[i] > 0 or self.pulse_left[i] > 0:
                continue                       # busy: nothing is legal for it
            alt, vel = self.alt[i], self.vel[i]
            s_left = max(self.s_total[i] - self.s_flown[i], 0.0)

            mask[base + FL_DEC] = alt - DALT >= ALT_MIN
            mask[base + FL_INC] = alt + DALT <= ALT_MAX
            mask[base + FL_DEC2_T] = alt - d2 >= ALT_MIN
            mask[base + FL_INC2_T] = alt + d2 <= ALT_MAX

            # The schedule this aircraft is held to, not the episode clock. Most
            # aircraft leave the sector well before the horizon, so using
            # max_steps here would overstate the time they have left and make
            # every speed clearance look reachable.
            t_sched_left = max(self.t_target[i] - t_now, 0.0)
            for cmd, dv in ((SPD_DN_T, -DVEL), (SPD_UP_T, DVEL)):
                if not (VEL_MIN <= vel + dv <= VEL_MAX):
                    continue
                if s_left <= 0.0 or t_sched_left <= 0.0:
                    mask[base + cmd] = True      # already out; nothing to break
                    continue
                pulse_t = 2 * HOLD_STEPS * DT
                s_pulse = vel * min(pulse_t, t_sched_left)
                s_rest = max(s_left - s_pulse, 0.0)
                t_rest = max(t_sched_left - pulse_t, 0.0)
                if t_rest <= 0.0:
                    mask[base + cmd] = pulse_t <= t_sched_left + 1e-6
                    continue
                mask[base + cmd] = bool(
                    s_rest <= VEL_MAX * t_rest + 1e-6
                    and s_rest >= VEL_MIN * t_rest - 1e-6)

            mask[base + RESUME] = not self._on_plan(i)
        return mask

    def _on_plan(self, i: int) -> bool:
        return (abs(self.alt[i] - self.alt0[i]) < 1e-9
                and abs(self.vel[i] - self.v0[i]) < 1e-9)

    # -- clearances -------------------------------------------------------
    def _change_altitude(self, i: int, delta: float) -> None:
        self.alt[i] += delta
        if self.alt[i] < ALT_MIN:          # Flight clamps; the mask prevents it
            self.alt[i] = ALT_MIN

    def _change_velocity(self, i: int, delta: float) -> None:
        self.vel[i] += delta
        if self.vel[i] < VEL_MIN:
            self.vel[i] = VEL_MIN

    def _issue(self, i: int, cmd: int) -> None:
        if cmd == FL_DEC:
            self._change_altitude(i, -DALT)
        elif cmd == FL_INC:
            self._change_altitude(i, DALT)
        elif cmd in (FL_DEC2_T, FL_INC2_T):
            sign = -1 if cmd == FL_DEC2_T else 1
            self._change_altitude(i, sign * 2 * DALT)
            self.hold_left[i] = HOLD_STEPS
        elif cmd in (SPD_DN_T, SPD_UP_T):
            # A symmetric pulse: +dv for hold_steps, -dv for hold_steps, then
            # nominal. 2H + 1 because `_advance_holds` ticks inside this very
            # step; 2H would leave one dv*DT of along-track debt unpaid.
            sign = -1 if cmd == SPD_DN_T else 1
            self._change_velocity(i, sign * DVEL)
            self.pulse_left[i] = 2 * HOLD_STEPS + 1
            self.pulse_sign[i] = sign
        elif cmd == RESUME:
            self._restore(i)

    def _advance_holds(self) -> None:
        for i in [k for k in range(self.F) if self.hold_left[k] > 0]:
            self.hold_left[i] -= 1
            if self.hold_left[i] == 0:
                self._restore(i)
        for i in [k for k in range(self.F) if self.pulse_left[k] > 0]:
            self.pulse_left[i] -= 1
            if self.pulse_left[i] == HOLD_STEPS:
                # halfway: reverse the excursion to pay back the along-track debt
                self._change_velocity(
                    i, self.v0[i] - self.pulse_sign[i] * DVEL - self.vel[i])
            elif self.pulse_left[i] == 0:
                self._restore(i)

    def _restore(self, i: int) -> None:
        self._change_altitude(i, self.alt0[i] - self.alt[i])
        self._change_velocity(i, self.v0[i] - self.vel[i])
        self.hold_left[i] = 0
        self.pulse_left[i] = 0
        self.pulse_sign[i] = 0

    # -- dynamics ---------------------------------------------------------
    def step(self, action: int):
        idx = int(action)
        if idx > 0:
            i, cmd = divmod(idx - 1, N_CLEARANCES)
            if not self.action_masks()[idx]:
                self.n_invalids += 1
            else:
                self._issue(i, cmd)
                self.n_clearances += 1
                self.touched[i] = True

        self._advance_holds()
        v_before = list(self.vel)
        for i in range(self.F):
            distance = self.vel[i] * DT
            self.x[i] += distance * self.ux[i]
            self.y[i] += distance * self.uy[i]
            self.s_flown[i] += v_before[i] * DT
        self.step_idx += 1
        self._record_exits(v_before)

        self.congestion_total += self._count_congestion()
        self.collided = self._is_collision()

        terminated = self.collided
        truncated = (self.step_idx >= MAX_STEPS
                     or self.n_clearances >= MAX_CLEARANCES)
        return terminated, truncated

    def _record_exits(self, v_before) -> None:
        """Stamp the crossing time for anything that left the box this step.

        Interpolated within the step, not rounded to the step boundary: at
        200-515 m/s one step is up to a kilometre, and rounding would put a
        quantisation floor under the very quantity the 4D objective scores.
        """
        t_after = self.step_idx * DT
        for i in range(self.F):
            if math.isfinite(self.t_exit_actual[i]):
                continue
            if self.s_flown[i] >= self.s_total[i]:
                overshoot = self.s_flown[i] - self.s_total[i]
                self.t_exit_actual[i] = t_after - overshoot / max(v_before[i], 1e-9)

    def _count_congestion(self) -> int:
        c = 0
        for i in range(self.F):
            for j in range(i + 1, self.F):
                if abs(self.alt[i] - self.alt[j]) >= ALT_MIN_SEP:
                    continue
                if math.hypot(self.x[i] - self.x[j],
                              self.y[i] - self.y[j]) <= D_SEP:
                    c += 1
        return c

    def _is_collision(self) -> bool:
        for i in range(self.F):
            for j in range(i + 1, self.F):
                if abs(self.alt[i] - self.alt[j]) >= DALT:
                    continue
                if math.hypot(self.x[i] - self.x[j],
                              self.y[i] - self.y[j]) <= COLLISION_RADIUS:
                    return True
        return False

    # -- the 4D gate ------------------------------------------------------
    def exit_deviation(self):
        """``(seconds late, flight levels off plan)``, per aircraft.

        Two regimes: before the boundary the deviation is a *projection* at the
        current speed; after it, the recorded crossing time is a *fact* and must
        stop moving. Projecting past the exit makes an untouched aircraft
        accumulate a phantom delay it had no way to cause.
        """
        t_now = self.step_idx * DT
        dt, d_alt = [], []
        for i in range(self.F):
            if math.isfinite(self.t_exit_actual[i]):
                t_exit = self.t_exit_actual[i]
            else:
                s_left = max(self.s_total[i] - self.s_flown[i], 0.0)
                t_exit = t_now + s_left / max(self.vel[i], 1e-9)
            dt.append(t_exit - self.t_target[i])
            d_alt.append((self.alt[i] - self.alt0[i]) / DALT)
        return dt, d_alt

    def kpis(self) -> dict:
        dt, d_alt = self.exit_deviation()
        scale = HOLD_STEPS * DT
        per_aircraft = [abs(a) / scale + abs(b) for a, b in zip(dt, d_alt)]
        return {
            "completed": self.step_idx >= MAX_STEPS,
            "congestion": self.congestion_total,
            "exit_miss": float(np.mean(per_aircraft)),
            "max_exit_miss": float(max(per_aircraft)),
            "clearances": self.n_clearances,
            "invalids": self.n_invalids,
            "aircraft_involved": int(sum(self.touched)),
            "steps": self.step_idx,
        }


def replay_episode(state0: np.ndarray, actions) -> dict:
    """Re-fly one episode.

    state0:  ``[n_flights, 5]`` initial (x, y, alt, vel, hdg-degrees), as the
             environment stands after ``reset()``.
    actions: ``[steps]`` flat clearance index per step. A trace shorter than the
             horizon is padded with the idle action, so an early-ending run is
             still measured over the full horizon rather than scoring less
             exposure.
    """
    episode = _Episode(state0)
    actions = np.asarray(actions, dtype=np.int64).ravel()
    for step in range(MAX_STEPS):
        action = int(actions[step]) if step < len(actions) else 0
        terminated, truncated = episode.step(action)
        if terminated or truncated:
            break
    return episode.kpis()


# --- Kaggle entry point --------------------------------------------------

_ID_PATTERN = re.compile(r"^s(\d+)_t(\d+)$")
_STATE_FIELDS = ("x", "y", "alt", "vel", "hdg")


def _parse_ids(ids: pd.Series):
    parts = ids.str.extract(_ID_PATTERN)
    if parts.isna().any().any():
        bad = ids[parts.isna().any(axis=1)].iloc[0]
        raise ParticipantVisibleError(
            f"Malformed row id {bad!r}. Expected s<seed>_t<step>."
        )
    return parts[0].astype(np.int64), parts[1].astype(int)


def _state_columns(solution: pd.DataFrame) -> list:
    """``f0_x … fN_hdg``, in aircraft order, checked for completeness."""
    n_flights = sum(1 for c in solution.columns
                    if re.fullmatch(r"f\d+_x", str(c)))
    if n_flights == 0:
        raise ParticipantVisibleError(
            "Solution file carries no aircraft state columns. This is a "
            "host-side problem, not yours — please report it.")
    columns = [f"f{i}_{field}"
               for i in range(n_flights) for field in _STATE_FIELDS]
    missing = [c for c in columns if c not in solution.columns]
    if missing:
        raise ParticipantVisibleError(
            f"Solution file is missing the {missing[0]!r} column. This is a "
            f"host-side problem, not yours — please report it.")
    return columns


def score(solution: pd.DataFrame, submission: pd.DataFrame,
          row_id_column_name: str) -> float:
    """Flight Challenge — air traffic conflict resolution. **Lower is better.**

    Participants control a sector of straight-line aircraft converging on a
    hotspot. They submit the **action taken at every (seed, step)**; this metric
    re-flies each episode from the initial state carried in the solution file and
    computes the KPIs itself. There is no number in the submission to inflate — a
    trace that was not produced by a good policy does not replay like one.

    Ranking is **lexicographic** on four objectives, every one lower-is-better:

    1. ``failed`` — episodes that did not fly the full horizon. First on purpose:
       an episode that ends early accumulates fewer conflicts simply by existing
       for less time, so ranking on safety alone would make crashing out a
       strategy.
    2. ``congestion`` — loss-of-separation events, summed over every step.
    3. ``exit_miss`` — how far each aircraft ends from the 4D exit point it was
       headed for before it was touched. This is what stops "slow everything
       down" from winning.
    4. ``clearances`` — instructions issued.

    Kaggle sorts a single float, so the four are packed positionally into one
    number: each is clamped to a documented bound and weighted by the product of
    all lower-priority ranges, so a difference in a higher-priority objective
    always dominates any possible difference below it. The packed value stays
    under ``2**53`` and is therefore exact in float64. **Only the ordering is
    meaningful** — the magnitude is not a quantity anyone should read. The
    competition's own fifth component, distinct aircraft touched, does not fit
    that budget and breaks ties only on the organisers' private board.

    Submission format — one row per (seed, step), two columns::

        id,action
        s0000042_t00,0
        s0000042_t01,18

    ``action`` is the environment's flat clearance index: ``0`` idles, and
    ``1 + 7 * aircraft + clearance`` issues one clearance to one aircraft. Every
    seed must carry the **whole horizon**; a trace that stops when its episode
    ended is missing rows and is rejected.

    Requires only pandas, numpy and the standard library — no gymnasium, no
    torch. Scoring 100 episodes takes well under a second.

    Two aircraft head-on at the same flight level, three steps, doing nothing:

    >>> import pandas as pd
    >>> state = {"f0_x": 0.0, "f0_y": -5000.0, "f0_alt": 350.0,
    ...          "f0_vel": 300.0, "f0_hdg": 90.0,
    ...          "f1_x": 0.0, "f1_y": 5000.0, "f1_alt": 350.0,
    ...          "f1_vel": 300.0, "f1_hdg": 270.0}
    >>> ids = ["s0000001_t00", "s0000001_t01", "s0000001_t02"]
    >>> solution = pd.DataFrame([dict(id=i, **state) for i in ids])
    >>> submission = pd.DataFrame({"id": ids, "action": [0, 0, 0]})
    >>> score(solution.copy(), submission.copy(), "id")
    160048002.0

    Issuing a clearance that was not needed scores *worse*, because efficiency is
    one of the ranked objectives:

    >>> spender = pd.DataFrame({"id": ids, "action": [2, 0, 0]})
    >>> score(solution.copy(), spender.copy(), "id")
    162048103.0

    Every rejection names the fix, so a participant can act on it unaided. (The
    messages are printed rather than raised here only so the example reads the
    same whether this file is imported as a module or pasted into a notebook
    cell, where the exception's qualified name would differ.)

    >>> short = pd.DataFrame({"id": ids[:2], "action": [0, 0]})
    >>> try:
    ...     score(solution.copy(), short.copy(), "id")
    ... except ParticipantVisibleError as err:
    ...     print(err)
    Submission is missing 1 of 3 required rows. Rebuild it with `python -m competition.run submit`.

    >>> illegal = pd.DataFrame({"id": ids, "action": [0, 0, 99]})
    >>> try:
    ...     score(solution.copy(), illegal.copy(), "id")
    ... except ParticipantVisibleError as err:
    ...     print(err)
    Every 'action' must be in the range 0-14.
    """
    for frame, name in ((solution, "solution"), (submission, "submission")):
        if row_id_column_name not in frame.columns:
            raise ParticipantVisibleError(
                f"{name} is missing the id column {row_id_column_name!r}.")

    state_columns = _state_columns(solution)
    n_flights = len(state_columns) // len(_STATE_FIELDS)
    highest = N_CLEARANCES * n_flights

    if "action" not in submission.columns:
        raise ParticipantVisibleError(
            "Submission needs an 'action' column holding the flat clearance "
            f"index: 0 idles, and 1 + {N_CLEARANCES} * aircraft + clearance "
            "issues one clearance to one aircraft.")

    try:
        actions = submission["action"].astype(int)
    except (TypeError, ValueError):
        raise ParticipantVisibleError(
            f"Every 'action' must be an integer 0-{highest}.")
    if not actions.between(0, highest).all():
        raise ParticipantVisibleError(
            f"Every 'action' must be in the range 0-{highest}.")

    sol_seed, sol_step = _parse_ids(solution[row_id_column_name])
    sub_seed, sub_step = _parse_ids(submission[row_id_column_name])

    sol = solution.assign(_seed=sol_seed, _step=sol_step)
    sub = submission.assign(_seed=sub_seed, _step=sub_step, _action=actions)

    required = set(zip(sol["_seed"], sol["_step"]))
    provided = set(zip(sub["_seed"], sub["_step"]))
    if not required <= provided:
        missing = len(required - provided)
        raise ParticipantVisibleError(
            f"Submission is missing {missing} of {len(required)} required rows. "
            f"Rebuild it with `python -m competition.run submit`.")

    scenario = sol[sol["_step"] == 0].drop_duplicates("_seed").set_index("_seed")
    lookup = {int(seed): np.asarray(row, dtype=np.float64).reshape(n_flights, -1)
              for seed, row in scenario[state_columns].iterrows()}

    sub = sub.sort_values(["_seed", "_step"])
    results = []
    for seed, group in sub[sub["_seed"].isin(lookup)].groupby("_seed"):
        trace = np.zeros(int(group["_step"].max()) + 1, dtype=np.int64)
        trace[group["_step"].to_numpy()] = group["_action"].to_numpy()
        results.append(replay_episode(lookup[int(seed)], trace))

    if not results:
        raise ParticipantVisibleError("No scored episodes found in the submission.")

    return pack(
        failed=sum(1 for r in results if not r["completed"]),
        congestion=sum(r["congestion"] for r in results),
        exit_miss_mean=float(np.mean([r["exit_miss"] for r in results])),
        clearances=sum(r["clearances"] for r in results),
    )
