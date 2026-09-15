"""Rule-based controller for :mod:`envs.flight_4d` — the bar an agent must clear.

The point of a hand-written controller is not that it is good. It is that
without one, "the learner beats do-nothing and a random legal policy" is the
strongest claim anyone can make, and that is a very low bar — a random policy
crashes in 34 of 100 episodes. This gives the leaderboard something competent to
sit above, the way a priority controller did for the environment this replaced.

Like that controller, this one works **from the observation vector only**. No
peeking at the simulator, no privileged conflict list. It gets exactly what the
policy gets.

Why it does NOT use the two-level macro
--------------------------------------
The obvious shortcut is ``FL_INC2_T``: two levels instantly, one clearance, and
it comes home by itself. It was tried first and it is worse than a random legal
policy — 20 failures in 40 against random's 34 in 100.

The reason is that the macro reverts on a **fixed timer** the controller does
not control. It snaps back inside ``env.step()``, before the next observation,
so there is no opportunity to react; and if the traffic has not cleared by then
it drops the aircraft straight back onto it. Diagnosed over 20 failures: 12 were
"displaced, reverted, hit", 8 were never commanded, and **zero** happened to an
aircraft while it was still displaced. Sweeping the just-in-time lead from 1 to 6
steps never got below 50% failures.

So this controller does what ``PriorityLevelController`` does: permanent level
changes for lasting separation, and an explicit ``RESUME`` once the aircraft is
genuinely clear. The recovery gate is the whole point — deciding *when* to come
home is a judgement the macro takes away from you. The macro is the right tool
for a transient encounter and the wrong one for sustained separation.

The strategy
------------
1. **Assign slots.** All converging traffic enters co-level. With ``ALT_MIN_SEP``
   two levels wide, the conflict-free slots are base, +/-2, +/-4. Earliest
   arrival keeps its level; later arrivals move progressively further.
2. **Steer, one level per step.** Only one aircraft may be commanded per step,
   so the most urgent gets the clearance.
3. **Come home when clear.** Once an aircraft is at least
   ``RECOVERY_SEP_FACTOR`` hotspot radii from every other converging flight,
   ``RESUME`` restores its level and speed in one instruction. Only one
   clearance is issued per step, so when several aircraft are clear at once
   they are resumed in the same order they were first displaced — the ordering
   the controller already committed to at assignment time — rather than by
   flight index.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from envs.flight_4d import (N_CLEARANCES, N_FEATS, N_GLOBALS, FL_DEC, FL_INC,
                            FL_DEC2_T, FL_INC2_T, SPD_DN_T, SPD_UP_T, RESUME)
from envs.simulator import CollisionCourseSimulator as Sim

# Feature offsets within a per-aircraft block. Mirrors `FEATURE_NAMES`.
X, Y, ALT = 0, 1, 2
D_ALT, HOLD_LEFT, SEVERITY, T_CONFLICT, ACTIVE = 7, 9, 10, 11, 12


def decode(obs: np.ndarray, n_flights: int, max_steps: int,
           hold_steps: int) -> dict:
    """Undo the observation scaling. Everything here is in physical units.

    The environment squashes two of these through ``tanh`` and one through
    ``t/(t+horizon)``, and both are inverted rather than approximated — a
    controller that guesses at its own inputs is not a baseline, it is a second
    source of error.
    """
    flights = obs[:n_flights * N_FEATS].reshape(n_flights, N_FEATS)
    clock = (obs[n_flights * N_FEATS] + 1.0) / 2.0          # in [0, 1]
    step = clock * max_steps
    horizon = max(max_steps - step, 1.0) * Sim.DT           # seconds

    alt01 = (flights[:, ALT] + 1.0) / 2.0
    level = alt01 * (Sim.N_ALTS - 1)                        # flight-level index

    # severity and the manoeuvre countdown are plain affine maps
    severity = (flights[:, SEVERITY] + 1.0) / 2.0
    hold_left = (flights[:, HOLD_LEFT] + 1.0) / 2.0 * (2 * hold_steps)

    # t_conflict is q = t/(t+horizon); q == 1 means "no conflict predicted",
    # which is a genuine infinity rather than a very large number — invert it
    # explicitly instead of dividing by an epsilon and hoping.
    q = np.clip((flights[:, T_CONFLICT] + 1.0) / 2.0, 0.0, 1.0)
    none_pending = q >= 1.0 - 1e-6
    t_conflict = np.where(none_pending, np.inf,
                          q * horizon / np.maximum(1.0 - q, 1e-12))

    # d_alt came through tanh(dalt / 2)
    d_alt = 2.0 * np.arctanh(np.clip(flights[:, D_ALT], -0.999999, 0.999999))

    return dict(level=level, severity=severity, hold_left=hold_left,
                t_conflict=t_conflict, d_alt=d_alt, step=step,
                x=flights[:, X] * Sim.AIRSPACE_LIMIT,
                y=flights[:, Y] * Sim.AIRSPACE_LIMIT)


class Priority4DController:
    """Displace the most endangered aircraft two levels, just in time.

    Exposes SB3's ``predict(obs, deterministic=...)`` so the same scoring loop
    runs it, a PPO checkpoint or a competition submission interchangeably.

    Parameters
    ----------
    n_flights, max_steps, hold_steps
        Must match the environment. They are not recoverable from the
        observation: its length pins down ``n_flights`` only once you already
        know ``N_FEATS``, and the horizon is needed to invert ``t_conflict``.
    severity_gate
        Ignore predicted conflicts weaker than this. Severity is
        ``exp(-d_cpa/R) * exp(-d_alt/ALT_MIN_SEP)``, so a pair two levels apart
        is not in the list at all and a distant co-level pair scores low. The
        gate stops the controller spending clearances on encounters that were
        never going to breach.
    lead_steps
        Act when the conflict is within this many steps. A hold lasts
        ``hold_steps``; issuing earlier than that wastes cover, because the
        aircraft snaps back before the closest approach.
    """

    #: Two flight levels is ALT_MIN_SEP: at or beyond this, no conflict.
    SEP_LEVELS = 2
    #: Clear of every other converging flight by this many hotspot radii before
    #: coming home. They all share an entry level, so returning while still
    #: bunched simply re-creates the conflict that was just resolved.
    #:
    #: This is THE tuning knob, and it is a straight safety-against-timeliness
    #: trade. Measured over 40 seeds:
    #:
    #:     radii   failed   congestion   conflict-free   exit_miss   clearances
    #:       3.0       16           66             38%       0.235          322
    #:       5.0        7           48             58%       0.355          316
    #:   -> 7.0         4           46             60%       0.580          301
    #:      10.0        3           42             62%       0.890          278
    #:     never        2           41             65%       1.265          253
    #:
    #: 7.0 is the last row that beats a random legal policy on EVERY column
    #: (10/234/8%/0.763/1262). Further out buys a failure or two at the cost of
    #: leaving aircraft parked off-level, and an `exit_miss` worse than random's
    #: means the 4D objective is no longer being served at all.
    RECOVERY_SEP_FACTOR = 7.0

    def __init__(self, n_flights: int = 5, max_steps: int = 50,
                 hold_steps: int = 5, severity_gate: float = 0.05):
        self.n_flights = int(n_flights)
        self.max_steps = int(max_steps)
        self.hold_steps = int(hold_steps)
        self.severity_gate = float(severity_gate)
        self.mask_offset = self.n_flights * N_FEATS + N_GLOBALS
        self._target: Optional[np.ndarray] = None
        self._order: list[int] = []
        self._last_step = np.inf

    def reset(self) -> None:
        self._target = None
        self._order = []
        self._last_step = np.inf

    # ------------------------------------------------------------------
    def predict(self, obs, deterministic: bool = True):
        obs = np.asarray(obs, dtype=np.float64).ravel()
        mask = obs[self.mask_offset:] > 0.5
        d = decode(obs, self.n_flights, self.max_steps, self.hold_steps)

        if d["step"] < self._last_step:          # new episode
            self._assign(d)
        self._last_step = d["step"]

        a = self._steer(d, mask)
        if a is None:
            a = self._recover(d, mask)
        return np.int64(0 if a is None else a), None

    # ------------------------------------------------------------------
    def _assign(self, d) -> None:
        """One conflict-free slot each, priority to the earliest arrival.

        Converging traffic is identified by having a predicted conflict at all —
        background flights already cross at other levels and never appear. That
        makes the controller indifferent to `n`, which varies per episode and is
        not in the observation.
        """
        base = np.rint(d["level"]).astype(int)
        self._target = base.copy()
        involved = [i for i in range(self.n_flights)
                    if d["severity"][i] >= self.severity_gate]
        if not involved:
            return
        involved.sort(key=lambda i: d["t_conflict"][i])   # earliest keeps its level
        # Recovery order mirrors assignment order (see `_recover`): the aircraft
        # displaced first is the one an earliest-first controller is already
        # committed to prioritising, so it comes home first too.
        self._order = list(involved)

        # Build the slot list ONCE, from the shared entry level, and hand out
        # distinct slots. The obvious version — base[i] + offset[k], clipped to
        # the band — is wrong near a rail: clipping collapses +2 and +4 onto the
        # same level and quietly assigns two aircraft the same slot. Measured,
        # that was 8 of 10 remaining failures, every one of them a pair sitting
        # exactly where it had been told to sit.
        lo, hi = 0, Sim.N_ALTS - 1
        ref = int(np.rint(np.median(d["level"][involved])))
        slots = [ref + o for o in (0, 2, -2, 4, -4, 6, -6) if lo <= ref + o <= hi]
        for k, i in enumerate(involved):
            self._target[i] = slots[k] if k < len(slots) else ref

    def _steer(self, d, mask) -> Optional[int]:
        """One level per step, toward the assigned slot, most urgent first."""
        if self._target is None:
            return None
        gap = self._target - np.rint(d["level"]).astype(int)
        need = np.flatnonzero(gap != 0)
        if len(need) == 0:
            return None
        # Soonest conflict first; that is the aircraft whose slot matters now.
        for i in sorted(need, key=lambda i: d["t_conflict"][i]):
            cmd = FL_INC if gap[i] > 0 else FL_DEC
            a = 1 + N_CLEARANCES * i + cmd
            if mask[a]:
                return int(a)
        return None

    def _recover(self, d, mask) -> Optional[int]:
        """Home again, once genuinely clear of the traffic that mattered.

        Distance from the other converging flights, not from the hotspot: they
        all share an entry level, so it is separation from *them* that decides
        whether coming home is safe. Keying on the hotspot over-waits the pairs
        that diverge quickly.

        Candidates are tried in **assignment order** (`self._order`, set once in
        `_assign` and sorted by `t_conflict` — earliest arrival first), not by
        flight index. Only one `RESUME` can be issued per step, so with several
        aircraft simultaneously clear the flight-index order silently favours
        low-index aircraft for no reason connected to the problem. Assignment
        order is a real ordering already computed for a reason: it is priority,
        so the aircraft priority put back on plan first is the one it decided
        mattered first.
        """
        D_SEP = Sim.AIRSPACE_LIMIT * Sim.hotspot2airspace_factor
        reach = self.RECOVERY_SEP_FACTOR * D_SEP
        off_plan = [i for i in self._order if abs(d["d_alt"][i]) > 1e-3]
        for i in off_plan:
            near = False
            for j in range(self.n_flights):
                if j == i:
                    continue
                if math.hypot(d["x"][i] - d["x"][j], d["y"][i] - d["y"][j]) < reach:
                    near = True
                    break
            if near:
                continue
            a = 1 + N_CLEARANCES * i + RESUME
            if mask[a]:
                # Retire the slot. Otherwise `_steer` sees a gap the instant the
                # aircraft is home and flies it straight back out — the
                # controller undoing its own recovery, one clearance at a time.
                # `d_alt` gives the deviation, so the entry level is recoverable
                # from the observation without remembering it.
                self._target[i] = int(np.rint(d["level"][i] - d["d_alt"][i]))
                return int(a)
        return None


class Noop4DController:
    """Do nothing, ever. The floor every other number is read against."""

    def reset(self) -> None:
        pass

    def predict(self, obs, deterministic: bool = True):
        return np.int64(0), None


class RandomLegalController:
    """Uniform over whatever the mask allows — what the action space scores for free.

    The second untrained baseline, beside the brick: the brick bounds what
    *inaction* buys, this bounds the *action space*. Nothing below it has
    learned anything.

    Reads the mask off the tail of the observation, exactly as the policy
    does, so it never issues an invalid action. Its random stream is seeded
    from the **first observation of the episode** (plus a fixed salt), so the
    same scenario always draws the same actions: the row is reproducible run
    to run, and a "deterministic" and a "stochastic" pass over it are the same
    pass — there is no distribution here to sample differently from.
    """

    def __init__(self, n_actions: int = 1 + N_CLEARANCES * 5, seed: int = 0):
        self.n_actions = int(n_actions)
        self._seed = int(seed)
        self._rng = None

    def reset(self) -> None:
        self._rng = None            # re-seed from the next observation

    def predict(self, obs, deterministic: bool = True):
        obs = np.asarray(obs, dtype=np.float32).ravel()
        if self._rng is None:
            import zlib
            self._rng = np.random.default_rng([self._seed, zlib.crc32(obs.tobytes())])
        legal = np.flatnonzero(obs[-self.n_actions:] > 0.5)
        return np.int64(self._rng.choice(legal)), None
