"""A fresh environment built around the 4D exit constraint.

The **baseline solution** for Session #3, and the environment the leaderboard
now scores. Not a subclass of anything: ``ShapedFlightEnv`` and ``EndpointTimelinessEnv``
grew by accretion on top of the original environment, and the overrides made
the reward hard to reason about end to end. Everything that matters now lives in
this one file; only the *kinematics* are reused, from
``CollisionCourseSimulator``, so scenarios stay comparable to the competition.

The problem this design attacks
-------------------------------
The tension is between **holding the 4D exit point** and **resolving conflicts
with speed**. A velocity clearance shifts an aircraft's whole temporal
trajectory, so meeting the exit slot afterwards needs an exact compensating
clearance later — which fights the third objective, *minimise clearances*. An
agent asked to discover a matched ``+dv … -dv`` pair by independent sampling at
each step is being asked for a ``p^2`` coincidence, and it will not find one.

Four things follow, and they are the whole design:

1. **Every disturbing clearance auto-reverts.** ``SPD_UP_T`` is "+dv, hold, then
   return to plan" as *one* instruction. The 4D recovery is baked into the
   clearance rather than left as a second decision the agent must remember to
   take. Same for the two-level altitude macro, which the injection probe showed
   is the minimum effective vertical intervention (``DALT`` 12 vs
   ``ALT_MIN_SEP`` 24: one level leaves the pair in conflict).

   Note what "minimum effective" does and does not claim. It is about the
   OUTBOUND half: two levels is the smallest displacement that resolves a pair,
   and one clearance is the cheapest way to buy it. The RETURN half is on a
   fixed timer the caller does not control, which makes the macro the right tool
   for a transient encounter and the wrong one for sustained separation --- see
   ``agents/rule_based_4d.py``, where using it measured worse than a random
   legal policy. Both statements are true and they are about different halves of
   the same instruction.

2. **The temporal deviation is in the observation.** A cost the agent cannot see
   is one it can only respond to by accident, and ``dt_exit`` is the quantity the
   third objective is actually about.

3. **Three direct per-step costs, not potentials.** Predicted severity is already
   dense — a conflict twenty steps out is in today's sum — so it does not need
   shaping to be learnable, and a direct cost says what we actually mean. The
   trade is that a direct cost *can* move the optimum, where potential-based
   shaping provably cannot (Ng, Harada & Russell 1999); that is a deliberate
   choice, not an oversight. The clearance count is a *transition cost* either
   way — a property of the action, not of the state.

4. **Hard action masking**, including kinematic dead ends. A clearance that makes
   the exit slot unreachable even at ``v_max`` is removed from the action space
   rather than punished after the fact.

Scenario
--------
``n + m = 5`` always, with ``n`` drawn uniformly from ``{2, 3, 4}`` each reset —
so between two and four aircraft start on a collision course and the rest is
background traffic. Small enough that a 50-step episode can actually resolve
everything, varied enough that the policy cannot memorise one geometry.

A note on the "3 minutes" in the design brief
---------------------------------------------
``DT`` is **2 seconds** and the horizon is 50 steps, so an episode is 100 s end
to end. A three-minute hold is longer than the whole episode. ``hold_steps``
is therefore expressed in steps and defaults to 5 (10 s), which is long enough
to carry an aircraft through the 1000 m hotspot and short enough to leave room
for the return leg.
"""
from __future__ import annotations

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .simulator import CollisionCourseSimulator


# --- the clearance vocabulary ---------------------------------------------
# Index 0 of the flat action space is IDLE (command nobody). Everything else is
# (aircraft, clearance). Permanent and temporary variants both exist so the
# agent can express "put it there and leave it" as well as "step aside and come
# back"; which it prefers is a result, not an assumption.
FL_DEC, FL_INC, FL_DEC2_T, FL_INC2_T, SPD_DN_T, SPD_UP_T, RESUME = range(7)
N_CLEARANCES = 7
CLEARANCE_NAMES = ("FL_DEC", "FL_INC", "FL_DEC2_T", "FL_INC2_T",
                   "SPD_DN_T", "SPD_UP_T", "RESUME")

#: Per-aircraft observation features. Every value is scaled into [-1, 1].
FEATURE_NAMES = (
    "x", "y", "alt", "vel", "sin_hdg", "cos_hdg",
    "dt_exit",        # temporal deviation: projected exit time - target exit time
    "d_alt",          # vertical deviation from the entry level, in flight levels
    "n_clearances",   # how many clearances this aircraft has already had
    "hold_left",      # steps until an active temporary manoeuvre reverts
    "severity",       # worst predicted conflict, in [0, 1]
    "t_conflict",     # squashed time until it
    "active",         # 1 if this slot holds a real aircraft
)
N_FEATS = len(FEATURE_NAMES)
#: Episode clock, clearance budget remaining.
N_GLOBALS = 2


class Flight4DEnv(gym.Env):
    """Straight-line conflict resolution against a fixed 4D exit gate.

    Observation
        ``Box(-1, 1, (N_FEATS * n_flights + N_GLOBALS,))``. A single frame, not a
        stack: headings are never commanded, so every aircraft's future is a
        straight line determined by its own ``(x, y, alt, v, heading)``. Pairwise
        closing geometry is therefore *derivable* from one frame, which makes the
        state Markov once the pending-manoeuvre countdown is included — hence
        ``hold_left``. This is the substantive difference from the prior design, whose full
        simulator is genuinely partially observed and needs a GRU.

    Action
        ``Discrete(1 + N_CLEARANCES * n_flights)``. Index 0 idles; index
        ``1 + N_CLEARANCES*i + c`` issues clearance ``c`` to aircraft ``i``.
        Use :meth:`action_masks` — the environment owns the legality rules, so
        the policy never re-derives physics from normalised observations.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_flights: int = 5, n_range: tuple[int, int] = (2, 4),
                 max_steps: int = 50, hold_steps: int = 5,
                 w_safe: float = 3.0,
                 w_clearance: float = 0.15,
                 collision_rate: float = 3.0, gamma: float = 0.99,
                 max_clearances: int = 40, seed: int | None = None):
        super().__init__()
        self.n_flights = int(n_flights)
        self.n_lo, self.n_hi = n_range
        self.max_steps = int(max_steps)
        self.hold_steps = int(hold_steps)
        self.gamma = float(gamma)
        self.max_clearances = int(max_clearances)

        # Reward weights. Only the RATIOS matter, and they are fixed by design:
        #
        #   one severity-1 infringement, per step      = w_safe
        #   the whole fleet at maximum 4D deviation    = w_safe   (equal)
        #   one clearance                              = w_safe / 20
        #
        # The absolute scale is chosen to keep an episode's return around -80
        # rather than -800. Ratios are what the policy optimises; magnitude is
        # what the critic has to fit from a zero initialisation, and a value head
        # starting three orders of magnitude away from its target wastes early
        # training closing that gap. `scripts/calibrate_4d.py` prints the comparison.
        #
        # At w_clearance=0.35 a fully-intruding pair was worth 30 on the safety
        # term, paid IMMEDIATELY, against 0.35 for a clearance — **86:1 in favour of
        # acting**. A whole episode of 22.7 clearances totalled 6.64 discounted.
        # There was no brake, and the trained policy went to 26 clearances per
        # episode and scored *worse than a random legal policy* on the 4D
        # objective (exit_miss 0.844 against 0.590).
        #
        # All three terms are DIRECT per-step costs, so all three can move the
        # optimum — unlike potential-based shaping, which provably cannot. That is
        # the deliberate trade made in design point 3 above: a direct cost says
        # what we actually mean, and the price is that these weights are a
        # statement about the optimum itself, not merely about how fast it is
        # found. Size them with `scripts/calibrate_4d.py` before a training run.
        self.w_safe = float(w_safe)
        self.w_clearance = float(w_clearance)
        # Charged PER REMAINING STEP, not as a flat lump. The purpose is to
        # regularise the return: an episode that ends at step 10 banks ten steps
        # of cost and one that runs to 50 banks fifty, so returns are not
        # comparable across episodes and the critic has to fit that spread on top
        # of everything else. Charging the steps not flown completes every
        # episode to the same horizon.
        #
        # The rate is in units of w_safe, so there is no separate constant to
        # keep in sync: 3.0 means "the remaining steps, spent with one pair at
        # full severity, three times over". That is comfortably above the ~3.3
        # per step a real episode costs, so a mid-air is also never the cheap way
        # out — but the disincentive is a side effect, not the point.
        self.collision_rate = float(collision_rate)

        sim = CollisionCourseSimulator
        self.DT = sim.DT
        self.DALT = sim.DALT
        self.DVEL = sim.DVEL
        self.ALT_MIN_SEP = sim.ALT_MIN_SEP
        self.AIRSPACE = sim.AIRSPACE_LIMIT
        self.D_SEP = sim.AIRSPACE_LIMIT * sim.hotspot2airspace_factor   # 1000 m
        self.COLLISION_RADIUS = 100.0
        self.vel_min = sim.VEL_MIN
        self.vel_max = sim.VEL_MIN + (sim.N_VELS - 1) * sim.DVEL

        self.n_actions = 1 + N_CLEARANCES * self.n_flights
        self.action_space = spaces.Discrete(self.n_actions)
        # The legality mask is part of the OBSERVATION, not a side channel.
        # SB3's rollout buffer stores observations and nothing else, so a mask
        # passed any other way is present when the action is chosen and absent
        # when its log-probability is recomputed at update time --- PPO then
        # compares two different distributions and optimises a ratio nobody
        # intended. Carrying it here makes that impossible by construction, and
        # means the policy never re-derives physics from normalised features.
        self.observation_space = spaces.Box(
            -1.0, 1.0,
            shape=(N_FEATS * self.n_flights + N_GLOBALS + self.n_actions,),
            dtype=np.float32)
        #: Where the mask starts within the observation vector.
        self.mask_offset = N_FEATS * self.n_flights + N_GLOBALS
        self._seed = seed

    def set_n_range(self, lo: int, hi: int) -> None:
        """Narrow or widen the per-episode draw of ``n``, the converging traffic.

        The curriculum knob. ``n_flights`` — and therefore the observation and
        action spaces — is deliberately NOT touched: changing it mid-run would
        change the policy's input dimension, and the whole point of holding
        ``n + m`` fixed is that difficulty can vary while the spaces cannot.
        Only the *split* between collision-course and background traffic moves.

        Takes effect on the next :meth:`reset`, so a vectorised env can be
        stepped down mid-rollout without invalidating the transitions already in
        the buffer.
        """
        lo, hi = int(lo), int(hi)
        if not 1 <= lo <= hi <= self.n_flights:
            raise ValueError(
                f"need 1 <= lo <= hi <= n_flights ({self.n_flights}), "
                f"got lo={lo}, hi={hi}")
        self.n_lo, self.n_hi = lo, hi

    # ------------------------------------------------------------------
    # scenario
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        # n on a collision course, the rest background. Drawn per episode so the
        # policy meets a distribution of densities rather than one geometry.
        self.n = int(rng.integers(self.n_lo, self.n_hi + 1))
        self.m = self.n_flights - self.n

        self.sim = CollisionCourseSimulator(
            n=self.n, m=self.m,
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)))
        if hasattr(self.sim, "advance_into_airspace"):
            self.sim.advance_into_airspace()

        f = self.sim.flights
        self.alt0 = np.array([x.altitude for x in f], dtype=np.float64)
        self.v0 = np.array([x.velocity for x in f], dtype=np.float64)

        # The 4D gate. Heading is never commanded, so each track is a fixed line
        # and the exit POINT cannot move; only the arrival TIME can. So the gate
        # reduces to a distance-to-run and the time that distance takes at the
        # entry speed.
        self.s_total = np.array([self._distance_to_boundary(x) for x in f])
        self.t_target = self.s_total / np.maximum(self.v0, 1e-9)
        self.s_flown = np.zeros(self.n_flights)
        #: Recorded crossing time per aircraft; NaN until it leaves the box.
        self.t_exit_actual = np.full(self.n_flights, np.nan)

        self.step_idx = 0
        self.n_clearances = 0
        self.n_invalids = 0
        self.per_aircraft_clearances = np.zeros(self.n_flights, dtype=np.int64)
        self.touched = np.zeros(self.n_flights, dtype=bool)
        self.hold_left = np.zeros(self.n_flights, dtype=np.int64)
        #: Steps left in a symmetric speed pulse, and its opening direction.
        self.pulse_left = np.zeros(self.n_flights, dtype=np.int64)
        self.pulse_sign = np.zeros(self.n_flights, dtype=np.int64)
        self.collided = False
        self.congestion_total = 0
        self.last_congestion = 0

        return self._obs(), self._info()

    def _distance_to_boundary(self, flight) -> float:
        """How far this aircraft still has to fly before leaving the box.

        Straight line from its current position along its heading, intersected
        with the airspace square. Clipped so a flight already outside returns a
        positive, finite distance rather than a negative one.
        """
        ux, uy = flight.get_heading_unit_vectors()
        L = self.AIRSPACE
        ts = []
        for pos, u in ((flight.x, ux), (flight.y, uy)):
            if abs(u) < 1e-12:
                continue
            for bound in (-L, L):
                t = (bound - pos) / u
                if t > 0:
                    ts.append(t)
        return float(min(ts)) if ts else float(2 * L)

    # ------------------------------------------------------------------
    # legality
    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """Bool mask over the flat action space. Index 0 (idle) is always legal.

        Three families of illegality, and the third is the one the design brief
        singles out:

        **Rails.** A climb at the ceiling, a descent at the floor, ``+dv`` at
        ``v_max``, ``-dv`` at ``v_min``. The two-level macro needs *two* levels of
        headroom, not one — booking a climb that stops half way is exactly the
        half-manoeuvre it exists to remove.

        **Inertness.** ``RESUME`` on an aircraft already on plan, or any clearance
        to an aircraft mid-manoeuvre. Re-commanding a busy aircraft is how a
        policy spends budget without changing anything.

        **Kinematic dead ends.** A speed clearance is illegal if, after it, the
        remaining distance could not be covered in the remaining scheduled time
        even at ``v_max`` — or could not be stretched to it even at ``v_min``.
        Past that point the 4D slot is unreachable and no later clearance
        recovers it, so the action is removed rather than punished.
        """
        mask = np.zeros(1 + N_CLEARANCES * self.n_flights, dtype=bool)
        mask[0] = True
        lo, hi = self.sim.allowable_altitudes.min(), self.sim.allowable_altitudes.max()
        d2 = 2 * self.DALT
        t_now = self.step_idx * self.DT

        for i, f in enumerate(self.sim.flights):
            base = 1 + N_CLEARANCES * i
            if self.hold_left[i] > 0 or self.pulse_left[i] > 0:
                continue                       # busy: nothing is legal for it
            alt, vel = f.altitude, f.velocity
            s_left = max(self.s_total[i] - self.s_flown[i], 0.0)

            mask[base + FL_DEC] = alt - self.DALT >= lo
            mask[base + FL_INC] = alt + self.DALT <= hi
            mask[base + FL_DEC2_T] = alt - d2 >= lo
            mask[base + FL_INC2_T] = alt + d2 <= hi

            # The schedule this aircraft is actually held to, not the episode
            # clock. Getting this wrong is not a small error: computing the
            # remaining time from `max_steps` instead of the aircraft's own exit
            # time made speed clearances legal on **13-16%** of (aircraft, step)
            # pairs, which silently deleted the entire velocity axis — the one
            # the 4D tension is about. Most aircraft leave the sector well before
            # step 50, so the episode horizon massively overstates the time they
            # have left and every clearance looked like it would arrive early.
            t_sched_left = max(self.t_target[i] - t_now, 0.0)
            for cmd, dv in ((SPD_DN_T, -self.DVEL), (SPD_UP_T, self.DVEL)):
                if not (self.vel_min <= vel + dv <= self.vel_max):
                    continue
                if s_left <= 0.0 or t_sched_left <= 0.0:
                    mask[base + cmd] = True      # already out; nothing to break
                    continue
                # A symmetric pulse is schedule-neutral by construction, so the
                # only question is whether both halves fit before the gate and
                # whether the aircraft can still be flown at a legal speed after.
                pulse_t = 2 * self.hold_steps * self.DT
                s_pulse = vel * min(pulse_t, t_sched_left)
                s_rest = max(s_left - s_pulse, 0.0)
                t_rest = max(t_sched_left - pulse_t, 0.0)
                if t_rest <= 0.0:
                    mask[base + cmd] = pulse_t <= t_sched_left + 1e-6
                    continue
                mask[base + cmd] = bool(
                    s_rest <= self.vel_max * t_rest + 1e-6
                    and s_rest >= self.vel_min * t_rest - 1e-6)

            mask[base + RESUME] = not self._on_plan(i)
        return mask

    def _on_plan(self, i: int) -> bool:
        f = self.sim.flights[i]
        return (abs(f.altitude - self.alt0[i]) < 1e-9
                and abs(f.velocity - self.v0[i]) < 1e-9)

    # ------------------------------------------------------------------
    # reward terms
    # ------------------------------------------------------------------
    def _severity(self) -> float:
        """Total PREDICTED infringement severity, summed over the conflicts present.

        Straight from the simulator's own conflict definition, which is what makes
        this the right quantity rather than a hand-made proximity function:

        * ``predict_conflicts`` **omits** any pair already stacked at or beyond
          ``ALT_MIN_SEP``. A pair two levels apart is not in the list at all.
        * within the threshold, ``severity = exp(-d_cpa/R) * exp(-d_alt/ALT_MIN_SEP)``.

        So the vertical credit falls out of the physics instead of being chosen:
        one flight level removes ``1 - exp(-0.5) = 39%`` of the severity, and the
        second removes the remaining **61%** by deleting the conflict outright.
        An earlier version of this used ``(1 - d_alt/ALT_MIN_SEP)``, which paid a
        flat **50%** for a first level that resolves nothing --- and the trained
        policy duly bought exactly that, on every aircraft, at step 1, and stopped.

        It is dense without needing potential-based shaping: a conflict twenty
        steps away is already in the sum today.
        """
        horizon = max(self.max_steps - self.step_idx, 0) * self.DT
        if horizon <= 0:
            return 0.0
        return float(sum(c.severity for c in self.sim.predict_conflicts(horizon)))

    def _exit_deviation(self):
        """(temporal deviation in seconds, vertical deviation in flight levels).

        Two regimes, and conflating them was a real bug. **Before** the aircraft
        reaches the boundary the deviation is a *projection*: how late it will be
        if it holds its current speed. **After** it has crossed, the deviation is
        a *fact* — the recorded crossing time — and must stop moving.

        Clamping ``s_left`` at zero and projecting anyway pins the estimate to
        "now", so every aircraft that had already exited drifted later and later
        and an untouched, undisturbed aircraft accumulated a phantom delay it had
        no way to cause. Caught by
        ``test_untouched_aircraft_have_zero_exit_deviation``.
        """
        v_now = np.array([x.velocity for x in self.sim.flights])
        t_now = self.step_idx * self.DT
        s_left = np.maximum(self.s_total - self.s_flown, 0.0)
        t_projected = t_now + s_left / np.maximum(v_now, 1e-9)
        t_exit = np.where(np.isfinite(self.t_exit_actual),
                          self.t_exit_actual, t_projected)
        alt_now = np.array([x.altitude for x in self.sim.flights])
        return t_exit - self.t_target, (alt_now - self.alt0) / self.DALT

    def _record_exits(self, v_before: np.ndarray) -> None:
        """Stamp the crossing time for anything that left the box this step.

        Interpolated within the step rather than rounded to the step boundary:
        at 200-515 m/s one step is up to a kilometre, so rounding would put a
        quantisation floor under the very quantity the 4D objective scores.
        """
        crossed = (~np.isfinite(self.t_exit_actual)) & (self.s_flown >= self.s_total)
        if not crossed.any():
            return
        overshoot = self.s_flown[crossed] - self.s_total[crossed]
        t_after = self.step_idx * self.DT
        self.t_exit_actual[crossed] = t_after - overshoot / np.maximum(
            v_before[crossed], 1e-9)

    def collision_charge(self) -> float:
        """Cost of the steps this episode will now never fly.

        ``collision_rate * w_safe`` per remaining step. At step 0 that is the
        whole horizon; at step 49 it is one step. A flat charge made a late
        mid-air disproportionately expensive and an early one disproportionately
        cheap, which is exactly the length-dependent spread this removes.
        """
        return (self.collision_rate * self.w_safe
                * max(self.max_steps - self.step_idx, 0))

    def _deviation(self) -> np.ndarray:
        """Per-aircraft 4D deviation, each in ``[0, 1]``.

        Bounded on purpose. The scaling rule for the timeliness term is that the
        **whole fleet at maximum deviation costs exactly one severity-1
        infringement**, so each aircraft carries ``w_safe / n_flights``. That
        makes the priority explicit and checkable rather than a matter of taste:
        a single pair in full conflict outweighs every aircraft in the sector
        being as late and as far off-level as it is possible to be.

        An untouched aircraft is 0 by construction, so background traffic is free.
        """
        dt, dalt = self._exit_deviation()
        return np.clip(np.hypot(dt / (self.max_steps * self.DT),
                                dalt / (self.sim.N_ALTS - 1)), 0.0, 1.0)


    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def step(self, action):
        idx = int(np.asarray(action).ravel()[0])
        issued = 0

        if idx > 0:
            i, cmd = divmod(idx - 1, N_CLEARANCES)
            if not self.action_masks()[idx]:
                # Unreachable with a masked policy, and deliberately NOT charged
                # as a soft penalty. An invalid action is not something to learn
                # to avoid; it is something to remove from the action space. It
                # is counted so a broken policy is loud rather than silent.
                self.n_invalids += 1
            else:
                self._issue(i, cmd)
                issued = 1
                self.n_clearances += 1
                self.per_aircraft_clearances[i] += 1
                self.touched[i] = True

        self._advance_holds()
        v_before = np.array([x.velocity for x in self.sim.flights])
        self.sim.advance_clock()
        self.s_flown += v_before * self.DT
        self.step_idx += 1
        self._record_exits(v_before)

        self.last_congestion = self._count_congestion()
        self.congestion_total += self.last_congestion
        self.collided = self._is_collision()

        terminated = bool(self.collided)
        truncated = (self.step_idx >= self.max_steps
                     or self.n_clearances >= self.max_clearances)

        # Three terms, and nothing else.
        #
        #   safety      w_safe x (total predicted infringement severity)
        #   timeliness  (w_safe / n_flights) x (per-aircraft 4D deviation)
        #   action      w_clearance, once, for any non-NOOP clearance
        #
        # The scaling between the first two is the design statement: the whole
        # fleet at maximum deviation costs exactly ONE severity-1 infringement,
        # so safety strictly dominates whenever more than one pair is in trouble.
        #
        # These are direct per-step costs, not potentials. Predicted severity is
        # already dense --- a conflict twenty steps out is in today's sum --- so
        # it does not need shaping to be learnable, and a direct cost says what we
        # actually mean. The trade is that this CAN move the optimum, where
        # potential-based shaping provably cannot; that is a deliberate choice.
        reward = -(self.w_safe * self._severity()
                   + (self.w_safe / self.n_flights) * float(self._deviation().sum())
                   + self.w_clearance * issued)
        if terminated:
            reward -= self.collision_charge()

        return self._obs(), float(reward), terminated, truncated, self._info()

    def _issue(self, i: int, cmd: int) -> None:
        f = self.sim.flights[i]
        if cmd == FL_DEC:
            f.change_altitude(-self.DALT)
        elif cmd == FL_INC:
            f.change_altitude(self.DALT)
        elif cmd in (FL_DEC2_T, FL_INC2_T):
            sign = -1 if cmd == FL_DEC2_T else 1
            f.change_altitude(sign * 2 * self.DALT)
            self.hold_left[i] = self.hold_steps
        elif cmd in (SPD_DN_T, SPD_UP_T):
            # A SYMMETRIC pulse: +dv for `hold_steps`, then -dv for `hold_steps`,
            # then nominal. Reverting to the nominal *speed* does not restore the
            # *schedule* — an aircraft that ran fast for 8 steps and then went
            # back to plan is permanently early by dv*8*DT of along-track, and
            # nothing in the action set recovers it. The equal-and-opposite half
            # is what makes the manoeuvre 4D-neutral, and folding it into the
            # same instruction is exactly the point: one clearance, zero net
            # schedule shift.
            sign = -1 if cmd == SPD_DN_T else 1
            f.change_velocity(sign * self.DVEL)
            # 2H + 1, not 2H. `_advance_holds` ticks inside this very step, so
            # a counter of 2H spends H-1 steps fast and H slow — leaving exactly
            # one dv*DT of along-track debt (70 m, or 0.298 s at 235 m/s) that
            # nothing ever pays back. The extra tick makes the halves equal and
            # the pulse genuinely schedule-neutral.
            self.pulse_left[i] = 2 * self.hold_steps + 1
            self.pulse_sign[i] = sign
        elif cmd == RESUME:
            self._restore(i)

    def _advance_holds(self) -> None:
        """Tick every temporary manoeuvre and revert the ones that have expired.

        The revert is *not* charged as a clearance. It is the second half of an
        instruction already issued and already paid for — the same contract a
        standing 'resume own navigation' has, and the reason a temporary
        manoeuvre costs one clearance instead of two.
        """
        for i in np.flatnonzero(self.hold_left > 0):
            self.hold_left[i] -= 1
            if self.hold_left[i] == 0:
                self._restore(i)
        for i in np.flatnonzero(self.pulse_left > 0):
            self.pulse_left[i] -= 1
            if self.pulse_left[i] == self.hold_steps:
                # halfway: reverse the excursion to pay back the along-track debt
                f = self.sim.flights[i]
                f.change_velocity(self.v0[i] - self.pulse_sign[i] * self.DVEL
                                  - f.velocity)
            elif self.pulse_left[i] == 0:
                self._restore(i)

    def _restore(self, i: int) -> None:
        f = self.sim.flights[i]
        f.change_altitude(self.alt0[i] - f.altitude)
        f.change_velocity(self.v0[i] - f.velocity)
        self.hold_left[i] = 0
        self.pulse_left[i] = 0
        self.pulse_sign[i] = 0

    def _count_congestion(self) -> int:
        f = self.sim.flights
        c = 0
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                if abs(f[i].altitude - f[j].altitude) >= self.ALT_MIN_SEP:
                    continue
                if math.hypot(f[i].x - f[j].x, f[i].y - f[j].y) <= self.D_SEP:
                    c += 1
        return c

    def _is_collision(self) -> bool:
        f = self.sim.flights
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                if abs(f[i].altitude - f[j].altitude) >= self.DALT:
                    continue
                if math.hypot(f[i].x - f[j].x, f[i].y - f[j].y) <= self.COLLISION_RADIUS:
                    return True
        return False

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        f = self.sim.flights
        dt, dalt = self._exit_deviation()
        conflicts = self.sim.predict_conflicts(
            max(self.max_steps - self.step_idx, 0) * self.DT) if self.step_idx < self.max_steps else []
        sev = np.zeros(self.n_flights)
        tcf = np.full(self.n_flights, np.inf)
        for c in conflicts:
            for k in (c.i, c.j):
                if c.severity > sev[k]:
                    sev[k] = c.severity
                    tcf[k] = c.t_cpa

        horizon = max(self.max_steps - self.step_idx, 1) * self.DT
        t_scale = self.hold_steps * self.DT
        rows = []
        for i, x in enumerate(f):
            ux, uy = x.get_heading_unit_vectors()
            rows.extend([
                np.clip(x.x / self.AIRSPACE, -1, 1),
                np.clip(x.y / self.AIRSPACE, -1, 1),
                2.0 * self.sim.norm_alt(x.altitude) - 1.0,
                2.0 * self.sim.norm_vel(x.velocity) - 1.0,
                ux, uy,
                float(np.tanh(dt[i] / t_scale)),
                float(np.tanh(dalt[i] / 2.0)),
                2.0 * min(self.per_aircraft_clearances[i] / 6.0, 1.0) - 1.0,
                2.0 * (max(self.hold_left[i], self.pulse_left[i])
                       / max(2 * self.hold_steps, 1)) - 1.0,
                2.0 * sev[i] - 1.0,
                2.0 * (1.0 if not np.isfinite(tcf[i]) else tcf[i] / (tcf[i] + horizon)) - 1.0,
                1.0,
            ])
        rows.extend([
            2.0 * (self.step_idx / self.max_steps) - 1.0,
            2.0 * (1.0 - self.n_clearances / self.max_clearances) - 1.0,
        ])
        obs = np.clip(np.array(rows, dtype=np.float32), -1.0, 1.0)
        return np.concatenate([obs, self.action_masks().astype(np.float32)])

    def _info(self) -> dict:
        dt, dalt = self._exit_deviation()
        return {
            "n": self.n, "m": self.m,
            "congestion_events": self.last_congestion,
            "congestion_total": self.congestion_total,
            "clearances": self.n_clearances,
            "invalids": self.n_invalids,
            "aircraft_involved": int(self.touched.sum()),
            "collided": bool(self.collided),
            "dt_exit": dt,
            "d_alt_exit": dalt,
            # The scored 4D miss: seconds late plus levels off, per aircraft.
            "exit_miss": float(np.mean(np.abs(dt) / (self.hold_steps * self.DT)
                                       + np.abs(dalt))),
        }
