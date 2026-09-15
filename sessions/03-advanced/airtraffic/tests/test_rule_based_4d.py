"""Contract for the Flight4DEnv hand-written baseline.

A baseline nobody checks is worse than no baseline: every number an agent is
compared against inherits its bugs. Two of the three bugs found while writing
this controller were silent — it kept flying, it just flew badly — and both are
pinned here.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import Priority4DController                     # noqa: E402
from agents.rule_based_4d import decode                     # noqa: E402
from envs.flight_4d import Flight4DEnv, N_CLEARANCES        # noqa: E402
from envs.simulator import CollisionCourseSimulator as Sim  # noqa: E402

SEEDS = range(900_000, 900_012)


@pytest.mark.parametrize("seed", SEEDS)
def test_decode_inverts_the_observation(seed):
    """The controller sees the observation, so it must read it exactly.

    `severity` and `t_conflict` go through squashing functions on the way out;
    a controller that approximates its own inputs is a second source of error,
    not a baseline.
    """
    env = Flight4DEnv()
    obs, _ = env.reset(seed=seed)
    for _ in range(6):
        d = decode(obs, env.n_flights, env.max_steps, env.hold_steps)
        horizon = max(env.max_steps - env.step_idx, 0) * env.DT
        conflicts = env.sim.predict_conflicts(horizon) if horizon > 0 else []
        sev = np.zeros(env.n_flights)
        for c in conflicts:
            for k in (c.i, c.j):
                sev[k] = max(sev[k], c.severity)
        np.testing.assert_allclose(d["severity"], sev, atol=1e-3)

        true_level = [(f.altitude - Sim.ALT_MIN) / Sim.DALT for f in env.sim.flights]
        np.testing.assert_allclose(d["level"], true_level, atol=1e-3)

        obs, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break


@pytest.mark.parametrize("seed", SEEDS)
def test_never_emits_a_masked_action(seed):
    """The environment counts an illegal clearance as an invalid. A baseline
    that racks them up is measuring its own sloppiness, not the problem."""
    env = Flight4DEnv()
    ctl = Priority4DController(hold_steps=env.hold_steps)
    obs, _ = env.reset(seed=seed)
    ctl.reset()
    while True:
        a, _ = ctl.predict(obs)
        assert env.action_masks()[int(a)], f"issued masked action {int(a)}"
        obs, _, term, trunc, info = env.step(int(a))
        if term or trunc:
            break
    assert info["invalids"] == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_assigned_slots_are_distinct(seed):
    """Every aircraft in the conflict gets a level of its own.

    The first version added a fixed offset to each aircraft's own level and
    clipped to the band — which near a rail collapses +2 and +4 onto the same
    slot and assigns two aircraft the same place to stand. It failed silently:
    the controller flew both aircraft exactly where it had told them to go, and
    they collided there. 8 of 10 remaining failures, until this was found.
    """
    env = Flight4DEnv()
    ctl = Priority4DController(hold_steps=env.hold_steps)
    obs, _ = env.reset(seed=seed)
    ctl.reset()
    ctl.predict(obs)

    d = decode(obs, env.n_flights, env.max_steps, env.hold_steps)
    involved = [i for i in range(env.n_flights)
                if d["severity"][i] >= ctl.severity_gate]
    if len(involved) < 2:
        pytest.skip("no multi-aircraft conflict on this seed")
    targets = [ctl._target[i] for i in involved]
    assert len(set(targets)) == len(targets), \
        f"two aircraft assigned the same slot: {targets}"
    for a in range(len(targets)):
        for b in range(a + 1, len(targets)):
            assert abs(targets[a] - targets[b]) >= ctl.SEP_LEVELS, \
                f"slots {targets[a]} and {targets[b]} are inside ALT_MIN_SEP"


@pytest.mark.parametrize("seed", SEEDS)
def test_recovery_is_not_undone(seed):
    """Sending an aircraft home must retire its slot.

    Otherwise the steering phase sees a gap the instant the aircraft is back on
    plan and flies it straight out again — the controller undoing its own
    recovery, one clearance at a time. It showed up as an `exit_miss` that would
    not move however the recovery gate was tuned.
    """
    env = Flight4DEnv()
    ctl = Priority4DController(hold_steps=env.hold_steps)
    obs, _ = env.reset(seed=seed)
    ctl.reset()
    RESUME_CMD = 6
    resumed = {}
    while True:
        a, _ = ctl.predict(obs)
        if int(a) > 0:
            i, cmd = divmod(int(a) - 1, N_CLEARANCES)
            if cmd == RESUME_CMD:
                d = decode(obs, env.n_flights, env.max_steps, env.hold_steps)
                home = int(np.rint(d["level"][i] - d["d_alt"][i]))
                assert ctl._target[i] == home, \
                    "slot not retired on RESUME; steering will undo it"
                resumed[i] = home
            elif i in resumed:
                assert False, f"aircraft {i} re-commanded after coming home"
        obs, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            break


def test_recovery_order_follows_assignment_not_flight_index():
    """With several aircraft simultaneously clear, RESUME goes out in
    assignment order (earliest `t_conflict` at reset), not ascending flight
    index. Only one clearance can be issued per step, so *some* order has to
    break the tie — flight index is arbitrary with respect to the problem;
    the priority the controller already computed is not.
    """
    n = 5
    ctl = Priority4DController(n_flights=n, hold_steps=5)
    ctl.reset()
    # Flight 3 was displaced before flight 1 (e.g. its conflict was sooner),
    # so it must come home first even though its index is larger.
    ctl._order = [3, 1]
    ctl._target = np.zeros(n, dtype=int)

    # Both off-plan, and far enough apart from everything (including each
    # other) to satisfy the recovery-separation gate simultaneously.
    d = dict(
        level=np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        d_alt=np.array([0.0, 4.0, 0.0, 4.0, 0.0]),
        x=np.array([0.0, 50_000.0, 0.0, -50_000.0, 0.0]),
        y=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    mask = np.ones(1 + N_CLEARANCES * n, dtype=bool)

    a = ctl._recover(d, mask)
    i, cmd = divmod(int(a) - 1, N_CLEARANCES)
    assert i == 3, f"expected flight 3 (assigned first) to recover first, got {i}"
    assert cmd == 6  # RESUME

    # Simulate the next observation: RESUME reads back at the entry level in
    # one instruction, so flight 3's deviation is gone; flight 1 has not moved.
    d["d_alt"][3] = 0.0
    a2 = ctl._recover(d, mask)
    i2, cmd2 = divmod(int(a2) - 1, N_CLEARANCES)
    assert i2 == 1, f"expected flight 1 next, got {i2}"


def test_beats_a_random_legal_policy_on_every_objective():
    """The bar has to actually be a bar.

    Not a tight threshold — a loose one that fails loudly if the controller
    regresses into something a coin flip could match.
    """
    seeds = list(range(900_000, 900_030))
    out = {}
    for who in ("rule-based", "random"):
        env = Flight4DEnv()
        ctl = Priority4DController(hold_steps=env.hold_steps)
        failed = cong = clr = 0
        miss = 0.0
        for s in seeds:
            obs, _ = env.reset(seed=s)
            ctl.reset()
            rng = np.random.default_rng(s)
            while True:
                a = (ctl.predict(obs)[0] if who == "rule-based"
                     else rng.choice(np.flatnonzero(env.action_masks())))
                obs, _, term, trunc, info = env.step(int(a))
                if term or trunc:
                    break
            failed += int(info["collided"])
            cong += info["congestion_total"]
            clr += info["clearances"]
            miss += info["exit_miss"]
        out[who] = (failed, cong, miss / len(seeds), clr)

    rb, rnd = out["rule-based"], out["random"]
    assert rb[0] < rnd[0], f"failures {rb[0]} vs random {rnd[0]}"
    assert rb[1] < rnd[1], f"congestion {rb[1]} vs random {rnd[1]}"
    assert rb[2] < rnd[2], f"exit_miss {rb[2]:.3f} vs random {rnd[2]:.3f}"
    assert rb[3] < rnd[3], f"clearances {rb[3]} vs random {rnd[3]}"
