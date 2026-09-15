"""Contract tests for the 4D environment and its autoregressive policy.

    conda activate rlbootcamp
    cd sessions/03-advanced/airtraffic && pytest tests/test_flight_4d.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# tests/conftest.py already puts the package root on sys.path.
from envs.flight_4d import (Flight4DEnv, N_CLEARANCES, N_FEATS,      # noqa: E402
                            N_GLOBALS, FL_INC2_T, SPD_UP_T, RESUME)
from policies.autoregressive import (AutoregressiveClearance,        # noqa: E402
                                     MaskInObsWrapper)

SEEDS = range(12)


# ==========================================================================
# the distribution
# ==========================================================================
@pytest.mark.parametrize("seed", range(6))
def test_autoregressive_matches_flat_categorical(seed):
    """The chain rule, asserted.

    Sampling ``i`` then ``c|i`` must give the same distribution as one flat
    Categorical over ``log p(i) + log p(c|i)``. If this drifts, log_prob and
    entropy are wrong and PPO silently optimises the wrong objective.
    """
    g = torch.Generator().manual_seed(seed)
    B, F, C = 7, 5, N_CLEARANCES
    s1 = torch.randn(B, 1 + F, generator=g)
    s2 = torch.randn(B, F, C, generator=g)
    # mask a few rows out entirely, as the env does for busy aircraft
    legal = torch.rand(B, F, C, generator=g) > 0.3
    legal[:, 0, :] = False
    s2 = s2.masked_fill(~legal, -1e8)
    s1[:, 1:] = s1[:, 1:].masked_fill(~legal.any(-1), -1e8)

    dist = AutoregressiveClearance(F, C).proba_distribution(s1, s2)

    lp1 = torch.log_softmax(s1, -1)
    lp2 = torch.log_softmax(s2, -1)
    flat = torch.cat([lp1[:, :1],
                      (lp1[:, 1:].unsqueeze(-1) + lp2).reshape(B, -1)], dim=1)
    ref = torch.distributions.Categorical(logits=flat)

    torch.testing.assert_close(dist.entropy(), ref.entropy(), atol=1e-5, rtol=1e-4)
    for a in range(1 + C * F):
        acts = torch.full((B,), a, dtype=torch.long)
        torch.testing.assert_close(dist.log_prob(acts), ref.log_prob(acts),
                                   atol=1e-5, rtol=1e-4)
    # a proper distribution sums to one
    torch.testing.assert_close(flat.exp().sum(-1), torch.ones(B), atol=1e-5, rtol=1e-4)


def test_sampling_never_returns_a_masked_action():
    torch.manual_seed(0)
    B, F, C = 64, 5, N_CLEARANCES
    s1 = torch.randn(B, 1 + F)
    s2 = torch.randn(B, F, C)
    legal = torch.rand(B, F, C) > 0.5
    s2 = s2.masked_fill(~legal, -1e8)
    s1[:, 1:] = s1[:, 1:].masked_fill(~legal.any(-1), -1e8)
    dist = AutoregressiveClearance(F, C).proba_distribution(s1, s2)
    for _ in range(20):
        a = dist.sample()
        for b, act in enumerate(a.tolist()):
            if act == 0:
                continue
            i, c = divmod(act - 1, C)
            assert legal[b, i, c], "sampled a masked clearance"


# ==========================================================================
# the environment
# ==========================================================================
@pytest.mark.parametrize("seed", SEEDS)
def test_scenario_shape(seed):
    env = Flight4DEnv()
    obs, info = env.reset(seed=seed)
    assert obs.shape == (N_FEATS * 5 + N_GLOBALS + env.n_actions,)
    assert env.observation_space.contains(obs)
    assert info["n"] + info["m"] == 5
    assert 2 <= info["n"] <= 4, "n must be drawn from {2,3,4}"


def test_n_actually_varies():
    env = Flight4DEnv()
    seen = {env.reset(seed=s)[1]["n"] for s in range(40)}
    assert seen == {2, 3, 4}, f"n should cover 2..4, saw {seen}"


@pytest.mark.parametrize("seed", SEEDS)
def test_masked_actions_are_never_executed(seed):
    """A masked policy should make this unreachable; assert the env agrees anyway."""
    env = Flight4DEnv()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(env.max_steps):
        mask = env.action_masks()
        assert mask[0], "idle must always be legal"
        legal = np.flatnonzero(mask)
        a = int(rng.choice(legal))
        _, _, term, trunc, info = env.step(a)
        assert info["invalids"] == 0, "a legal action was charged as invalid"
        if term or trunc:
            break


@pytest.mark.parametrize("seed", SEEDS)
def test_temporary_manoeuvre_reverts_and_costs_one_clearance(seed):
    """The whole design rests on this: one instruction, out and back."""
    env = Flight4DEnv()
    env.reset(seed=seed)
    i = next((k for k in range(env.n_flights)
              if env.action_masks()[1 + N_CLEARANCES * k + FL_INC2_T]), None)
    if i is None:
        pytest.skip("no aircraft with two levels of headroom on this seed")

    alt0 = env.sim.flights[i].altitude
    env.step(1 + N_CLEARANCES * i + FL_INC2_T)
    assert env.sim.flights[i].altitude == pytest.approx(alt0 + 2 * env.DALT)
    assert env.n_clearances == 1

    for _ in range(env.hold_steps + 1):
        if env.step_idx >= env.max_steps:
            pytest.skip("episode ended before the hold expired")
        env.step(0)
        if env.hold_left[i] == 0:
            break
    assert env.sim.flights[i].altitude == pytest.approx(alt0), "did not revert"
    assert env.n_clearances == 1, "the revert must not be charged again"


@pytest.mark.parametrize("seed", SEEDS)
def test_speed_pulse_is_schedule_neutral(seed):
    """The property the whole design turns on.

    Reverting to the nominal *speed* does not restore the *schedule*: an
    aircraft that ran 1 dv fast for 8 steps and then went back to plan is
    permanently early by ``dv * 8 * DT`` of along-track, and no action recovers
    it. So the pulse is symmetric — ``+dv`` for ``hold_steps``, ``-dv`` for
    ``hold_steps``, then nominal — and the assertion is not "speed came back"
    but **"the 4D slot came back"**.
    """
    env = Flight4DEnv()
    env.reset(seed=seed)
    i = next((k for k in range(env.n_flights)
              if env.action_masks()[1 + N_CLEARANCES * k + SPD_UP_T]), None)
    if i is None:
        pytest.skip("no aircraft may speed up on this seed")

    v0 = env.sim.flights[i].velocity
    env.step(1 + N_CLEARANCES * i + SPD_UP_T)
    assert env.sim.flights[i].velocity == pytest.approx(v0 + env.DVEL)
    # `_advance_holds` runs inside the same step that issued the pulse, so the
    # counter has already ticked once. Assert the behaviour, not the bookkeeping.
    assert env.pulse_left[i] >= 2 * env.hold_steps - 1

    saw_slow_half = False
    for _ in range(2 * env.hold_steps + 2):
        if env.step_idx >= env.max_steps:
            pytest.skip("episode ended before the pulse completed")
        _, _, term, _, info = env.step(0)
        if term:
            pytest.skip("mid-air ended the episode before the pulse completed")
        if env.sim.flights[i].velocity < v0 - 1e-9:
            saw_slow_half = True
        if env.pulse_left[i] == 0:
            break

    assert saw_slow_half, "the pulse never paid back its along-track debt"
    assert env.sim.flights[i].velocity == pytest.approx(v0)
    assert env.n_clearances == 1, "the pulse must cost ONE clearance, not two"
    # and the point of it all: no net schedule shift
    assert info["dt_exit"][i] == pytest.approx(0.0, abs=1e-6), \
        "a symmetric pulse must leave the 4D slot intact"


@pytest.mark.parametrize("seed", SEEDS)
def test_speed_clearances_are_usually_available(seed):
    """Regression guard on the dead-end mask.

    It once computed the remaining time from ``max_steps`` rather than from the
    aircraft's own exit schedule, which left speed clearances legal on 13-16% of
    (aircraft, step) pairs and silently deleted the velocity axis. A mask that
    rejects nearly everything is indistinguishable from a missing action.
    """
    env = Flight4DEnv()
    env.reset(seed=seed)
    legal = total = 0
    for _ in range(env.max_steps):
        m = env.action_masks()
        for i in range(env.n_flights):
            if env.hold_left[i] or env.pulse_left[i]:
                continue
            legal += int(m[1 + N_CLEARANCES * i + SPD_UP_T])
            total += 1
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    assert total > 0
    # It cannot be near 100%, and should not be: the pulse runs for
    # `2 * hold_steps` and an aircraft with less scheduled time than that left
    # genuinely cannot fly one without missing its gate. That is a real
    # kinematic dead end, correctly refused. The guard is against the
    # *regression* — 13-16%, where the axis was effectively gone.
    assert legal / total > 0.25, (
        f"SPD_UP_T legal on only {100*legal/total:.0f}% of idle aircraft-steps")


@pytest.mark.parametrize("seed", SEEDS)
def test_reward_is_exactly_three_terms(seed):
    """Safety + timeliness + action cost, and nothing else.

    Recomputed from the environment's own accessors and compared to what `step`
    returned. If a fourth term ever creeps in, this fails rather than quietly
    changing what the agent optimises.
    """
    env = Flight4DEnv()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(env.max_steps):
        legal = np.flatnonzero(env.action_masks())
        a = int(rng.choice(legal))
        before = env.n_clearances
        _, r, term, trunc, _ = env.step(a)
        issued = int(env.n_clearances > before)
        expected = -(env.w_safe * env._severity()
                     + (env.w_safe / env.n_flights) * float(env._deviation().sum())
                     + env.w_clearance * issued)
        if term:
            expected -= env.collision_charge()
        assert r == pytest.approx(expected, abs=1e-9)
        if term or trunc:
            break


@pytest.mark.parametrize("seed", SEEDS)
def test_whole_fleet_at_max_deviation_costs_one_infringement(seed):
    """The scaling rule, asserted rather than asserted-in-a-comment.

    Every aircraft as deviated as it is possible to be must cost exactly what one
    severity-1 infringement costs. That is what makes the priority between the
    two objectives checkable instead of a matter of taste.
    """
    env = Flight4DEnv()
    env.reset(seed=seed)
    per_aircraft = env.w_safe / env.n_flights
    assert per_aircraft * env.n_flights == pytest.approx(env.w_safe)
    # and a deviation is bounded at 1, so the bill cannot exceed it
    dev = env._deviation()
    assert dev.min() >= 0.0 and dev.max() <= 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_crash_charge_covers_the_steps_not_flown(seed):
    """The charge is there to regularise the return, and must cover the tail.

    An episode ending at step 10 banks ten steps of cost; one running to 50 banks
    fifty. Charging the steps not flown completes every episode to the same
    horizon, so returns are comparable and the critic is not also fitting a
    length-driven spread.

    Checked per state, because the charge now varies with the step: at every k,
    crashing must cost at least as much as flying on would have.
    """
    env = Flight4DEnv()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rewards, charges = [], []
    while True:
        legal = np.flatnonzero(env.action_masks())
        charges.append(env.collision_charge())     # charge if we crashed HERE
        _, r, term, trunc, _ = env.step(int(rng.choice(legal)))
        # Strip the terminal charge back out: the comparison is against the cost
        # of a SURVIVING trajectory, or it is circular.
        rewards.append(r + env.collision_charge() if term else r)
        if term or trunc:
            break

    g = env.gamma
    for k in range(len(rewards)):
        remaining = -sum(g ** (j - k) * rewards[j] for j in range(k, len(rewards)))
        assert charges[k] >= remaining - 1e-9, (
            f"at step {k}: crashing costs {charges[k]:.1f} but flying on costs "
            f"{remaining:.1f} — crashing is the cheap way out")


@pytest.mark.parametrize("seed", SEEDS)
def test_crash_charge_falls_to_zero_at_the_horizon(seed):
    """Proportional to the steps forfeited, so a crash at the last step is free.

    That is the point: at step 50 there is nothing left to forfeit, so there is
    nothing to charge. A flat charge instead made an early mid-air too cheap and
    a late one too expensive — the length-dependent spread this removes.
    """
    env = Flight4DEnv()
    env.reset(seed=seed)
    first = env.collision_charge()
    assert first == pytest.approx(env.collision_rate * env.w_safe * env.max_steps)
    env.step_idx = env.max_steps
    assert env.collision_charge() == pytest.approx(0.0)
    env.step_idx = env.max_steps // 2
    assert env.collision_charge() == pytest.approx(first / 2, rel=1e-6)


def test_one_flight_level_earns_the_severity_it_should():
    """39% for the first level, the remaining 61% for the second.

    Not a chosen exponent: `predict_conflicts` omits any pair at or beyond
    `ALT_MIN_SEP` entirely, and within it severity carries
    `exp(-d_alt/ALT_MIN_SEP)`. So the credit for a partial climb falls out of the
    simulator's own conflict definition.
    """
    import math
    env = Flight4DEnv()
    one = math.exp(-env.DALT / env.ALT_MIN_SEP)
    assert one == pytest.approx(0.6065, abs=1e-3), "first level should leave ~61%"
    assert 1 - one == pytest.approx(0.3935, abs=1e-3), "first level removes ~39%"
    # the second level removes the rest by deleting the conflict outright
    assert 2 * env.DALT >= env.ALT_MIN_SEP


def test_mask_is_carried_in_the_observation(_seed=0):
    """The policy must never have to re-derive legality from normalised features."""
    env = Flight4DEnv()
    obs, _ = env.reset(seed=_seed)
    assert obs.shape[0] == N_FEATS * 5 + N_GLOBALS + env.n_actions
    mask = obs[env.mask_offset:]
    assert set(np.unique(mask)) <= {0.0, 1.0}
    assert mask[0] == 1.0, "idle is always legal"
    np.testing.assert_array_equal(mask.astype(bool), env.action_masks())
    for _ in range(10):
        obs, _, t, tr, _ = env.step(0)
        np.testing.assert_array_equal(obs[env.mask_offset:].astype(bool),
                                      env.action_masks())
        if t or tr:
            break


@pytest.mark.parametrize("seed", SEEDS)
def test_untouched_aircraft_have_zero_exit_deviation(seed):
    """The 4D potential must cost nothing for traffic you never command."""
    env = Flight4DEnv()
    env.reset(seed=seed)
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(0)
        if term or trunc:
            break
    assert np.allclose(info["dt_exit"], 0.0, atol=1e-6)
    assert np.allclose(info["d_alt_exit"], 0.0, atol=1e-9)
    assert info["exit_miss"] == pytest.approx(0.0, abs=1e-6)


def test_mask_wrapper_is_now_a_noop():
    """The wrapper used to append the mask; the environment does it itself now.

    It must not append a second copy — that would shift every index the policy
    reads and fail as a wrong answer rather than an error.
    """
    env = Flight4DEnv()
    wrapped = MaskInObsWrapper(env)
    assert wrapped.observation_space.shape == env.observation_space.shape


def test_figure_constants_match_env():
    """`make_figures_4d.py` mirrors env constants; catch them drifting apart.

    The figure script deliberately does not import this package -- it runs from
    the repo root against committed CSVs and must work without the airtraffic
    package importable. The price of that is a copy of four constants, and the
    copy is only safe if something fails when it goes stale.
    """
    import importlib.util

    script = (Path(__file__).resolve().parents[4]
              / "slides/scripts/make_figures_4d.py")
    if not script.is_file():
        pytest.skip("slides/ not present in this export")

    spec = importlib.util.spec_from_file_location("_mk4d", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    env = Flight4DEnv()
    env.reset(seed=0)          # `sim` is built on reset, not in __init__
    assert mod.ALT_MIN_SEP == env.sim.ALT_MIN_SEP
    assert mod.DALT == env.sim.DALT
    assert mod.W_SAFE == env.w_safe
    assert mod.W_CLEARANCE == env.w_clearance
    assert mod.COLLISION_RATE == env.collision_rate
    assert mod.MAX_STEPS == env.max_steps
    assert mod.R_HOTSPOT == env.sim.hotspot_limit
