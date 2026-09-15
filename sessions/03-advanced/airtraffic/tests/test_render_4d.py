"""Smoke tests for :mod:`scripts.render_4d`.

Not a pixel-level check — that would be brittle against matplotlib version
drift for no real safety gain. What matters is that a rollout records a
coherent trace and that the trace actually produces a playable video.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from render_4d import CLEARANCE_STYLE, render, rollout  # noqa: E402
from agents import Priority4DController  # noqa: E402
from envs.flight_4d import CLEARANCE_NAMES, Flight4DEnv  # noqa: E402


@pytest.mark.parametrize("spec", ["noop", "random", "rule-based"])
def test_rollout_records_one_frame_per_step_plus_the_reset(spec):
    env, frames = rollout(spec, seed=900_000, env_kwargs={})
    assert len(frames) == env.step_idx + 1
    assert frames[0][1] is None  # nothing is commanded before the first step


def test_rollout_scenario_matches_the_seed():
    """seed 900000 is documented (README, docs, the deck) as n=3 m=2 — if the
    environment's own randomisation ever changes, this is what breaks first."""
    env, _ = rollout("noop", seed=900_000, env_kwargs={})
    assert (env.n, env.m) == (3, 2)


def test_every_clearance_name_has_a_render_style():
    assert set(CLEARANCE_NAMES) <= set(CLEARANCE_STYLE)


def test_recorded_clearance_targets_a_real_aircraft():
    env, frames = rollout("rule-based", seed=900_000, env_kwargs={})
    for _, clearance, *_ in frames:
        if clearance is not None:
            i, name = clearance
            assert 0 <= i < env.n_flights
            assert name in CLEARANCE_NAMES


def test_recorded_severity_and_deviation_match_the_environment():
    """The KPI bars read severity/deviation straight off `frames`, so a
    render that agreed with the reward only approximately would be a second
    source of error, not a picture of it — pin that they are the live values,
    not something reconstructed after the fact from positions alone."""
    env, frames = rollout("rule-based", seed=900_000, env_kwargs={})
    replay = Flight4DEnv()
    obs, _ = replay.reset(seed=900_000)
    ctl = Priority4DController(hold_steps=replay.hold_steps)
    ctl.reset()

    _, _, _, severity0, deviation0, s_left0 = frames[0]
    assert severity0 == pytest.approx(replay._severity())
    np.testing.assert_allclose(deviation0, replay._deviation())
    np.testing.assert_allclose(s_left0, np.maximum(replay.s_total - replay.s_flown, 0.0))

    for k in range(1, len(frames)):
        a, _ = ctl.predict(obs)
        obs, _, term, trunc, _ = replay.step(int(a))
        _, _, _, severity_k, deviation_k, s_left_k = frames[k]
        assert severity_k == pytest.approx(replay._severity())
        np.testing.assert_allclose(deviation_k, replay._deviation())
        np.testing.assert_allclose(
            s_left_k, np.maximum(replay.s_total - replay.s_flown, 0.0))
        if term or trunc:
            break


def test_deviation_and_s_left_are_bounded_sensibly():
    env, frames = rollout("random", seed=900_004, env_kwargs={})
    for _, _, _, severity, deviation, s_left in frames:
        assert severity >= 0.0
        assert (deviation >= 0.0).all() and (deviation <= 1.0).all()
        assert (s_left >= 0.0).all()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_writes_a_playable_video(tmp_path):
    env, frames = rollout("noop", seed=900_000, env_kwargs={})
    out = tmp_path / "test.mp4"
    render(env, frames, title="test", save_path=out, fps=5)
    assert out.is_file() and out.stat().st_size > 0


# --- the per-action demos ---------------------------------------------------
#: What DEMOS documents for each single-clearance clip on DEMO_SEED at
#: n_flights=2 — (ended early, congestion_total, clearances, exit_miss). The
#: deck quotes these numbers next to the clips, so they are pinned here rather
#: than trusted to a comment.
ACTION_DEMO_OUTCOMES = {
    "noop":      (True, 3, 0, 0.0),
    "fl_inc":    (False, 5, 1, 0.5),
    "fl_dec":    (False, 5, 1, 0.5),
    "fl_inc2_t": (False, 2, 1, 0.0),
    "fl_dec2_t": (False, 2, 1, 0.0),
    "spd_dn_t":  (False, 5, 1, 0.0),
    "spd_up_t":  (False, 4, 1, 0.0),
    "resume":    (False, 5, 2, 0.0),
}


@pytest.mark.parametrize("spec,expected", ACTION_DEMO_OUTCOMES.items())
def test_action_demo_outcome_is_what_the_deck_quotes(spec, expected):
    from render_4d import DEMO_SEED
    collided, cong, clearances, exit_miss = expected
    env, frames = rollout(spec, seed=DEMO_SEED,
                          env_kwargs=dict(n_flights=2, n_range=(2, 2)))
    assert (env.n, env.m) == (2, 0)
    assert env.collided is collided
    assert env.congestion_total == cong
    assert env.n_clearances == clearances
    assert env.n_invalids == 0, "a demo must never issue a masked-out clearance"
    assert env._info()["exit_miss"] == pytest.approx(exit_miss, abs=1e-3)


def test_two_aircraft_sector_is_the_same_pair_as_the_handbook_demo():
    """The handbook's three-resolution demos run at n_flights=5 (n=2, m=3);
    the per-action clips at n_flights=2 (n=2, m=0). Both are DEMO_SEED, and the
    converging pair is drawn before the background traffic, so the two clips
    show the *same* encounter with and without bystanders. Pin that."""
    from render_4d import DEMO_SEED
    big, _ = rollout("noop", seed=DEMO_SEED, env_kwargs=dict(n_range=(2, 2)))
    small, _ = rollout("noop", seed=DEMO_SEED, env_kwargs=dict(n_flights=2, n_range=(2, 2)))
    assert (big.n, big.m) == (2, 3) and (small.n, small.m) == (2, 0)
    for i in range(2):
        a, b = big.sim.flights[i], small.sim.flights[i]
        assert (a.x, a.y, a.altitude, a.velocity, a.heading) == \
               (b.x, b.y, b.altitude, b.velocity, b.heading)
    assert big.step_idx == small.step_idx == 18
