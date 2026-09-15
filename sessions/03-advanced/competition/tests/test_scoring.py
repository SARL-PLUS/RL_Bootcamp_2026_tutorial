"""Contract for the competition scoring core.

The score decides trophies, so the properties that matter are the ones that stop
it being gamed, not the ones that make it pretty.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]      # sessions/03-advanced
sys.path.insert(0, str(ROOT))

from competition.config import SCENARIO, EXIT_MISS_BUCKET
from competition.score import EpisodeKPIs, aggregate, rank
from competition.seeds import derive_seeds
from competition.submission import (
    max_action, parse_row_id, read_submission, row_id, write_submission,
)

N_FLIGHTS = SCENARIO.n_flights
MAX_ACTION = max_action(N_FLIGHTS)


def kpis(seed=0, completed=True, congestion=0, exit_miss=0.0, clearances=0,
         involved=0, invalids=0, steps=SCENARIO.max_steps):
    return EpisodeKPIs(seed=seed, steps=steps, completed=completed,
                       congestion=congestion, exit_miss=exit_miss,
                       clearances=clearances, aircraft_involved=involved,
                       invalids=invalids, max_exit_miss=exit_miss)


# ---------------------------------------------------------------- the scenario

def test_the_frozen_scenario_is_the_environments_own_vocabulary():
    """``as_env_kwargs`` is passed straight to the constructor, so a key the
    environment does not take is a crash and a key it silently ignores is worse:
    the board would be ranked on a scenario nobody configured."""
    import inspect
    from competition.rollout import make_env
    from envs.flight_4d import Flight4DEnv

    accepted = set(inspect.signature(Flight4DEnv.__init__).parameters)
    assert set(SCENARIO.as_env_kwargs()) <= accepted

    env = make_env()
    assert env.n_flights == SCENARIO.n_flights
    assert (env.n_lo, env.n_hi) == SCENARIO.n_range
    assert env.max_steps == SCENARIO.max_steps
    assert env.hold_steps == SCENARIO.hold_steps
    # the unit `exit_miss` is measured in, restated in config and asserted here
    assert SCENARIO.exit_time_scale == env.hold_steps * env.DT


# ---------------------------------------------------------------- ranking

def test_safety_outranks_everything():
    safe = aggregate([kpis(congestion=1, exit_miss=9.0, clearances=9999, involved=99)])
    fast = aggregate([kpis(congestion=2, exit_miss=0.0, clearances=0, involved=0)])
    ordered = [name for _, name, _ in rank({"safe": safe, "fast": fast})]
    assert ordered == ["safe", "fast"]


def test_timeliness_breaks_a_safety_tie_before_efficiency():
    prompt = aggregate([kpis(congestion=5, exit_miss=0.01, clearances=500, involved=50)])
    cheap = aggregate([kpis(congestion=5, exit_miss=1.00, clearances=1, involved=1)])
    ordered = [name for _, name, _ in rank({"prompt": prompt, "cheap": cheap})]
    assert ordered == ["prompt", "cheap"]


def test_efficiency_breaks_a_timeliness_tie():
    lean = aggregate([kpis(congestion=5, exit_miss=0.10, clearances=10, involved=2)])
    busy = aggregate([kpis(congestion=5, exit_miss=0.10, clearances=90, involved=9)])
    ordered = [name for _, name, _ in rank({"lean": lean, "busy": busy})]
    assert ordered == ["lean", "busy"]


def test_exit_miss_is_bucketed_so_efficiency_is_not_dead_weight():
    """Floats never tie exactly, so without bucketing the third objective could
    never be reached and 'efficiency' would be decoration."""
    a = aggregate([kpis(congestion=5, exit_miss=0.100, clearances=90)])
    b = aggregate([kpis(congestion=5, exit_miss=0.100 + EXIT_MISS_BUCKET / 10,
                        clearances=10)])
    assert a.rank_key[2] == b.rank_key[2]          # same bucket
    ordered = [name for _, name, _ in rank({"a": a, "b": b})]
    assert ordered == ["b", "a"]                    # so clearances decide


def test_dying_early_cannot_buy_a_safety_win():
    """A short episode accumulates less congestion purely by existing for less
    time. If failure were not ranked first, crashing out would be a strategy."""
    quitter = aggregate([kpis(completed=False, steps=5, congestion=0)])
    finisher = aggregate([kpis(completed=True, congestion=40)])
    assert quitter.congestion_total < finisher.congestion_total
    ordered = [name for _, name, _ in rank({"quitter": quitter, "finisher": finisher})]
    assert ordered == ["finisher", "quitter"]


def test_zero_conflict_episodes_are_counted_for_the_trophy():
    score = aggregate([kpis(congestion=0), kpis(congestion=3), kpis(congestion=0)])
    assert score.zero_conflict_episodes == 2


def test_aggregate_rejects_an_empty_run():
    with pytest.raises(ValueError, match="empty"):
        aggregate([])


# ---------------------------------------------------------------- seeds

def test_seed_derivation_is_deterministic_and_distinct():
    a = derive_seeds("salt", 50)
    assert a == derive_seeds("salt", 50)
    assert len(set(a)) == 50
    assert a != derive_seeds("other-salt", 50)
    assert all(0 <= s < 2 ** 31 - 1 for s in a)


# ---------------------------------------------------------------- submission IO

def test_row_id_round_trips():
    assert parse_row_id(row_id(1829385696, 7)) == (1829385696, 7)


def test_submission_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    traces = {11: rng.integers(0, MAX_ACTION + 1, size=SCENARIO.max_steps),
              22: rng.integers(0, MAX_ACTION + 1, size=SCENARIO.max_steps)}
    path = tmp_path / "sub.csv"
    rows = write_submission(path, traces)
    assert rows == 2 * SCENARIO.max_steps

    back = read_submission(path, N_FLIGHTS, SCENARIO.max_steps)
    assert set(back) == set(traces)
    for seed in traces:
        np.testing.assert_array_equal(back[seed], traces[seed])


def test_one_row_per_step_not_per_aircraft(tmp_path):
    """At most one aircraft is commanded per step, so a per-aircraft grid would
    imply the sector can be re-tasked all at once. It cannot."""
    path = tmp_path / "sub.csv"
    write_submission(path, {7: np.zeros(SCENARIO.max_steps, dtype=np.int64)})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1 + SCENARIO.max_steps
    assert lines[1].split(",")[0] == "s0000007_t00"


def test_short_trace_is_padded_not_rejected(tmp_path):
    """A file that stops early — hand-written, or from another harness — is read,
    not rejected. The replay holds the sector with idle actions rather than
    scoring less exposure. ``steps=None`` is what lets the writer produce one."""
    path = tmp_path / "sub.csv"
    write_submission(path, {7: np.ones(3, dtype=np.int64)}, steps=None)
    back = read_submission(path, N_FLIGHTS, SCENARIO.max_steps)
    assert back[7].shape == (3,)


def test_early_ending_episodes_still_cover_the_horizon(tmp_path):
    """Kaggle's metric requires every (seed, step) row the solution exports, and
    every real agent ends at least one episode early. Padding here is what keeps
    a submission from being refused outright."""
    path = tmp_path / "sub.csv"
    rows = write_submission(path, {7: np.ones(3, dtype=np.int64)})
    assert rows == SCENARIO.max_steps

    back = read_submission(path, N_FLIGHTS, SCENARIO.max_steps)
    assert back[7].shape == (SCENARIO.max_steps,)
    np.testing.assert_array_equal(back[7][:3], 1)
    np.testing.assert_array_equal(back[7][3:], 0)


def test_write_rejects_a_trace_longer_than_the_horizon(tmp_path):
    with pytest.raises(ValueError, match="horizon"):
        write_submission(tmp_path / "sub.csv",
                         {7: np.zeros(SCENARIO.max_steps + 1, dtype=np.int64)})


@pytest.mark.parametrize("bad,match", [
    ("id,action\nnonsense,0\n", "malformed"),
    ("id,action\ns0000007_t00_f0,0\n", "malformed"),
    (f"id,action\ns0000007_t00,{MAX_ACTION + 1}\n", "outside"),
    ("id,action\ns0000007_t99,0\n", "step"),
    ("id,action\ns0000007_t00,notanint\n", "not an integer"),
    ("wrong,header\na,1\n", "expected header"),
])
def test_malformed_submissions_are_rejected(tmp_path, bad, match):
    """Silence here would turn a broken file into a quietly bad score."""
    path = tmp_path / "bad.csv"
    path.write_text(bad)
    with pytest.raises(ValueError, match=match):
        read_submission(path, N_FLIGHTS, SCENARIO.max_steps)


# ---------------------------------------------------------------- replay

def test_replay_reproduces_the_rollout_score():
    """The public leaderboard is only honest if re-flying a submitted trace gives
    the same KPIs as the run that produced it."""
    from competition.agents import load_agent
    from competition.rollout import replay_actions, rollout_agent

    agent = load_agent("rule-based")
    for seed in (101, 202):
        original, actions = rollout_agent(agent, seed)
        replayed = replay_actions(actions, seed)
        assert replayed.congestion == original.congestion
        assert replayed.clearances == original.clearances
        assert replayed.aircraft_involved == original.aircraft_involved
        assert replayed.completed == original.completed
        assert replayed.exit_miss == pytest.approx(original.exit_miss, abs=1e-12)


def test_replay_path_does_not_import_torch():
    """Kaggle's metric sandbox has no torch. If the replay drags it in through an
    innocent-looking import, the public leaderboard stops working."""
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(ROOT)!r});"
        "import numpy as np;"
        "from competition.rollout import replay_actions;"
        "from competition.submission import read_submission;"
        "k = replay_actions(np.zeros(5, dtype=np.int64), 3);"
        "assert k.clearances == 0;"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("False"), "replay pulled torch into the sandbox"


def test_an_sb3_policy_is_scorable_and_its_trace_replays():
    """A checkpoint is one of the three things a submission can be, so the
    harness has to take one end to end: roll it out, record the flat action
    index it chose, and re-fly that trace to the same KPIs.

    The trace is the raw ``Discrete`` index the environment was handed. There is
    nothing to decode any more — one action space, one integer per step — which
    is the whole reason the old ``action_mode`` dispatch could be deleted.
    """
    from stable_baselines3 import PPO

    from competition import _paths  # noqa: F401  (side effect: sys.path)
    from envs.flight_4d import N_CLEARANCES, N_FEATS, N_GLOBALS
    from policies import AutoregressivePolicy
    from competition.rollout import make_env, replay_actions, rollout_agent

    env = make_env()
    model = PPO(AutoregressivePolicy, env, device="cpu", seed=0,
                policy_kwargs=dict(n_flights=env.n_flights, n_feats=N_FEATS,
                                   n_globals=N_GLOBALS,
                                   n_clearances=N_CLEARANCES))

    original, actions = rollout_agent(model, seed=1234, deterministic=True)
    assert actions.ndim == 1, "one decision per step, not one per aircraft"
    assert actions.min() >= 0 and actions.max() <= MAX_ACTION

    replayed = replay_actions(actions, seed=1234)
    assert replayed.congestion == original.congestion
    assert replayed.clearances == original.clearances
    assert replayed.exit_miss == pytest.approx(original.exit_miss, abs=1e-12)


# ------------------------------------------------------------ load_agent

def test_random_agent_is_legal_and_reproducible():
    """`random` is the action space's own score: uniform over the mask, never
    an invalid action, and the same scenario always draws the same actions —
    so its row on the board can be quoted."""
    from competition.agents import load_agent
    from competition.rollout import rollout_agent

    a, b = load_agent("random"), load_agent("random")
    kpis_a, actions_a = rollout_agent(a, seed=11)
    kpis_b, actions_b = rollout_agent(b, seed=11)
    assert (actions_a == actions_b).all()
    assert kpis_a == kpis_b
    assert kpis_a.invalids == 0
    assert kpis_a.clearances > 0                    # it does act
    # a different scenario draws a different stream
    _, actions_c = rollout_agent(load_agent("random"), seed=12)
    assert len(actions_c) != len(actions_a) or not (actions_c == actions_a).all()


def test_unknown_agent_name_is_a_clear_error():
    from competition.agents import load_agent
    with pytest.raises(SystemExit, match="unknown agent 'randmo'.*noop, random, rule-based"):
        load_agent("randmo")
