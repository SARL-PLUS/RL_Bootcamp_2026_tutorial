"""The Kaggle metric is a second implementation of the rules. These tests are the
only thing stopping it drifting from the first one.

It exists separately because Kaggle's metric sandbox has no gymnasium, no
matplotlib and no bootcamp package — so it cannot simply call the environment.
That makes divergence the standing risk, and equivalence the standing test.

The parity here is asserted **exactly**, not to a tolerance. The metric does the
same arithmetic in the same order as ``Flight4DEnv``, so anything less than
equality is a real divergence rather than floating-point noise, and loosening
these bounds would be the quietest possible way to make the leaderboard wrong.
"""
import doctest
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from competition.config import EXIT_MISS_BUCKET, SCENARIO
from competition.kaggle import metric as km
from competition.kaggle.export import (scenario_states, split_usage,
                                       write_solution)
from competition.rollout import make_env, replay_actions
from competition.seeds import load_public
from competition.submission import max_action, write_submission

N_FLIGHTS = SCENARIO.n_flights
N_ACTIONS = 1 + max_action(N_FLIGHTS)
SEEDS = load_public()[:6]


def random_traces(rng, seeds, noop_bias=0.0):
    """Flat action traces, optionally weighted toward the idle action.

    Sampled over the *whole* action space, legal or not: a submission may carry
    anything, and the mask's rejection of an illegal index is one of the things
    the two implementations have to agree about.
    """
    traces = {}
    for seed in seeds:
        a = rng.integers(0, N_ACTIONS, size=SCENARIO.max_steps)
        if noop_bias:
            a = np.where(rng.random(a.shape) < noop_bias, 0, a)
        traces[seed] = a.astype(np.int64)
    return traces


def legal_traces(rng, seeds):
    """Traces a masked policy could actually have produced.

    Random indices are mostly illegal, so they exercise the mask but barely
    exercise the *dynamics*. These fly the auto-reverting macros for real.
    """
    traces = {}
    env = make_env()
    for seed in seeds:
        env.reset(seed=seed)
        actions = []
        for _ in range(SCENARIO.max_steps):
            legal = np.flatnonzero(env.action_masks())
            a = int(rng.choice(legal))
            actions.append(a)
            _, _, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break
        traces[seed] = np.array(actions, dtype=np.int64)
    return traces


def assert_same_episode(mirror, reference, seed):
    assert mirror["completed"] == reference.completed, f"seed {seed}"
    assert mirror["steps"] == reference.steps, f"seed {seed}"
    assert mirror["congestion"] == reference.congestion, f"seed {seed}"
    assert mirror["clearances"] == reference.clearances, f"seed {seed}"
    assert mirror["invalids"] == reference.invalids, f"seed {seed}"
    assert mirror["aircraft_involved"] == reference.aircraft_involved, f"seed {seed}"
    assert mirror["exit_miss"] == reference.exit_miss, f"seed {seed}"
    assert mirror["max_exit_miss"] == reference.max_exit_miss, f"seed {seed}"


@pytest.mark.parametrize("noop_bias", [0.0, 0.5, 0.9, 1.0])
def test_metric_replay_matches_the_real_environment(noop_bias):
    """Same actions, same seeds: the sandbox replay and the environment must
    agree on every KPI. This is the test that makes the public board mean
    anything."""
    rng = np.random.default_rng(int(noop_bias * 100))
    traces = random_traces(rng, SEEDS, noop_bias)
    states = scenario_states(SEEDS)

    for seed in SEEDS:
        reference = replay_actions(traces[seed], seed)
        mirror = km.replay_episode(np.array(states[seed]), traces[seed])
        assert_same_episode(mirror, reference, seed)


def test_metric_matches_a_masked_random_policy():
    """The mask makes most random indices illegal, so a trace of legal actions
    is the only one that really flies the macros — the hold, the revert and the
    symmetric speed pulse."""
    rng = np.random.default_rng(4)
    traces = legal_traces(rng, SEEDS)
    states = scenario_states(SEEDS)
    assert max(t.max() for t in traces.values()) > 0, "trace is all idle"

    for seed in SEEDS:
        reference = replay_actions(traces[seed], seed)
        mirror = km.replay_episode(np.array(states[seed]), traces[seed])
        assert_same_episode(mirror, reference, seed)


def test_metric_matches_the_rule_based_agent():
    """Random actions exercise the rails; a real controller exercises the
    behaviour the competition actually scores."""
    from competition.agents import load_agent
    from competition.rollout import rollout_agent

    agent = load_agent("rule-based")
    states = scenario_states(SEEDS)
    for seed in SEEDS:
        reference, actions = rollout_agent(agent, seed)
        mirror = km.replay_episode(np.array(states[seed]), actions)
        assert_same_episode(mirror, reference, seed)


def test_the_metric_mirrors_the_environments_frozen_constants():
    """The metric restates the physics because it cannot import it. Every number
    it restates is checked against the real one here — that is the whole deal."""
    env = make_env()
    # `sim` is built by reset(), and the altitude ladder lives on it.
    env.reset(seed=0)
    assert km.DT == env.DT
    assert km.DALT == env.DALT
    assert km.DVEL == env.DVEL
    assert km.ALT_MIN_SEP == env.ALT_MIN_SEP
    assert km.D_SEP == env.D_SEP
    assert km.COLLISION_RADIUS == env.COLLISION_RADIUS
    assert km.VEL_MIN == env.vel_min
    assert km.VEL_MAX == env.vel_max
    assert km.ALT_MIN == env.sim.allowable_altitudes.min()
    assert km.ALT_MAX == env.sim.allowable_altitudes.max()
    assert km.N_CLEARANCES * env.n_flights + 1 == env.n_actions
    assert km.MAX_STEPS == SCENARIO.max_steps == env.max_steps
    assert km.HOLD_STEPS == SCENARIO.hold_steps == env.hold_steps
    assert km.MAX_CLEARANCES == env.max_clearances
    assert km.EXIT_MISS_BUCKET == EXIT_MISS_BUCKET


# ---------------------------------------------------------------- packing

def test_pack_preserves_objective_priority():
    """Safety must beat timeliness must beat efficiency, at the extremes."""
    safe = km.pack(failed=0, congestion=1, exit_miss_mean=9.9, clearances=19_999)
    unsafe = km.pack(failed=0, congestion=2, exit_miss_mean=0.0, clearances=0)
    assert safe < unsafe

    prompt = km.pack(failed=0, congestion=5, exit_miss_mean=0.01, clearances=19_999)
    cheap = km.pack(failed=0, congestion=5, exit_miss_mean=5.00, clearances=0)
    assert prompt < cheap

    lean = km.pack(failed=0, congestion=5, exit_miss_mean=0.10, clearances=10)
    busy = km.pack(failed=0, congestion=5, exit_miss_mean=0.10, clearances=900)
    assert lean < busy

    assert km.pack(0, 99_999, 9.9, 19_999) < km.pack(1, 0, 0.0, 0)


def test_packed_score_is_exact_in_float64():
    """If the span exceeded 2**53 the ordering would silently break on ties."""
    worst = km.pack(km.BOUND_FAILED, km.BOUND_CONGESTION,
                    km.BOUND_EXIT_MISS * km.EXIT_MISS_BUCKET, km.BOUND_CLEARANCES)
    assert worst < 2 ** 53
    assert worst == float(int(worst))
    assert km.pack(0, 0, 0.0, 0) == 0.0


def test_pack_clamps_rather_than_wrapping():
    """An absurd submission must saturate at the bottom of the board, never wrap
    around and appear near the top."""
    huge = km.pack(failed=10 ** 6, congestion=10 ** 9,
                   exit_miss_mean=10 ** 6, clearances=10 ** 9)
    assert huge == km.pack(km.BOUND_FAILED, km.BOUND_CONGESTION,
                           km.BOUND_EXIT_MISS * km.EXIT_MISS_BUCKET,
                           km.BOUND_CLEARANCES)


def test_the_exit_miss_bound_is_above_anything_reachable():
    """A clamp that bites would flatten real differences into a tie."""
    rng = np.random.default_rng(0)
    traces = random_traces(rng, SEEDS)
    worst = max(replay_actions(traces[s], s).exit_miss for s in SEEDS)
    assert worst < km.BOUND_EXIT_MISS * km.EXIT_MISS_BUCKET / 2


# ---------------------------------------------------------------- score()

def _frames(tmp_path, seeds, traces):
    sol = tmp_path / "solution.csv"
    sub = tmp_path / "submission.csv"
    write_solution(sol, seeds)
    write_submission(sub, traces)
    return pd.read_csv(sol), pd.read_csv(sub)


def test_score_end_to_end_agrees_with_the_harness(tmp_path):
    from competition.score import aggregate

    rng = np.random.default_rng(7)
    traces = random_traces(rng, SEEDS, noop_bias=0.8)
    solution, submission = _frames(tmp_path, SEEDS, traces)

    kaggle_value = km.score(solution, submission, "id")

    episodes = [replay_actions(traces[s], s) for s in SEEDS]
    local = aggregate(episodes)
    expected = km.pack(local.failed_episodes, local.congestion_total,
                       local.exit_miss_mean, local.clearances_total)
    assert kaggle_value == expected


def test_score_accepts_traces_from_episodes_that_ended_early(tmp_path):
    """The regression that would have voided the whole public leaderboard.

    ``legal_traces`` stops when the episode does, which is what every real agent
    produces — the brick ends most episodes in a mid-air after a dozen steps.
    The solution file is exported at the full horizon, so a submission written
    without padding is missing rows and the metric refuses it. Every real
    submission was refused; none of the tests above noticed, because all of them
    build traces that happen to run the full 50 steps.
    """
    from competition.score import aggregate

    rng = np.random.default_rng(11)
    traces = legal_traces(rng, SEEDS)
    assert min(len(t) for t in traces.values()) < SCENARIO.max_steps, \
        "this test is pointless unless some episode ends early"

    solution, submission = _frames(tmp_path, SEEDS, traces)
    assert len(submission) == len(SEEDS) * SCENARIO.max_steps

    local = aggregate([replay_actions(traces[s], s) for s in SEEDS])
    assert km.score(solution, submission, "id") == km.pack(
        local.failed_episodes, local.congestion_total,
        local.exit_miss_mean, local.clearances_total)


# ------------------------------------------------- Kaggle's own constraints

def test_score_satisfies_kaggles_metric_notebook_contract():
    """Kaggle's metric notebook validates four things, and rejects the metric if
    any of them is wrong. All four are invisible from inside this repository, so
    a refactor could break one and we would only find out on the leaderboard.
    """
    sig = inspect.signature(km.score)
    params = list(sig.parameters)
    assert params[:3] == ["solution", "submission", "row_id_column_name"], \
        "Kaggle calls score() positionally; the first three names are fixed"
    for name, param in sig.parameters.items():
        assert param.annotation is not inspect.Parameter.empty, \
            f"every argument to score() must be annotated; {name} is not"
    assert sig.return_annotation is float

    # "must return a single, finite, non-null float"
    value = km.pack(failed=0, congestion=0, exit_miss_mean=0.0, clearances=0)
    assert isinstance(value, float) and np.isfinite(value)


def test_score_docstring_is_the_host_facing_description():
    """Kaggle shows this docstring when a host selects the metric, capped at
    8 000 characters, and its examples are the only executable documentation of
    the submission format. The examples print their errors instead of raising so
    they read identically whether this file is imported or pasted into a
    notebook cell, where the exception's qualified name would differ.
    """
    doc = km.score.__doc__
    assert doc and len(doc) < 8_000
    assert "Lower is better" in doc, "the sort-order toggle has to match this"

    runner = doctest.DocTestRunner()
    for test in doctest.DocTestFinder().find(km.score, "score", globs=vars(km)):
        runner.run(test)
    assert runner.failures == 0
    assert runner.tries >= 8, "the worked examples went missing"


def test_unpack_docstring_examples_pass():
    """`unpack` is what a participant or organiser reaches for when a raw
    leaderboard number needs reading instead of sorting. Its worked example is
    the only executable proof that the round-trip actually holds, the same way
    score()'s doctest is the only executable proof of the submission format.
    """
    runner = doctest.DocTestRunner()
    for test in doctest.DocTestFinder().find(km.unpack, "unpack", globs=vars(km)):
        runner.run(test)
    assert runner.failures == 0
    assert runner.tries >= 2, "the worked example went missing"


def test_pack_unpack_round_trip():
    """`unpack` must invert `pack` exactly, everywhere `pack` can land — it is a
    mixed-radix encoding, not a fit, so this is decode-and-compare, not a
    tolerance.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        failed = int(rng.integers(0, km.BOUND_FAILED + 1))
        congestion = int(rng.integers(0, km.BOUND_CONGESTION + 1))
        clearances = int(rng.integers(0, km.BOUND_CLEARANCES + 1))
        bucket = int(rng.integers(0, km.BOUND_EXIT_MISS + 1))
        packed = km.pack(failed, congestion, bucket * km.EXIT_MISS_BUCKET,
                         clearances)
        assert km.unpack(packed) == {
            "failed": failed, "congestion": congestion,
            "exit_miss_bucket": bucket, "clearances": clearances}


def test_score_reads_only_the_split_kaggle_hands_it(tmp_path):
    """Kaggle calls the metric twice — once with the Public rows, once with the
    Private ones — against the same full submission. Each call must score that
    split alone, or the live board and the final board would be the same number.
    """
    from competition.score import aggregate

    rng = np.random.default_rng(13)
    traces = random_traces(rng, SEEDS, noop_bias=0.8)
    sol_path, sub_path = tmp_path / "solution.csv", tmp_path / "submission.csv"
    write_solution(sol_path, SEEDS, holdout_frac=0.5)
    write_submission(sub_path, traces)
    solution, submission = pd.read_csv(sol_path), pd.read_csv(sub_path)

    split = split_usage(SEEDS, 0.5)
    assert set(split.values()) == {"Public", "Private"}

    for usage in ("Public", "Private"):
        rows = solution[solution["Usage"] == usage]
        seeds = [s for s in SEEDS if split[s] == usage]
        local = aggregate([replay_actions(traces[s], s) for s in seeds])
        assert km.score(rows.drop(columns=["Usage"]), submission, "id") == km.pack(
            local.failed_episodes, local.congestion_total,
            local.exit_miss_mean, local.clearances_total)


def test_score_rejects_a_submission_missing_rows(tmp_path):
    rng = np.random.default_rng(1)
    traces = random_traces(rng, SEEDS, noop_bias=0.8)
    solution, submission = _frames(tmp_path, SEEDS, traces)
    with pytest.raises(km.ParticipantVisibleError, match="missing"):
        km.score(solution, submission.iloc[:-25], "id")


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.assign(action=99), "range 0-35"),
    (lambda d: d.assign(action="x"), "integer"),
    (lambda d: d.drop(columns=["action"]), "'action' column"),
    (lambda d: d.assign(id="nonsense"), "Malformed row id"),
])
def test_score_gives_participants_an_actionable_error(tmp_path, mutate, match):
    rng = np.random.default_rng(2)
    traces = random_traces(rng, SEEDS, noop_bias=0.8)
    solution, submission = _frames(tmp_path, SEEDS, traces)
    with pytest.raises(km.ParticipantVisibleError, match=match):
        km.score(solution, mutate(submission), "id")


def test_score_reports_a_broken_solution_as_the_hosts_problem(tmp_path):
    rng = np.random.default_rng(3)
    traces = random_traces(rng, SEEDS, noop_bias=0.8)
    solution, submission = _frames(tmp_path, SEEDS, traces)
    with pytest.raises(km.ParticipantVisibleError, match="host-side"):
        km.score(solution.drop(columns=["f2_alt"]), submission, "id")


def test_metric_imports_without_the_bootcamp_stack():
    """The sandbox has pandas and numpy. If the metric quietly grew a dependency
    on gymnasium, matplotlib or torch, the public board would stop working."""
    import subprocess
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(ROOT / 'competition' / 'kaggle')!r});"
        "import metric;"
        "print(any(m in sys.modules for m in "
        "('torch', 'gymnasium', 'matplotlib', 'stable_baselines3')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("False"), "metric grew a heavy dependency"
