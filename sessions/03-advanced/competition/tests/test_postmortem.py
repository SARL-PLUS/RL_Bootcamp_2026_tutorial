"""Contract for the post-mortem passes.

These decide trophies and get shown to participants next to their name, so the
properties that matter are the ones that stop a pass saying something flattering
but untrue.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from competition.agents import load_agent
from competition.perturb import (EpsilonRandomAgent, NextBestAgent, action_probs,
                                 legal_mask)
from competition.postmortem import (
    PostMortem, award_trophies, best_extraction, run_postmortem,
)
from competition.rollout import make_env
from competition.score import EpisodeKPIs, aggregate
from competition.seeds import load_public

SEEDS = load_public()[:3]


@pytest.fixture(scope="module")
def policy_agent(tmp_path_factory):
    """An untrained policy is enough: these passes test plumbing and semantics,
    not competence."""
    sys.path.insert(0, str(ROOT / "airtraffic"))
    from stable_baselines3 import PPO
    from policies import AutoregressivePolicy
    from envs.flight_4d import N_CLEARANCES, N_FEATS, N_GLOBALS

    env = make_env()
    model = PPO(AutoregressivePolicy, env,
                policy_kwargs=dict(n_flights=env.n_flights, n_feats=N_FEATS,
                                   n_globals=N_GLOBALS, n_clearances=N_CLEARANCES),
                device="cpu", seed=0)
    path = tmp_path_factory.mktemp("m") / "policy.zip"
    model.save(path)
    return load_agent(str(path))


def _obs(seed=0):
    env = make_env()
    obs, _ = env.reset(seed=seed)
    return obs


# ---------------------------------------------------------------- probes

def test_heuristics_have_no_action_distribution():
    """They are deterministic functions, not policies. Inventing a distribution
    for them would make the robustness passes report on something that does not
    exist."""
    assert action_probs(load_agent("noop"), _obs()) is None
    assert action_probs(load_agent("rule-based"), _obs()) is None


def test_policy_probabilities_are_one_flat_distribution(policy_agent):
    """The action does not factorise across flights any more: `Flight4DEnv`
    commands at most one aircraft per step through a single `Discrete`, so there
    is one distribution over `1 + N_CLEARANCES * F`, not one per aircraft."""
    probs = action_probs(policy_agent, _obs())
    env = make_env()
    assert probs.shape == (env.n_actions,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------- next-best

def test_next_best_picks_the_runner_up_everywhere(policy_agent):
    obs = _obs()
    probs = action_probs(policy_agent, obs)
    action, _ = NextBestAgent(policy_agent).predict(obs)
    expected = np.argsort(-probs)[1]
    np.testing.assert_array_equal(action, expected)
    # …and that really is a different decision from the argmax.
    assert (action != probs.argmax(axis=-1)).any()


def test_next_best_refuses_an_agent_without_a_distribution():
    with pytest.raises(TypeError, match="action distribution"):
        NextBestAgent(load_agent("rule-based")).predict(_obs())


# ---------------------------------------------------------------- eps-random

def test_epsilon_zero_is_the_identity():
    base = load_agent("rule-based")
    obs = _obs()
    wrapped = EpsilonRandomAgent(base, 0.0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(wrapped.predict(obs)[0], base.predict(obs)[0])


def test_epsilon_one_replaces_everything():
    base = load_agent("noop")          # always all-zero, so any change shows
    obs = _obs()
    wrapped = EpsilonRandomAgent(base, 1.0, rng=np.random.default_rng(1))
    assert (wrapped.predict(obs)[0] != 0).any()


def test_perturbation_only_ever_substitutes_a_legal_action():
    """A corrupted action must still be one the environment would accept.

    Perturbing into an *illegal* action would measure the mask, not the policy:
    every agent would look equally fragile because the environment rejects the
    move before the policy's judgement is ever tested.
    """
    base = load_agent("noop")
    obs = _obs()
    legal = set(np.flatnonzero(legal_mask(obs)).tolist())
    wrapped = EpsilonRandomAgent(base, 1.0, rng=np.random.default_rng(3))
    drawn = {int(np.asarray(wrapped.predict(obs)[0]).ravel()[0]) for _ in range(60)}
    assert drawn <= legal, f"illegal actions drawn: {sorted(drawn - legal)}"
    assert len(drawn) > 1, "epsilon=1 should explore more than a single action"


# ---------------------------------------------------------------- passes

def test_heuristic_skips_the_passes_that_need_a_distribution():
    report = run_postmortem(load_agent("rule-based"), SEEDS, name="rule-based")
    assert "next-best" in report.skipped
    assert "best-extraction" in report.skipped
    assert not report.has_distribution
    assert "do-nothing" in report.passes and "stochastic" in report.passes


def test_policy_runs_every_pass(policy_agent):
    report = run_postmortem(policy_agent, SEEDS, name="policy", epsilons=(0.1,), k=2)
    assert report.has_distribution
    assert not report.skipped
    for name in ("do-nothing", "stochastic", "next-best", "best-extraction", "eps=0.1"):
        assert name in report.passes


def test_best_extraction_is_reproducible(policy_agent):
    """Sampling draws on torch's global RNG. Without an explicit seed the 'best
    of k' differs run to run and the number cannot be quoted."""
    a = best_extraction(policy_agent, SEEDS, k=3, seed=11)
    b = best_extraction(policy_agent, SEEDS, k=3, seed=11)
    assert a.rank_key == b.rank_key


def test_best_extraction_is_never_worse_than_its_own_samples(policy_agent):
    """It keeps the best episode per seed, so it cannot come out behind a single
    stochastic pass on the same seeds by construction."""
    extracted = best_extraction(policy_agent, SEEDS, k=4, seed=5)
    assert extracted.congestion_total >= 0
    assert extracted.n_episodes == len(SEEDS)


# ---------------------------------------------------------------- trophies

def _report(name, congestion, clearances, failed=0, gap=None, has_dist=False,
            baseline=False):
    episodes = [EpisodeKPIs(seed=i, steps=50, completed=(i >= failed),
                            congestion=congestion, exit_miss=0.0,
                            clearances=clearances, aircraft_involved=1,
                            invalids=0, max_exit_miss=0.0)
                for i in range(3)]
    report = PostMortem(name=name, deterministic=aggregate(episodes),
                        has_distribution=has_dist, is_baseline=baseline)
    brick = aggregate([EpisodeKPIs(seed=i, steps=50, completed=True,
                                   congestion=100, exit_miss=0.0, clearances=0,
                                   aircraft_involved=0, invalids=0,
                                   max_exit_miss=0.0) for i in range(3)])
    report.passes["do-nothing"] = brick
    if gap is not None:
        shifted = aggregate([EpisodeKPIs(seed=i, steps=50, completed=True,
                                         congestion=congestion + gap,
                                         exit_miss=0.0, clearances=clearances,
                                         aircraft_involved=1, invalids=0,
                                         max_exit_miss=0.0)
                             for i in range(3)])
        report.passes["stochastic"] = shifted
    return report


def test_iron_stomach_excludes_deterministic_agents():
    """A heuristic's stochastic gap is 0 because it has nothing to sample from.
    That is not robustness, and it must not win a robustness trophy."""
    awards = award_trophies({
        "heuristic": _report("heuristic", 5, 10, gap=0, has_dist=False),
        "policy": _report("policy", 5, 10, gap=2, has_dist=True),
    })
    assert awards["Iron Stomach"] == ["policy"]


def test_brick_with_wings_excludes_the_brick_itself():
    awards = award_trophies({
        "noop": _report("noop", 100, 0, baseline=True),
        "good": _report("good", 5, 10),
    })
    assert awards["The Brick with Wings"] == ["good"]


def test_zero_conflict_wings_needs_zero_everywhere():
    awards = award_trophies({
        "perfect": _report("perfect", 0, 40),
        "nearly": _report("nearly", 1, 10),
    })
    assert awards["Zero-Conflict Wings"] == ["perfect"]
    assert awards["Minimal-Intervention"] == "perfect"


def test_jury_trophies_are_not_invented():
    awards = award_trophies({"a": _report("a", 5, 10)})
    assert "Black Box Award" not in awards
    assert "Icarus Award" not in awards
