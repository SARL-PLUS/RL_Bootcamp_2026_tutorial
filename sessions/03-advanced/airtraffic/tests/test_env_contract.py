"""The environment contract every submitted environment must satisfy.

This suite is deliberately **thin and generic**. It says nothing about what an
observation feature *means*, how a clearance is applied, or what the reward
rewards — those are yours to redesign, and they are covered for the shipped
baseline by ``test_flight_4d.py`` and ``test_rule_based_4d.py``.

What it does pin down is the handful of properties the competition harness
relies on to score you at all:

* the spaces are well formed, and the action space is the flat ``Discrete``
  index a submission row carries;
* ``reset(seed=)`` is reproducible, different seeds give different scenarios,
  and the same seed with the same actions replays to the same trajectory —
  **this is what makes an action trace scoreable**, and an environment that
  fails it cannot be ranked on a leaderboard at all;
* the episode ends inside ``max_steps``;
* ``action_masks()`` exists, is consistent with the copy carried in the
  observation, and always offers at least the idle action;
* the info dict carries the KPI keys the score is built from.

Point ``env_factory`` in ``conftest.py`` at your own class to check it.
"""
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from envs.flight_4d import N_FEATS, N_GLOBALS

#: Everything :func:`competition.score.episode_kpis` reads out of ``info``.
#: Rename one of these in your environment and the harness scores you on a
#: quantity that no longer exists.
SCORED_INFO_KEYS = (
    "congestion_events",   # loss-of-separation pairs, this step
    "congestion_total",    # …and cumulatively
    "clearances",          # instructions issued so far
    "invalids",            # illegal actions the mask should have prevented
    "aircraft_involved",   # distinct aircraft touched
    "collided",            # mid-air
    "exit_miss",           # the scored 4D deviation, mean over aircraft
)


def mask_offset(env) -> int:
    """Where the legality mask starts inside the observation.

    Read off the environment when it says, derived from the declared spaces
    otherwise — a subclass is free to lay its features out differently as long
    as the mask is the tail of the vector.
    """
    declared = getattr(env, "mask_offset", None)
    if declared is not None:
        return int(declared)
    return int(env.observation_space.shape[0] - env.action_space.n)


def legal_action(env, rng) -> int:
    """A uniformly random *legal* action, the way a masked policy would act."""
    return int(rng.choice(np.flatnonzero(np.asarray(env.action_masks()))))


# ---------------------------------------------------------------- the spaces

def test_official_gymnasium_checker(make_env):
    # skip_render_check: the sector is drawn with matplotlib, not as pixels
    check_env(make_env(), skip_render_check=True)


def test_action_space_is_a_flat_discrete_index(make_env):
    """A submission row is one integer per (seed, step). That only has a meaning
    if the action space is a single flat ``Discrete``."""
    env = make_env()
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n > 1, "an environment with one action is not a task"


def test_observation_space_is_a_bounded_float_box(make_env):
    env = make_env()
    space = env.observation_space
    assert isinstance(space, spaces.Box)
    assert len(space.shape) == 1, "the policy expects a flat feature vector"
    assert space.dtype == np.float32
    assert np.isfinite(space.low).all() and np.isfinite(space.high).all()


def test_observation_leaves_room_for_the_mask(make_env):
    """The mask is carried in the observation, so the vector is features + mask.

    Its position is the one layout fact the harness needs; everything before it
    is yours.
    """
    env = make_env()
    offset = mask_offset(env)
    assert offset > 0
    assert env.observation_space.shape[0] == offset + env.action_space.n
    # …and the shipped layout is the per-aircraft block plus the globals.
    assert offset == N_FEATS * env.n_flights + N_GLOBALS


# ---------------------------------------------------------------- reset/step

def test_reset_returns_obs_in_space_and_info(make_env):
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs in env.observation_space
    assert isinstance(info, dict)


def test_step_returns_five_tuple_with_correct_types(make_env):
    env = make_env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs in env.observation_space
    assert np.isscalar(reward) and np.isfinite(float(reward))
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_random_legal_rollout_stays_in_observation_space(make_env):
    """200 steps of masked-random play, across resets. The observation must
    never leave the box the policy's input layer was built for."""
    env = make_env()
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(200):
        obs, _, terminated, truncated, _ = env.step(legal_action(env, rng))
        assert obs in env.observation_space, (
            f"obs left the declared space: min={obs.min()}, max={obs.max()}")
        if terminated or truncated:
            env.reset()


def test_episode_ends_within_max_steps(make_env):
    """The horizon is a hard bound, and only a mid-air may cut it short.

    An episode that quietly ends early accumulates fewer conflicts purely by
    existing for less time, so the harness ranks a short episode as a *failure*.
    Idling must not trip that: the only legitimate early ending is a collision,
    which idling into converging traffic can genuinely cause.
    """
    horizon = 10
    env = make_env(max_steps=horizon)
    env.reset(seed=3)
    for t in range(horizon + 5):
        *_, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert terminated or truncated
    assert t < horizon, "the episode ran past its own horizon"
    if t < horizon - 1:
        assert terminated and info["collided"], (
            "idling was penalised into an early ending")


# ---------------------------------------------------------------- determinism

def test_same_seed_gives_same_scenario(make_env):
    obs_a = make_env().reset(seed=42)[0]
    obs_b = make_env().reset(seed=42)[0]
    np.testing.assert_array_equal(obs_a, obs_b)


def test_different_seeds_give_different_scenarios(make_env):
    obs_a = make_env().reset(seed=1)[0]
    obs_b = make_env().reset(seed=2)[0]
    assert not np.array_equal(obs_a, obs_b)


def test_same_seed_same_actions_same_trajectory(make_env):
    """The property the whole competition rests on: a submitted action trace is
    re-flown by the organisers, and must land where it landed for you."""
    def rollout():
        env = make_env()
        obs, _ = env.reset(seed=7)
        rng = np.random.default_rng(7)
        trace, rewards = [obs], []
        for _ in range(20):
            obs, reward, terminated, truncated, _ = env.step(legal_action(env, rng))
            trace.append(obs)
            rewards.append(reward)
            if terminated or truncated:
                break
        return np.concatenate(trace), np.array(rewards)

    obs_a, rew_a = rollout()
    obs_b, rew_b = rollout()
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(rew_a, rew_b)


def test_reset_carries_nothing_over_from_the_last_episode(make_env):
    """One env instance, reset twice on the same seed, with a rollout in
    between. Anything left standing from the first episode shows up here."""
    env = make_env()
    first, _ = env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(15):
        _, _, terminated, truncated, _ = env.step(legal_action(env, rng))
        if terminated or truncated:
            break
    again, _ = env.reset(seed=11)
    np.testing.assert_array_equal(first, again)


# ---------------------------------------------------------------- masking

def test_action_masks_are_well_formed_and_never_empty(make_env):
    env = make_env()
    env.reset(seed=5)
    for _ in range(20):
        mask = np.asarray(env.action_masks())
        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == bool
        assert mask.any(), "a step with no legal action leaves the policy stuck"
        assert mask[0], "the idle action must always be available"
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break


def test_the_observation_carries_the_same_mask_action_masks_returns(make_env):
    """SB3's rollout buffer stores observations and nothing else, so a mask
    passed any other way is present when the action is chosen and absent when
    its log-probability is recomputed. The two copies must never disagree."""
    env = make_env()
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(2)
    for _ in range(20):
        carried = obs[mask_offset(env):]
        assert set(np.unique(carried)) <= {0.0, 1.0}
        np.testing.assert_array_equal(carried.astype(bool),
                                      np.asarray(env.action_masks()))
        obs, _, terminated, truncated, _ = env.step(legal_action(env, rng))
        if terminated or truncated:
            break


def test_a_masked_action_is_refused_rather_than_flown(make_env):
    """Submissions are replayed, and a replay may present an action the live
    policy would never have chosen. It must be rejected and counted, not
    silently applied."""
    env = make_env()
    env.reset(seed=4)
    illegal = np.flatnonzero(~np.asarray(env.action_masks()))
    if not illegal.size:
        pytest.skip("every action is legal in this state")
    before = env.step(0)[4]["invalids"]
    _, _, _, _, info = env.step(int(illegal[0]))
    assert info["invalids"] == before + 1
    assert info["clearances"] == 0, "a masked action must not be charged as one"


# ---------------------------------------------------------------- the KPIs

@pytest.mark.parametrize("key", SCORED_INFO_KEYS)
def test_info_carries_the_scored_kpi_keys(make_env, key):
    env = make_env()
    _, info = env.reset(seed=0)
    assert key in info, f"the scorer reads info[{key!r}]"
    assert key in env.step(0)[4]


def test_the_scored_kpis_are_scalars_of_the_right_kind(make_env):
    env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    for key in ("congestion_events", "congestion_total", "clearances",
                "invalids", "aircraft_involved"):
        assert int(info[key]) == info[key] and info[key] >= 0
    assert isinstance(bool(info["collided"]), bool)
    assert np.isfinite(float(info["exit_miss"])) and info["exit_miss"] >= 0.0


def test_cumulative_kpis_never_go_backwards(make_env):
    """`clearances`, `invalids` and `congestion_total` are episode totals. The
    scorer reads only their final value, so a counter that resets mid-episode
    would report a number no run ever had."""
    env = make_env()
    env.reset(seed=6)
    rng = np.random.default_rng(6)
    last = dict.fromkeys(("clearances", "invalids", "congestion_total"), 0)
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step(legal_action(env, rng))
        for key in last:
            assert info[key] >= last[key], f"{key} decreased mid-episode"
            last[key] = info[key]
        if terminated or truncated:
            break
