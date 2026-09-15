"""Contract tests for the CrippledAnt wrapper.

These tests define the behaviour every injury implementation must satisfy —
if you modify or replace the wrapper, keep this suite green.
"""
import gymnasium as gym
import numpy as np
import pytest

from envs.crippled_ant import LEG_JOINTS, N_JOINTS, CrippledAnt, make_ant


class FakeAnt(gym.Env):
    """Records the action it receives; no MuJoCo needed."""

    action_space = gym.spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(27,), dtype=np.float64)

    def __init__(self):
        self.last_action = None
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(27), {}

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        self._t += 1
        return np.zeros(27), 1.0, self._t >= 5, False, {}


def test_disabled_joints_receive_zero_torque():
    env = CrippledAnt(FakeAnt(), disabled_joints=[2, 3])
    env.reset()
    env.step(np.ones(N_JOINTS, dtype=np.float32))
    received = env.unwrapped.last_action
    assert np.all(received[[2, 3]] == 0.0)
    assert np.all(received[[0, 1, 4, 5, 6, 7]] == 1.0)


def test_disabled_legs_expand_to_hip_and_ankle():
    env = CrippledAnt(FakeAnt(), disabled_legs=[0, 3])
    assert env.disabled_joints == (*LEG_JOINTS[0], *LEG_JOINTS[3])


def test_callers_action_array_is_not_mutated():
    env = CrippledAnt(FakeAnt(), disabled_joints=[0])
    env.reset()
    action = np.ones(N_JOINTS, dtype=np.float32)
    env.step(action)
    assert np.all(action == 1.0)


def test_spaces_unchanged():
    inner = FakeAnt()
    env = CrippledAnt(inner, disabled_legs=[1])
    assert env.action_space == inner.action_space
    assert env.observation_space == inner.observation_space


def test_info_reports_disabled_joints_on_reset_and_step():
    env = CrippledAnt(FakeAnt(), disabled_legs=[2])
    _, info = env.reset()
    assert info["disabled_joints"] == LEG_JOINTS[2]
    *_, info = env.step(np.zeros(N_JOINTS))
    assert info["disabled_joints"] == LEG_JOINTS[2]


def test_random_legs_disable_correct_count():
    env = CrippledAnt(FakeAnt(), n_random_legs=2)
    env.reset(seed=7)
    assert len(env.disabled_joints) == 4  # 2 legs x (hip + ankle)


def test_random_legs_reproducible_with_same_seed():
    env = CrippledAnt(FakeAnt(), n_random_legs=1)
    env.reset(seed=42)
    first = env.disabled_joints
    env.reset(seed=42)
    assert env.disabled_joints == first


def test_random_legs_resampled_across_resets():
    env = CrippledAnt(FakeAnt(), n_random_legs=1)
    env.reset(seed=0)
    seen = {env.disabled_joints for _ in range(10) if env.reset() or True}
    assert len(seen) > 1  # P(all 10 draws identical) = (1/4)^9


def test_static_and_random_modes_are_mutually_exclusive():
    with pytest.raises(ValueError):
        CrippledAnt(FakeAnt(), disabled_legs=[0], n_random_legs=1)


def test_fixed_and_ranged_random_modes_are_mutually_exclusive():
    with pytest.raises(ValueError):
        CrippledAnt(FakeAnt(), n_random_legs=1, n_random_legs_max=2)


@pytest.mark.parametrize("kwargs", [dict(disabled_joints=[8]), dict(disabled_legs=[4]),
                                    dict(n_random_legs=5), dict(n_random_legs_max=5)])
def test_invalid_indices_raise(kwargs):
    with pytest.raises(ValueError):
        CrippledAnt(FakeAnt(), **kwargs)


def test_random_legs_max_severity_is_in_support():
    """0 legs disabled must be reachable — a healthy Ant belongs in the
    training distribution, which the fixed-count variant never provides."""
    env = CrippledAnt(FakeAnt(), n_random_legs_max=4)
    severities = set()
    for i in range(60):
        _, info = env.reset(seed=i)
        severities.add(info["n_disabled_legs"])
    assert severities == {0, 1, 2, 3, 4}


def test_random_legs_max_zero_severity_means_no_joints_disabled():
    env = CrippledAnt(FakeAnt(), n_random_legs_max=4)
    for i in range(200):
        _, info = env.reset(seed=i)
        if info["n_disabled_legs"] == 0:
            assert env.disabled_joints == ()
            return
    pytest.fail("never sampled severity 0 in 200 resets — check the RNG bounds")


def test_random_legs_max_reproducible_with_same_seed():
    env = CrippledAnt(FakeAnt(), n_random_legs_max=4)
    env.reset(seed=42)
    first = env.disabled_joints
    env.reset(seed=42)
    assert env.disabled_joints == first


def test_random_legs_max_zero_disables_nothing_deterministically():
    """max=0 is the degenerate case: always healthy, never a static mode."""
    env = CrippledAnt(FakeAnt(), n_random_legs_max=0)
    for i in range(10):
        _, info = env.reset(seed=i)
        assert info["n_disabled_legs"] == 0
        assert env.disabled_joints == ()


# --- MuJoCo integration (skipped automatically if mujoco is unavailable) ---
mujoco = pytest.importorskip("mujoco")


def test_make_ant_healthy_is_not_wrapped():
    env = make_ant(include_cfrc_ext_in_observation=False)
    assert not any(isinstance(w, CrippledAnt) for w in _wrapper_chain(env))
    assert env.observation_space.shape == (27,)
    env.close()


def test_make_ant_crippled_steps():
    env = make_ant(disabled_legs=[0], include_cfrc_ext_in_observation=False)
    obs, info = env.reset(seed=0)
    assert info["disabled_joints"] == LEG_JOINTS[0]
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert obs in env.observation_space
    env.close()


def test_gym_make_registration():
    import envs  # noqa: F401  (triggers gym.register)

    env = gym.make("CrippledAnt-v5", disabled_legs=[1])
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()


def _wrapper_chain(env):
    while isinstance(env, gym.Wrapper):
        yield env
        env = env.env
