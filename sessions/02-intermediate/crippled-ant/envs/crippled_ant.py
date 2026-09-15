"""CrippledAnt — a Gymnasium wrapper that disables joints of the MuJoCo Ant.

The wrapper models a hardware failure: torques commanded to the disabled
joints are forced to zero before they reach the simulator. The observation
is left untouched — the agent still *sees* the crippled joints, it just
cannot actuate them. (A real robot with a dead motor still has encoders.)

Ant-v5 action layout (one hip+ankle pair per leg):

    index  0      1        2      3        4      5        6      7
    joint  hip_1  ankle_1  hip_2  ankle_2  hip_3  ankle_3  hip_4  ankle_4
    leg    0 front-left    1 front-right   2 back-left     3 back-right

Four injury modes (mutually exclusive):

- ``disabled_joints=[2, 3]``  — fixed joint indices (Exercise 3).
- ``disabled_legs=[0]``       — fixed legs, expanded to their joint pairs.
- ``n_random_legs=1``         — domain randomisation at a fixed severity:
  re-draw which ``n_random_legs`` legs are disabled on *every* reset
  (Exercise 5). Reproducible via ``reset(seed=)``.
- ``n_random_legs_max=4``     — domain randomisation over severity as well as
  identity: draw the *count* of disabled legs uniformly from
  ``{0, 1, ..., n_random_legs_max}`` on every reset, then draw that many legs.
  ``0`` is in the support, so a healthy Ant is part of the training
  distribution — the fixed-count variant above never shows the policy one.
"""
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np

N_JOINTS = 8
N_LEGS = 4

#: leg index -> (hip joint, ankle joint) action indices
LEG_JOINTS = {leg: (2 * leg, 2 * leg + 1) for leg in range(N_LEGS)}


class CrippledAnt(gym.Wrapper):
    """Zero out the torques applied to a set of disabled joints."""

    def __init__(
        self,
        env: gym.Env,
        disabled_joints: Optional[Sequence[int]] = None,
        disabled_legs: Optional[Sequence[int]] = None,
        n_random_legs: Optional[int] = None,
        n_random_legs_max: Optional[int] = None,
    ):
        super().__init__(env)

        static = bool(disabled_joints) or bool(disabled_legs)
        modes = [static, n_random_legs is not None, n_random_legs_max is not None]
        if sum(modes) > 1:
            raise ValueError(
                "disabled_joints/disabled_legs, n_random_legs and "
                "n_random_legs_max are mutually exclusive"
            )
        if n_random_legs is not None and not 0 <= n_random_legs <= N_LEGS:
            raise ValueError(f"n_random_legs must be in [0, {N_LEGS}]")
        if n_random_legs_max is not None and not 0 <= n_random_legs_max <= N_LEGS:
            raise ValueError(f"n_random_legs_max must be in [0, {N_LEGS}]")

        joints = set(disabled_joints or [])
        for leg in disabled_legs or []:
            if leg not in LEG_JOINTS:
                raise ValueError(f"leg index {leg} not in [0, {N_LEGS - 1}]")
            joints.update(LEG_JOINTS[leg])
        if any(j not in range(N_JOINTS) for j in joints):
            raise ValueError(f"joint indices must be in [0, {N_JOINTS - 1}]")

        self.n_random_legs = n_random_legs
        self.n_random_legs_max = n_random_legs_max
        self._static_joints = np.array(sorted(joints), dtype=int)
        self._current_joints = self._static_joints
        self._current_n_legs = len(disabled_legs or [])
        self._rng = np.random.default_rng()

    @property
    def disabled_joints(self) -> tuple:
        """Joint indices currently receiving zero torque."""
        return tuple(self._current_joints.tolist())

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.n_random_legs is not None:
            n = self.n_random_legs
            legs = self._rng.choice(N_LEGS, size=n, replace=False)
            self._current_joints = np.array(
                sorted(j for leg in legs for j in LEG_JOINTS[leg]), dtype=int
            )
            self._current_n_legs = n
        elif self.n_random_legs_max is not None:
            # Sample the SEVERITY too, not just which legs — 0 is in the
            # support, so a healthy Ant is part of the training distribution.
            n = int(self._rng.integers(0, self.n_random_legs_max + 1))
            legs = self._rng.choice(N_LEGS, size=n, replace=False)
            self._current_joints = np.array(
                sorted(j for leg in legs for j in LEG_JOINTS[leg]), dtype=int
            )
            self._current_n_legs = n
        obs, info = self.env.reset(seed=seed, options=options)
        info["disabled_joints"] = self.disabled_joints
        info["n_disabled_legs"] = self._current_n_legs
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        action[self._current_joints] = 0.0
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["disabled_joints"] = self.disabled_joints
        info["n_disabled_legs"] = self._current_n_legs
        return obs, reward, terminated, truncated, info


def make_ant(
    disabled_joints: Optional[Sequence[int]] = None,
    disabled_legs: Optional[Sequence[int]] = None,
    n_random_legs: Optional[int] = None,
    n_random_legs_max: Optional[int] = None,
    **ant_kwargs,
) -> gym.Env:
    """Build Ant-v5, wrapped in CrippledAnt only if an injury is specified.

    ``ant_kwargs`` are forwarded to ``gym.make("Ant-v5", ...)`` — e.g.
    ``include_cfrc_ext_in_observation=False`` for the 27-dim observation,
    ``render_mode="rgb_array"`` for video recording.
    """
    env = gym.make("Ant-v5", **ant_kwargs)
    if (disabled_joints or disabled_legs or n_random_legs is not None
            or n_random_legs_max is not None):
        env = CrippledAnt(
            env,
            disabled_joints=disabled_joints,
            disabled_legs=disabled_legs,
            n_random_legs=n_random_legs,
            n_random_legs_max=n_random_legs_max,
        )
    return env
