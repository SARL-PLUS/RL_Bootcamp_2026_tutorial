"""Training callbacks for :mod:`envs.flight_4d` that PPO does not ship with.

Two knobs, both of which this project needed because a *constant* was measured
to fail in both directions:

- :class:`CurriculumCallback4D` — start on thin traffic and widen to the scored
  density, because the task is learnable at the easy end and not from cold at
  the hard one.
- :class:`AutoEntropyCallback` — hold the policy's entropy at a target instead
  of holding ``ent_coef`` at a guess.

Both are deliberately outer-loop and cheap: one ``env_method`` call when the
difficulty changes, one entropy estimate per rollout. Neither touches the
gradient path.
"""
from __future__ import annotations

import math

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback4D(BaseCallback):
    """Widen ``n_range`` from thin traffic to the scored density during training.

    ``Flight4DEnv`` already draws ``n`` — the number of aircraft on a collision
    course — fresh from ``n_range`` every episode, with ``n + m`` fixed at
    ``n_flights``. So the curriculum here is not "add aircraft" (that would
    change the observation and action spaces mid-run); it is **widen the draw**.
    Training starts with ``n`` pinned at ``n_start`` and the upper bound ramps
    linearly to the environment's configured ``n_hi`` over the first
    ``ramp_frac`` of the run.

    Why widen rather than shift: the easy end must stay in the distribution.
    A curriculum that *moves* the range trains the policy out of the geometries
    it already solved, and the scored configuration contains those geometries —
    ``n`` is drawn uniformly, so a fifth of scored episodes are the easy ones.

    The lower bound never moves. It is the environment's own ``n_lo``.
    """

    def __init__(self, total_steps: int, n_start: int = 2,
                 ramp_frac: float = 0.2, verbose: int = 0):
        super().__init__(verbose)
        self.total_steps = int(total_steps)
        self.n_start = int(n_start)
        self.ramp_frac = float(ramp_frac)
        self._lo = None
        self._hi_end = None
        self._last = None

    def _on_training_start(self) -> None:
        # Read the target range off the environment rather than duplicating the
        # numbers here. Duplicating them has already silently overridden this
        # environment twice (see train_4d.py's note on --hold-steps).
        self._lo = int(self.training_env.get_attr("n_lo")[0])
        self._hi_end = int(self.training_env.get_attr("n_hi")[0])
        self.n_start = max(self.n_start, self._lo)
        if self.verbose:
            print(f"[curriculum] n in [{self._lo}, {self.n_start}] -> "
                  f"[{self._lo}, {self._hi_end}] over "
                  f"{int(self.ramp_frac * self.total_steps)} steps", flush=True)

    def _hi_now(self) -> int:
        progress = self.num_timesteps / max(self.total_steps, 1)
        frac = min(progress / max(self.ramp_frac, 1e-9), 1.0)
        return int(round(self.n_start + (self._hi_end - self.n_start) * frac))

    def _on_step(self) -> bool:
        hi = self._hi_now()
        if hi != self._last:
            self.training_env.env_method("set_n_range", self._lo, hi)
            self._last = hi
            if self.verbose:
                print(f"[curriculum] n ~ U({self._lo}, {hi}) "
                      f"at {self.num_timesteps} steps", flush=True)
        self.logger.record("train/curriculum_n_hi", hi)
        return True


class AutoEntropyCallback(BaseCallback):
    """Hold the policy's entropy near a target by adjusting ``ent_coef``.

    SB3's PPO takes ``ent_coef`` as a **float only** — there is no ``"auto"``
    setting. Only SAC has one, where a dual variable is tuned so the policy's
    entropy tracks a target. This is that mechanism ported to PPO.

    Why bother, rather than picking a constant or a decay schedule: this project
    has failed in both directions with a fixed coefficient. Too high and the
    policy diffuses toward uniform, inflating the clearance rate until the budget
    truncates the episode. Too low and it collapses onto idling. Neither failure
    is about the *coefficient* — they are about the *entropy*, and the mapping
    between the two moves as the reward scale and the advantage normalisation
    change under it. Controlling the quantity you actually care about is more
    robust than guessing the knob that produces it.

    The update is a dual ascent in log space::

        log_coef += lr * (target_entropy - measured_entropy)

    so the coefficient rises while the policy is more deterministic than the
    target and falls while it is more random. ``target`` is a fraction of the
    maximum entropy for the action space, ``ln(n_actions)``, so it stays
    meaningful without retuning if ``n_flights`` changes.
    """

    def __init__(self, target_frac: float = 0.35, lr: float = 0.02,
                 min_coef: float = 1e-5, max_coef: float = 0.1, verbose: int = 0):
        super().__init__(verbose)
        self.target_frac = float(target_frac)
        self.lr = float(lr)
        self.min_coef = float(min_coef)
        self.max_coef = float(max_coef)
        self.log_coef = None
        self.target = None

    def _on_training_start(self) -> None:
        n = int(self.training_env.action_space.n)
        self.target = self.target_frac * math.log(max(n, 2))
        start = min(max(float(self.model.ent_coef), self.min_coef), self.max_coef)
        self.log_coef = math.log(start)
        if self.verbose:
            print(f"[auto-ent] target {self.target:.3f} nats "
                  f"({self.target_frac:.0%} of ln {n}), start coef {start:g}",
                  flush=True)

    def _on_rollout_end(self) -> None:
        obs = self.model.rollout_buffer.observations.reshape(
            -1, *self.model.observation_space.shape)
        # A sample is enough: this is a slow outer-loop controller, not a gradient.
        if obs.shape[0] > 2048:
            idx = np.random.choice(obs.shape[0], 2048, replace=False)
            obs = obs[idx]
        with th.no_grad():
            t = th.as_tensor(obs).to(self.model.device)
            entropy = float(self.model.policy.get_distribution(t).entropy().mean())

        self.log_coef += self.lr * (self.target - entropy)
        coef = float(np.clip(math.exp(self.log_coef), self.min_coef, self.max_coef))
        self.log_coef = math.log(coef)          # keep the dual inside the clip
        self.model.ent_coef = coef

        self.logger.record("train/auto_ent_coef", coef)
        self.logger.record("train/auto_ent_entropy", entropy)
        self.logger.record("train/auto_ent_target", self.target)

    def _on_step(self) -> bool:
        return True
