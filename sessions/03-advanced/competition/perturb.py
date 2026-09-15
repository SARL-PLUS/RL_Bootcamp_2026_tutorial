"""Agent wrappers for the post-mortem passes.

Each wrapper presents the same ``predict(obs, deterministic=...)`` interface as
the agent it wraps, so the existing rollout code scores a perturbed policy
without knowing anything has changed.

A step is now **one decision** — idle, or one clearance to one aircraft — so a
perturbation is per step rather than per aircraft, and it stays inside the
environment's own legality mask. Corrupting an action into an illegal one would
measure how often the agent was pushed into a no-op, which is a much blunter
question than the robustness curve is meant to ask.
"""
import numpy as np

from . import _paths  # noqa: F401  (import side effect: sys.path)
from envs.flight_4d import N_CLEARANCES, N_FEATS, N_GLOBALS


def legal_mask(obs) -> np.ndarray:
    """The environment's legality mask, read out of the observation itself.

    The observation is ``N_FEATS`` per aircraft plus ``N_GLOBALS`` plus one
    entry per flat action, so the sector size follows from its length alone and
    the wrappers never have to be told it.
    """
    obs = np.asarray(obs, dtype=np.float64).ravel()
    n_flights = (obs.size - N_GLOBALS - 1) // (N_FEATS + N_CLEARANCES)
    offset = N_FEATS * n_flights + N_GLOBALS
    return obs[offset:] > 0.5


def action_probs(agent, obs):
    """Distribution over the flat action space ``[1 + N_CLEARANCES * F]``, or None.

    Only policies expose one. The rule-based controller and the brick are
    deterministic functions with no distribution behind them, so the passes that
    need probabilities skip them rather than inventing one.
    """
    policy = getattr(agent, "policy", None)
    if policy is None or not hasattr(policy, "get_distribution"):
        return None

    import torch  # local: the replay path must stay torch-free

    with torch.no_grad():
        tensor, _ = policy.obs_to_tensor(obs)
        distribution = policy.get_distribution(tensor)
        # The two-stage clearance distribution knows its own flat joint; a plain
        # SB3 Categorical carries `probs` on the wrapped torch distribution.
        if hasattr(distribution, "_flat_log_probs"):
            probs = distribution._flat_log_probs().exp()
        else:
            inner = getattr(distribution, "distribution", None)
            probs = getattr(inner, "probs", None)
            if probs is None:
                return None
    return probs.cpu().numpy().reshape(-1)


class NextBestAgent:
    """Always take the **second** most likely action.

    A policy whose score collapses under this was winning on razor-thin argmax
    margins — the ranking it produced was luck dressed as a decision. One whose
    score barely moves has a genuinely flat preference there, which is its own
    kind of finding.

    The runner-up is taken among actions the policy gives non-zero probability,
    which under masking is exactly the legal ones. Where only one action is
    available there is no runner-up and the argmax stands.
    """

    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        getattr(self.agent, "reset", lambda: None)()

    def predict(self, obs, deterministic: bool = True):
        probs = action_probs(self.agent, obs)
        if probs is None:
            raise TypeError("next-best needs a policy with an action distribution")
        order = np.argsort(-probs)
        viable = [a for a in order if probs[a] > 0.0]
        choice = viable[1] if len(viable) > 1 else order[0]
        return np.int64(choice), None


class EpsilonRandomAgent:
    """Replace the step's decision with a uniformly random legal one, prob eps.

    Uniform over the **legal** actions rather than over the whole flat space:
    most indices are masked at any moment, so a uniform draw over all of them
    would mostly land on something the environment refuses, and the curve would
    measure refusal rates instead of robustness.
    """

    def __init__(self, agent, eps: float, rng=None):
        self.agent = agent
        self.eps = float(eps)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def reset(self):
        getattr(self.agent, "reset", lambda: None)()

    def predict(self, obs, deterministic: bool = True):
        action, _ = self.agent.predict(obs, deterministic=deterministic)
        action = np.int64(np.asarray(action).ravel()[0])
        if self.rng.random() < self.eps:
            legal = np.flatnonzero(legal_mask(obs))
            action = np.int64(self.rng.choice(legal))
        return action, None
