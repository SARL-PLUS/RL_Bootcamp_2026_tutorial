"""Autoregressive, order-invariant, mask-aware policy for :mod:`envs.flight_4d`.

Ported from a prior air-traffic control project's ``ATCPolicyNetwork``,
which is genuinely two-stage::

    aircraft_logits  = aircraft_head(combined)                    # [F]
    selected_index   = Categorical(aircraft_logits).sample()
    clearance_logits = clearance_head(combined[selected_index])   # [C]

Stage two is applied to the **selected** aircraft's embedding, and its mask is
that aircraft's row of the legality matrix.

An honest note on what autoregression buys here
------------------------------------------------
If stage two is the *same shared MLP* applied to aircraft *i*'s embedding — which
is what that design does and what this does — then sampling ``i`` and then ``c|i`` gives
**exactly** the joint you would get by computing every ``log p(i) + log p(c|i)``
in parallel and sampling once from the flat Categorical. The two are
distributionally identical; only the compute differs.

So the win is not the autoregression as such. It is:

* **stage-two masking conditioned on the selected aircraft** — the legality of
  ``+dv`` depends on which aircraft you picked, and
* **an aircraft with no legal clearance being unselectable**, which a flat head
  has to be told about separately.

Both are implemented below. The structure is kept two-stage anyway because it is
what the design calls for and because it is the natural place to later give stage
two something stage one did not see (a selection embedding, attention over the
other traffic conditioned on the choice) — at which point the equivalence
genuinely breaks and autoregression starts to pay.

Verification
------------
``tests/test_flight_4d.py`` asserts that ``log_prob`` and ``entropy`` of this
distribution agree with a flat ``Categorical`` built from
``log p(i) + log p(c|i)``, to float tolerance, on random logits and masks. If
that ever fails, the factorisation is wrong.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

NEG_INF = -1e8


# ==========================================================================
# distribution
# ==========================================================================
class AutoregressiveClearance(Distribution):
    """Two-stage categorical over ``IDLE | (aircraft, clearance)``.

    Actions are the flat index the environment expects — ``0`` for idle,
    ``1 + C*i + c`` otherwise — so the environment, the rollout buffer and the
    submission format never see the factorisation.
    """

    def __init__(self, n_flights: int, n_clearances: int):
        super().__init__()
        self.F = n_flights
        self.C = n_clearances

    def proba_distribution_net(self, latent_dim: int):    # pragma: no cover
        raise NotImplementedError("logits come from the policy's own heads")

    def proba_distribution(self, stage1_logits: torch.Tensor,
                           stage2_logits: torch.Tensor) -> "AutoregressiveClearance":
        """``stage1_logits`` is ``[B, 1+F]`` (idle first); ``stage2`` is ``[B, F, C]``."""
        self.log_p1 = F.log_softmax(stage1_logits, dim=-1)          # [B, 1+F]
        self.log_p2 = F.log_softmax(stage2_logits, dim=-1)          # [B, F, C]
        return self

    # -- the joint, used for log_prob / entropy / mode ----------------------
    def _flat_log_probs(self) -> torch.Tensor:
        """``[B, 1 + C*F]`` log-probabilities of the flat action space.

        ``log p(idle)`` unchanged, and ``log p(i) + log p(c|i)`` for the rest —
        which is the chain rule, not an approximation of it.
        """
        joint = self.log_p1[:, 1:].unsqueeze(-1) + self.log_p2       # [B, F, C]
        return torch.cat([self.log_p1[:, :1],
                          joint.reshape(joint.shape[0], -1)], dim=1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        flat = self._flat_log_probs()
        return flat.gather(1, actions.long().reshape(-1, 1)).squeeze(1)

    def entropy(self) -> torch.Tensor:
        """``H(joint) = H(stage1) + E_i[H(stage2_i)]``, computed exactly.

        Masked entries contribute ``0 * -inf``; ``nan_to_num`` is the standard
        guard, and is safe because a masked entry has probability exactly zero.
        """
        p1 = self.log_p1.exp()
        h1 = -(p1 * self.log_p1).nan_to_num(0.0).sum(-1)             # [B]
        p2 = self.log_p2.exp()
        h2 = -(p2 * self.log_p2).nan_to_num(0.0).sum(-1)             # [B, F]
        return h1 + (p1[:, 1:] * h2).sum(-1)

    def sample(self) -> torch.Tensor:
        """Two-stage sampling: pick the aircraft, then its clearance."""
        i = torch.distributions.Categorical(logits=self.log_p1).sample()   # [B]
        act = torch.zeros_like(i)
        acting = i > 0
        if acting.any():
            rows = torch.nonzero(acting, as_tuple=False).squeeze(-1)
            sel = i[rows] - 1                                              # aircraft idx
            logits_sel = self.log_p2[rows, sel]                            # [k, C]
            c = torch.distributions.Categorical(logits=logits_sel).sample()
            act[rows] = 1 + self.C * sel + c
        return act

    def mode(self) -> torch.Tensor:
        """The true joint argmax, not greedy stage-one-then-stage-two.

        Greedy is not the mode: the best aircraft on stage one can have a flat
        clearance distribution while a slightly worse one has a peaked one.
        """
        return self._flat_log_probs().argmax(dim=1)

    def actions_from_params(self, stage1_logits, stage2_logits,
                            deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(stage1_logits, stage2_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, stage1_logits, stage2_logits):
        actions = self.actions_from_params(stage1_logits, stage2_logits)
        return actions, self.log_prob(actions)


# ==========================================================================
# encoder
# ==========================================================================
class _Encoder(nn.Module):
    """Shared per-aircraft MLP plus mean/max pooling — a Deep Sets encoder.

    One set of weights sees every aircraft, so "converging traffic should step
    aside" is learned once rather than once per slot, and permuting the traffic
    permutes the outputs. Equivariance by construction, not by training.
    """

    def __init__(self, n_flights: int, n_feats: int, n_globals: int, hidden: int):
        super().__init__()
        self.F, self.n_feats, self.n_globals = n_flights, n_feats, n_globals
        self.per_flight = nn.Sequential(
            nn.Linear(n_feats, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.context_dim = 2 * hidden + n_globals
        self.hidden = hidden

    def forward(self, obs: torch.Tensor):
        B = obs.shape[0]
        flights = obs[:, :self.F * self.n_feats].reshape(B, self.F, self.n_feats)
        globals_ = obs[:, self.F * self.n_feats:]
        emb = self.per_flight(flights)                                # [B, F, H]
        pooled = torch.cat([emb.mean(1), emb.max(1).values, globals_], dim=-1)
        return emb, pooled


# ==========================================================================
# policy
# ==========================================================================
class AutoregressivePolicy(ActorCriticPolicy):
    """``ActorCriticPolicy`` with a two-stage head and env-supplied masks.

    The mask arrives as the **last ``1 + C*F`` entries of the observation**,
    supplied by the environment itself. SB3's rollout buffer only carries
    observations, so a mask passed any other way is silently absent at update
    time — where the ratio is recomputed and a mismatch would corrupt the
    gradient rather than raise.
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 n_flights: int, n_feats: int, n_globals: int,
                 n_clearances: int, hidden: int = 128, idle_bias: float = 0.0,
                 **kwargs):
        self._n_flights = n_flights
        self._n_feats = n_feats
        self._n_globals = n_globals
        self._n_clearances = n_clearances
        self._hidden = hidden
        self._idle_bias = float(idle_bias)
        self._obs_dim = n_flights * n_feats + n_globals
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        h = self._hidden
        self.encoder = _Encoder(self._n_flights, self._n_feats,
                                self._n_globals, h)
        head_in = h + self.encoder.context_dim

        # Stage one: which aircraft. One scalar per flight, shared weights.
        self.aircraft_head = nn.Sequential(
            nn.Linear(head_in, h), nn.ReLU(), nn.Linear(h, 1))
        # Stage two: what to tell it. Same shape, C outputs.
        self.clearance_head = nn.Sequential(
            nn.Linear(head_in, h), nn.ReLU(), nn.Linear(h, self._n_clearances))
        # Idling is a decision about the whole sector, so it reads the context.
        self.idle_head = nn.Sequential(
            nn.Linear(self.encoder.context_dim, h), nn.ReLU(), nn.Linear(h, 1))
        self.value_net_ = nn.Sequential(
            nn.Linear(self.encoder.context_dim, h), nn.ReLU(), nn.Linear(h, 1))

        relu_gain = float(sqrt(2.0))
        for head, last in ((self.aircraft_head, 0.01), (self.clearance_head, 0.01),
                           (self.idle_head, 0.01), (self.value_net_, 1.0)):
            linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
            for k, m in enumerate(linears):
                init.orthogonal_(m.weight, gain=last if k == len(linears) - 1 else relu_gain)
                init.zeros_(m.bias)
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=relu_gain)
                init.zeros_(m.bias)

        # Start biased towards idling. Not a style preference: an unbiased policy
        # over 5 aircraft x 7 clearances acts on ~97% of steps and exhausts the
        # clearance budget long before the conflicts arrive.
        with torch.no_grad():
            self.idle_head[-1].bias.fill_(self._idle_bias)

        self.action_dist = AutoregressiveClearance(self._n_flights, self._n_clearances)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    # -- logits ------------------------------------------------------------
    def _logits_and_value(self, obs: torch.Tensor):
        obs = obs.float()
        core, mask = obs[:, :self._obs_dim], obs[:, self._obs_dim:] > 0.5
        emb, ctx = self.encoder(core)
        ctx_b = ctx.unsqueeze(1).expand(-1, self._n_flights, -1)
        head_in = torch.cat([emb, ctx_b], dim=-1)                    # [B, F, H+ctx]

        stage2 = self.clearance_head(head_in)                        # [B, F, C]
        legal = mask[:, 1:].reshape(-1, self._n_flights, self._n_clearances)
        stage2 = stage2.masked_fill(~legal, NEG_INF)

        select = self.aircraft_head(head_in).squeeze(-1)             # [B, F]
        # An aircraft with no legal clearance must not be selectable, or stage
        # two is handed a fully masked row and the softmax is undefined.
        select = select.masked_fill(~legal.any(-1), NEG_INF)
        idle = self.idle_head(ctx)                                   # [B, 1]

        stage1 = torch.cat([idle, select], dim=1)                    # [B, 1+F]
        values = self.value_net_(ctx).squeeze(-1)
        return stage1, stage2, values

    # -- SB3 surface -------------------------------------------------------
    def forward(self, obs, deterministic: bool = False):
        s1, s2, values = self._logits_and_value(obs)
        dist = self.action_dist.proba_distribution(s1, s2)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, values, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        s1, s2, values = self._logits_and_value(obs)
        dist = self.action_dist.proba_distribution(s1, s2)
        return values, dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        s1, s2, _ = self._logits_and_value(obs)
        return self.action_dist.proba_distribution(s1, s2)

    def predict_values(self, obs):
        return self._logits_and_value(obs)[2]

    def _predict(self, observation, deterministic: bool = False):
        return self.forward(observation, deterministic=deterministic)[0]


class MaskInObsWrapper(__import__("gymnasium").Wrapper):
    """Deprecated no-op. :class:`Flight4DEnv` now carries the mask natively.

    Kept so older scripts import cleanly; it asserts rather than double-appending,
    because silently adding a second copy of the mask would shift every feature
    index the policy reads and fail as a wrong answer rather than an error.
    """

    def __init__(self, env):
        super().__init__(env)
        assert hasattr(env.unwrapped, "mask_offset"), \
            "this env does not carry its own mask; the wrapper is no longer the way"
