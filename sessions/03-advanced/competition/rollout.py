"""Run an agent through the frozen scenario and collect KPIs.

Two entry points, deliberately separate:

- :func:`rollout_agent` needs the agent, so it needs whatever the agent needs
  (torch, SB3). Students run it locally to build a submission.
- :func:`replay_actions` needs **only numpy and the environment**, because it has
  to run inside Kaggle's metric sandbox where torch is not available.
- :func:`rollout_submission` is the private board's path: a submitted zip, whose
  policy may observe an environment the student wrote. See :mod:`competition.ingest`.

Both produce the same KPIs from the same environment, which is what makes the
public leaderboard trustworthy: a submitted action trace is re-flown here, and a
trace that was not actually produced by a good policy will not score like one.

There is exactly one action space — ``Discrete(1 + N_CLEARANCES * n_flights)``,
one aircraft commanded per step — so nothing here dispatches on how an agent
talks. An earlier harness had to, and the dispatch was a bug farm: a controller
was once scored in an environment that silently rejected one of its commands.
"""
import numpy as np

from . import _paths
from .config import SCENARIO

#: The scored environment class — the repository's own, loaded by file path so
#: a submitted ``envs/`` package cannot stand in for it. See ``_paths``.
Flight4DEnv = _paths.reference_envs().Flight4DEnv
from .score import EpisodeKPIs, episode_kpis


def _completed(env, scenario) -> bool:
    """Did this episode actually fly the whole sector?

    Not the same as ``truncated``. Flight4DEnv sets ``truncated`` both when the
    horizon is reached *and* when the clearance budget runs out, and sets
    ``terminated`` on a mid-air. Both early exits cut the episode short, and a
    shorter episode accumulates fewer congestion events purely by existing for
    less time — so reading ``truncated`` as success would let a spammy agent buy
    a good safety number by stopping early. The step counter is the only
    unambiguous answer.
    """
    return env.step_idx >= scenario.max_steps


def make_env(scenario=SCENARIO) -> Flight4DEnv:
    """The scored environment. Always ours, never the student's.

    "Ours" is enforced, not assumed: the class comes from
    :func:`competition._paths.reference_envs`, which loads this repository's
    ``envs`` by file path. A zip that ships its own ``envs/`` shadows the public
    name for the team's *observed* environment and nothing else.
    """
    return Flight4DEnv(**scenario.as_env_kwargs())


def rollout_agent(agent, seed: int, scenario=SCENARIO, deterministic: bool = True):
    """One episode. Returns ``(EpisodeKPIs, actions)`` with ``actions[steps]``
    holding the flat action index issued at each step."""
    env = make_env(scenario)
    getattr(agent, "reset", lambda: None)()
    obs, _ = env.reset(seed=seed)

    actions, congestion, info = [], 0, {}
    for _ in range(scenario.max_steps):
        action, _ = agent.predict(obs, deterministic=deterministic)
        action = int(np.asarray(action).ravel()[0])
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        congestion += info["congestion_events"]
        if terminated or truncated:
            break

    kpis = episode_kpis(seed, env.step_idx, _completed(env, scenario),
                        congestion, info, scenario)
    return kpis, np.array(actions, dtype=np.int64)


def replay_actions(actions: np.ndarray, seed: int, scenario=SCENARIO) -> EpisodeKPIs:
    """Re-fly a recorded action trace. numpy only — no torch, no student code."""
    actions = np.asarray(actions, dtype=np.int64).ravel()
    env = make_env(scenario)
    env.reset(seed=seed)

    congestion, info = 0, {}
    for step in range(scenario.max_steps):
        # A trace shorter than the horizon means the submitting run ended early.
        # Idle through the rest so the episode is still measured over the full
        # horizon rather than silently scoring less exposure.
        action = int(actions[step]) if step < len(actions) else 0
        _, _, terminated, truncated, info = env.step(action)
        congestion += info["congestion_events"]
        if terminated or truncated:
            break

    return episode_kpis(seed, env.step_idx, _completed(env, scenario),
                        congestion, info, scenario)


def rollout_submission(submission, seed: int, scenario=SCENARIO):
    """One episode of a submitted zip. Returns ``(EpisodeKPIs, actions)``.

    Two environments run in lockstep on the same seed. The **student's** — which
    is the stock one unless they shipped their own — produces the observations
    their policy was trained to read. **Ours** is stepped with the same action
    and is where every KPI comes from.

    Nothing crosses the boundary except the action index. Their reward, their
    termination shaping and their info dict are never consulted, so an
    environment that models the sector wrongly cannot buy a better score — it
    only feeds their policy worse information. That is why this needs no
    defending against, and does none.

    If their episode ends before ours, the sector is idled out to the horizon.
    Ending early is already how ``failed_episodes`` is earned; it must not also
    be a way to stop accumulating congestion.
    """
    scored = make_env(scenario)
    observed = submission.observed_env(scenario)
    policy = submission.policy()

    obs, _ = observed.reset(seed=seed)
    scored.reset(seed=seed)

    actions, congestion, info = [], 0, {}
    done = False
    for _ in range(scenario.max_steps):
        action, _ = policy.predict(obs, deterministic=submission.deterministic)
        action = int(np.asarray(action).ravel()[0])
        actions.append(action)

        _, _, terminated, truncated, info = scored.step(action)
        congestion += info["congestion_events"]
        if terminated or truncated:
            done = True
            break

        obs, _, their_term, their_trunc, _ = observed.step(action)
        if their_term or their_trunc:
            break

    while not done and scored.step_idx < scenario.max_steps:
        _, _, terminated, truncated, info = scored.step(0)
        congestion += info["congestion_events"]
        if terminated or truncated:
            break

    kpis = episode_kpis(seed, scored.step_idx, _completed(scored, scenario),
                        congestion, info, scenario)
    return kpis, np.array(actions, dtype=np.int64)
