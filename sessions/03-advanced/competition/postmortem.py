"""Squeeze every number out of a submitted policy.

A leaderboard position is one measurement of one policy under one set of
conditions. These passes ask the questions the position cannot:

- **do-nothing** — does it beat the brick? An agent below that line is worse than
  not being there, and gets the honest number printed beside it.
- **stochastic** — how much of the score survives sampling instead of argmax?
- **next-best** — swap every decision for the runner-up. How much of the score
  rested on razor-thin margins?
- **eps-random** — corrupt actions with probability eps. A robustness curve.
- **best-extraction** — k stochastic rollouts per seed, keep the best. How good is
  this policy when we actively try to extract its best behaviour?

The point, delivered with the trophies: RL is a constant experiment in
controlling a complex dynamic environment. A single leaderboard number is the
beginning of the evaluation, not the end.
"""
from dataclasses import dataclass, field

import numpy as np

from . import _paths  # noqa: F401
from agents import Noop4DController

from .perturb import EpsilonRandomAgent, NextBestAgent, action_probs
from .config import SCENARIO
from .rollout import rollout_agent
from .score import SubmissionScore, aggregate

#: Default robustness curve, from the competition design.
EPSILONS = (0.05, 0.1, 0.25)
#: Rollouts per seed for the extraction search.
EXTRACTION_K = 8


@dataclass
class PostMortem:
    name: str
    deterministic: SubmissionScore
    passes: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    #: Whether the agent has an action distribution at all. A heuristic does not,
    #: so its "stochastic gap" is zero by construction rather than by merit —
    #: which matters when handing out a robustness trophy.
    has_distribution: bool = False
    is_baseline: bool = False

    @property
    def beats_the_brick(self) -> bool:
        brick = self.passes.get("do-nothing")
        if brick is None:
            raise KeyError("do-nothing baseline was not run")
        return self.deterministic.rank_key < brick.rank_key

    @property
    def stochastic_gap(self):
        """Congestion added by sampling instead of taking the argmax.

        Small means the policy's competence is not balanced on its argmax.
        """
        stochastic = self.passes.get("stochastic")
        if stochastic is None:
            return None
        return (stochastic.congestion_total
                - self.deterministic.congestion_total)

    @property
    def margin_reliance(self):
        """Congestion added by always taking the runner-up action."""
        next_best = self.passes.get("next-best")
        if next_best is None:
            return None
        return next_best.congestion_total - self.deterministic.congestion_total


def _score(agent, seeds, deterministic=True, scenario=SCENARIO) -> SubmissionScore:
    return aggregate([rollout_agent(agent, s, scenario,
                                    deterministic=deterministic)[0]
                      for s in seeds])


def best_extraction(agent, seeds, k=EXTRACTION_K, seed=0,
                    scenario=SCENARIO) -> SubmissionScore:
    """k stochastic rollouts per seed, keeping the best episode of each.

    Best is decided by the competition's own priorities — completion, then
    congestion, then the 4D exit miss, then clearances — rather than by return, so the
    search extracts the behaviour the leaderboard actually rewards.
    """
    def episode_key(kpis):
        return (not kpis.completed, kpis.congestion, kpis.exit_miss,
                kpis.clearances)

    # Sampling draws on torch's global RNG, so without this the "best of k"
    # differs run to run and the number cannot be quoted or reproduced.
    if getattr(agent, "policy", None) is not None:
        import torch
        torch.manual_seed(seed)

    best = []
    for s in seeds:
        attempts = [rollout_agent(agent, s, scenario, deterministic=False)[0]
                    for _ in range(k)]
        best.append(min(attempts, key=episode_key))
    return aggregate(best)


def run_postmortem(agent, seeds, name="submission", epsilons=EPSILONS,
                   k=EXTRACTION_K, rng_seed=0, verbose=False,
                   scenario=SCENARIO) -> PostMortem:
    from .agents import load_agent

    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    log("deterministic")
    has_dist = action_probs(agent, _first_obs(seeds[0], scenario)) is not None
    report = PostMortem(name=name, deterministic=_score(agent, seeds,
                                                       scenario=scenario),
                        has_distribution=has_dist,
                        is_baseline=isinstance(agent, Noop4DController))

    log("do-nothing baseline")
    report.passes["do-nothing"] = _score(load_agent("noop", scenario), seeds,
                                         scenario=scenario)

    log("stochastic")
    report.passes["stochastic"] = _score(agent, seeds, deterministic=False,
                                         scenario=scenario)

    if not has_dist:
        # A heuristic has no action distribution, so these two passes are not
        # undefined-but-skipped, they are meaningless for it. Say which and why.
        report.skipped["next-best"] = "agent exposes no action distribution"
        report.skipped["best-extraction"] = "no distribution to draw k samples from"
    else:
        log("next-best action")
        report.passes["next-best"] = _score(NextBestAgent(agent), seeds,
                                            scenario=scenario)
        log(f"best-extraction (k={k})")
        report.passes["best-extraction"] = best_extraction(agent, seeds, k=k,
                                                           seed=rng_seed,
                                                           scenario=scenario)

    for eps in epsilons:
        log(f"eps-random {eps}")
        perturbed = EpsilonRandomAgent(agent, eps,
                                       rng=np.random.default_rng(rng_seed))
        report.passes[f"eps={eps}"] = _score(perturbed, seeds,
                                             scenario=scenario)

    return report


def _first_obs(seed, scenario=SCENARIO):
    from .rollout import make_env
    env = make_env(scenario)
    obs, _ = env.reset(seed=seed)
    return obs


def format_report(report: PostMortem) -> str:
    det = report.deterministic
    header = (f"{'pass':<20}{'failed':>7}{'congestion':>12}{'exit miss':>12}"
              f"{'clearances':>12}{'touched':>9}")
    lines = [f"Post-mortem — {report.name}  ({det.n_episodes} episodes, "
             f"score v{det.version})", "", header, "-" * len(header)]

    def row(label, score):
        return (f"{label:<20}{score.failed_episodes:>7}"
                f"{score.congestion_total:>12}{score.exit_miss_mean:>12.4f}"
                f"{score.clearances_total:>12}{score.aircraft_involved_total:>9}")

    lines.append(row("deterministic", det))
    for label, score in report.passes.items():
        lines.append(row(label, score))
    for label, why in report.skipped.items():
        lines.append(f"{label:<20}{'— ' + why:>52}")

    lines.append("")
    brick = report.passes.get("do-nothing")
    if brick is not None:
        if report.is_baseline:
            lines.append("This agent IS the do-nothing baseline — it is the bar, "
                         "not a competitor against it.")
        elif report.beats_the_brick:
            margin = brick.congestion_total - det.congestion_total
            lines.append(f"Beats the do-nothing baseline by {margin} "
                         f"congestion events.")
        else:
            lines.append(f"DOES NOT BEAT THE DO-NOTHING BASELINE. "
                         f"Brick: {brick.congestion_total} congestion. "
                         f"This agent: {det.congestion_total}. An agent below "
                         f"this line is worse than not being there.")
    gap = report.stochastic_gap
    if gap is not None and report.has_distribution:
        lines.append(f"Sampling instead of argmax costs {gap:+d} congestion "
                     f"events — the smaller, the less the policy's competence "
                     f"balances on its argmax.")
    elif gap is not None:
        lines.append("No action distribution to sample from, so the stochastic "
                     "pass is the same pass and the gap is 0 by definition, "
                     "not by merit.")
    reliance = report.margin_reliance
    if reliance is not None:
        lines.append(f"Always taking the runner-up action costs {reliance:+d} "
                     f"— how much of the score rested on thin margins.")
    return "\n".join(lines)


# ---------------------------------------------------------------- trophies

def award_trophies(reports: dict) -> dict:
    """Compute the trophies that follow from the numbers.

    The Black Box (best QA writeup) and Icarus (boldest failure) awards are
    deliberately absent — they are jury calls, and pretending a metric decides
    them would be worse than admitting a human does.
    """
    if not reports:
        return {}
    awards = {}

    ranked = sorted(reports.items(), key=lambda kv: kv[1].deterministic.rank_key)
    awards["Golden Holding Pattern"] = ranked[0][0]

    perfect = [n for n, r in reports.items()
               if r.deterministic.congestion_total == 0
               and r.deterministic.failed_episodes == 0]
    if perfect:
        awards["Zero-Conflict Wings"] = sorted(
            perfect, key=lambda n: reports[n].deterministic.clearances_total)

    if perfect:
        awards["Minimal-Intervention"] = min(
            perfect, key=lambda n: reports[n].deterministic.clearances_total)

    # Only policies can earn this: a heuristic's stochastic gap is 0 because it
    # has no distribution to sample from, which is not robustness.
    gaps = {n: r.stochastic_gap for n, r in reports.items()
            if r.stochastic_gap is not None and r.has_distribution}
    if gaps:
        best = min(abs(g) for g in gaps.values())
        awards["Iron Stomach"] = sorted(n for n, g in gaps.items()
                                        if abs(g) == best)

    flew = [n for n, r in reports.items()
            if "do-nothing" in r.passes and not r.is_baseline
            and r.beats_the_brick]
    awards["The Brick with Wings"] = sorted(flew)

    return awards
