"""The score function. We own this; students own their reward.

Three objectives in strict priority order — **safety, then timeliness, then
efficiency** — compared lexicographically. A submission that is safer outranks a
more punctual or cheaper one no matter by how much.

Timeliness is measured by the environment's 4D objective, ``exit_miss``: how far
each aircraft was from its exit gate in space and time, averaged over the sector.
Everything here is computed from ``step()``'s info dict, never from the student's
reward, so shaping the reward moves the leaderboard only through the behaviour it
produces.
"""
from dataclasses import dataclass, asdict, field

import numpy as np

from .config import SCENARIO, SCORE_VERSION, EXIT_MISS_BUCKET


@dataclass
class EpisodeKPIs:
    """One episode's contribution to the score."""
    seed: int
    steps: int
    completed: bool          # ran the full horizon rather than dying early
    congestion: int          # loss-of-separation pairs, summed over steps
    exit_miss: float         # mean 4D deviation at the exit gate, over aircraft
    clearances: int
    aircraft_involved: int
    invalids: int
    max_exit_miss: float     # the worst single aircraft, for diagnostics

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubmissionScore:
    """Aggregate over the whole seed list, plus the key that ranks it."""
    version: str
    n_episodes: int
    failed_episodes: int
    congestion_total: int
    congestion_per_episode: float
    zero_conflict_episodes: int
    exit_miss_mean: float
    clearances_total: int
    clearances_per_episode: float
    aircraft_involved_total: int
    invalids_total: int
    episodes: list = field(default_factory=list, repr=False)

    @property
    def rank_key(self) -> tuple:
        """Lexicographic sort key, every component lower-is-better.

        `failed_episodes` comes first and that placement is load-bearing. An
        episode that dies early accumulates fewer congestion events simply by
        existing for less time, so ranking on safety alone would reward crashing
        out. Failing to fly the sector is not a way to fly it safely.
        """
        return (
            self.failed_episodes,
            self.congestion_total,
            round(self.exit_miss_mean / EXIT_MISS_BUCKET),
            self.clearances_total,
            self.aircraft_involved_total,
        )

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("episodes")
        d["rank_key"] = self.rank_key
        return d


def episode_kpis(seed: int, steps: int, completed: bool, congestion: int,
                 final_info: dict, scenario=SCENARIO) -> EpisodeKPIs:
    """Assemble one episode's KPIs from the running congestion count and the
    final info dict (clearances, involvement and deviation are cumulative, so
    only their last value matters).

    ``max_exit_miss`` is rebuilt here from the per-aircraft deviations rather
    than read out of ``info``: the environment reports the fleet *mean* as the
    scored quantity, and the worst single aircraft is a diagnostic the harness
    wants beside it. The two must be in the same units, hence
    ``scenario.exit_time_scale`` — seconds late per flight level off plan.
    """
    dt = np.abs(np.asarray(final_info["dt_exit"], dtype=np.float64))
    d_alt = np.abs(np.asarray(final_info["d_alt_exit"], dtype=np.float64))
    per_aircraft = dt / scenario.exit_time_scale + d_alt
    return EpisodeKPIs(
        seed=seed,
        steps=steps,
        completed=completed,
        congestion=int(congestion),
        exit_miss=float(final_info["exit_miss"]),
        clearances=int(final_info["clearances"]),
        aircraft_involved=int(final_info["aircraft_involved"]),
        invalids=int(final_info["invalids"]),
        max_exit_miss=float(np.max(per_aircraft)),
    )


def aggregate(episodes: list[EpisodeKPIs]) -> SubmissionScore:
    if not episodes:
        raise ValueError("cannot score an empty episode list")
    n = len(episodes)
    return SubmissionScore(
        version=SCORE_VERSION,
        n_episodes=n,
        failed_episodes=sum(1 for e in episodes if not e.completed),
        congestion_total=sum(e.congestion for e in episodes),
        congestion_per_episode=sum(e.congestion for e in episodes) / n,
        zero_conflict_episodes=sum(1 for e in episodes if e.congestion == 0),
        exit_miss_mean=float(np.mean([e.exit_miss for e in episodes])),
        clearances_total=sum(e.clearances for e in episodes),
        clearances_per_episode=sum(e.clearances for e in episodes) / n,
        aircraft_involved_total=sum(e.aircraft_involved for e in episodes),
        invalids_total=sum(e.invalids for e in episodes),
        episodes=episodes,
    )


def rank(named_scores: dict) -> list[tuple]:
    """Sort ``{name: SubmissionScore}`` best-first. Returns (rank, name, score)."""
    ordered = sorted(named_scores.items(), key=lambda kv: kv[1].rank_key)
    return [(i + 1, name, score) for i, (name, score) in enumerate(ordered)]


def format_leaderboard(named_scores: dict) -> str:
    rows = rank(named_scores)
    head = (f"{'#':>2}  {'submission':<22}{'failed':>7}{'congestion':>12}"
            f"{'exit miss':>12}{'clearances':>12}{'touched':>9}")
    lines = [head, "-" * len(head)]
    for pos, name, s in rows:
        lines.append(
            f"{pos:>2}  {name:<22}{s.failed_episodes:>7}{s.congestion_total:>12}"
            f"{s.exit_miss_mean:>12.4f}{s.clearances_total:>12}"
            f"{s.aircraft_involved_total:>9}"
        )
    lines.append("")
    lines.append(f"ranked on safety -> timeliness -> efficiency "
                 f"(score v{SCORE_VERSION}, {rows[0][2].n_episodes} episodes)")
    return "\n".join(lines)
