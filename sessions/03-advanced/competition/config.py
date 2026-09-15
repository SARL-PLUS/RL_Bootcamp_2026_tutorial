"""The frozen competition setup.

Every number here is part of the contract. Changing one after submissions open
invalidates the leaderboard, so they live in one file, versioned, rather than
scattered across CLI defaults.
"""
from dataclasses import dataclass, asdict

#: Bumped whenever the score function or the scenario changes. Recorded in every
#: report so a number can always be traced back to the rules that produced it.
#:
#: 2.0.0 — the scored environment became ``Flight4DEnv``: one aircraft commanded
#: per step from a flat ``Discrete`` action space, a per-episode draw of the
#: converging traffic, and the 4D exit gate (``exit_miss``) as the second
#: objective in place of the old along-track timeliness. No v1 number is
#: comparable to a v2 one.
SCORE_VERSION = "2.0.0"

#: The simulator's clock step, in seconds. Mirrors
#: ``CollisionCourseSimulator.DT``; restated rather than imported so this module
#: stays a plain description of the contract with no dependency on the
#: environment package. ``test_scoring.py`` asserts the two agree.
SIM_DT = 2


@dataclass(frozen=True)
class ScenarioConfig:
    """The traffic every submission is scored on.

    ``n_flights`` is the whole sector and never changes: it fixes the
    observation and action shapes, so a checkpoint trained on the scored
    scenario loads against it. What varies per episode is the **split** — ``n``
    aircraft on a collision course, drawn uniformly from ``n_range``, and
    ``n_flights - n`` of background traffic. That is the difficulty knob, and it
    moves without touching either space.
    """
    n_flights: int = 5
    n_range: tuple[int, int] = (2, 4)
    max_steps: int = 50
    #: Steps a temporary manoeuvre holds before it reverts. Frozen here because
    #: the scored ``exit_miss`` is measured in units of it.
    hold_steps: int = 5

    def as_env_kwargs(self) -> dict:
        """Keyword arguments for ``Flight4DEnv``. Every key is one of its own."""
        return asdict(self)

    @property
    def exit_time_scale(self) -> float:
        """Seconds of lateness that count as one full flight level of miss.

        ``exit_miss`` adds a temporal and a vertical deviation, so the two need
        a common unit: one hold's worth of delay weighs the same as one level
        off plan. The environment uses ``hold_steps * DT`` for exactly this and
        the harness must use the same number or its own derived KPIs drift from
        the one the leaderboard ranks on.
        """
        return float(self.hold_steps * SIM_DT)


SCENARIO = ScenarioConfig()

#: The 4D exit miss is a float, so exact ties never happen and efficiency would
#: never break one. Rounding to this resolution restores real ties. 0.005 is
#: about half a percent of a typical episode's miss — finer than anyone can
#: control for, coarse enough that two genuinely comparable submissions land in
#: the same bucket.
EXIT_MISS_BUCKET = 0.005
