"""Load the thing being scored.

Four kinds of agent, one interface (SB3's ``predict``):

- ``noop``        — the brick. Everyone must out-fly the brick.
- ``random``      — uniform over the legal actions. What the action space
                    scores for free; nothing below it has learned anything.
- ``rule-based``  — the hand-written controller, the real bar to beat.
- a path to an SB3 ``.zip`` — a trained policy.

The same four names ``scripts/score_4d.py`` takes, so a row on the board and a
row here are the same agent.

Importing this module pulls in torch only when a checkpoint is actually loaded,
so the replay path stays dependency-light.
"""
from . import _paths  # noqa: F401
from agents import Noop4DController, Priority4DController, RandomLegalController

from .config import SCENARIO
from ._paths import reference_envs

N_CLEARANCES = reference_envs().flight_4d.N_CLEARANCES


#: The names that are not files.
BUILTIN = ("noop", "random", "rule-based")


def load_agent(spec: str, scenario=SCENARIO):
    """``noop`` | ``random`` | ``rule-based`` | path to an SB3 checkpoint."""
    if spec == "noop":
        return Noop4DController()
    if spec == "random":
        n_actions = 1 + N_CLEARANCES * scenario.n_flights
        return RandomLegalController(n_actions=n_actions)
    if spec == "rule-based":
        # The controller reads the observation and nothing else, but it cannot
        # recover the sector's shape from it — so the scenario is handed over
        # explicitly. A mismatch here is silent and expensive: the controller
        # would invert the observation's squashed features against the wrong
        # horizon and issue plausible-looking nonsense.
        return Priority4DController(n_flights=scenario.n_flights,
                                    max_steps=scenario.max_steps,
                                    hold_steps=scenario.hold_steps)

    from pathlib import Path
    path = Path(spec)
    if not (path.is_file() or path.with_suffix(".zip").is_file()):
        # SB3 would otherwise report "No such file: random.zip" for a typo in
        # a builtin name, which points at the wrong thing entirely.
        raise SystemExit(
            f"unknown agent {spec!r}: expected one of {', '.join(BUILTIN)}, "
            f"or a path to an SB3 .zip checkpoint")

    from stable_baselines3 import PPO  # imported lazily: torch is heavy
    return PPO.load(spec, device="cpu")
