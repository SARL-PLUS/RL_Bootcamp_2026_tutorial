"""Submission I/O: the action-trace CSV the public leaderboard is built on.

Format is one row per (seed, step)::

    id,action
    s0000042_t00,0
    s0000042_t01,18

The action is the environment's **flat** action index: ``0`` idles, and
``1 + N_CLEARANCES * aircraft + clearance`` issues one clearance to one
aircraft. One row per step, because the environment commands at most one
aircraft per step — the old (seed, step, aircraft) grid described a simultaneous
command vector that no longer exists, and writing one would imply the sector can
be re-tasked all at once.

Every seed carries the **full horizon** of rows, idled out past the point where
the producing episode ended. Locally that padding is invisible — the replay
holds the sector with idle actions either way — but the Kaggle solution file is
exported at the full horizon and the metric rejects a submission that does not
cover it. An unpadded file was refused for *every* real agent, because every
real agent ends at least one episode early.

Long and boring on purpose. It is diffable, it is trivially validated, and there
is no aggregate number in it for anyone to inflate — the score comes from
re-flying the actions, not from reading a column.
"""
import csv
from pathlib import Path

import numpy as np

from . import _paths
from .config import SCENARIO

#: Read off the SCORED environment, so a submission that widened its own
#: action space cannot write indices the replay would refuse.
N_CLEARANCES = _paths.reference_envs().flight_4d.N_CLEARANCES

ID_FIELD, ACTION_FIELD = "id", "action"


def max_action(n_flights: int) -> int:
    """Highest legal flat index for a sector of ``n_flights``.

    ``1 + N_CLEARANCES * n_flights`` actions exist, so the last index is
    ``N_CLEARANCES * n_flights``.
    """
    return N_CLEARANCES * int(n_flights)


def row_id(seed: int, step: int) -> str:
    return f"s{seed:07d}_t{step:02d}"


def parse_row_id(row: str) -> tuple[int, int]:
    try:
        seed_s, step_s = row.split("_")
        return int(seed_s[1:]), int(step_s[1:])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"malformed submission id {row!r}") from exc


def write_submission(path: Path, traces: dict,
                     steps: int = SCENARIO.max_steps) -> int:
    """Write ``{seed: actions[steps]}``, idled out to ``steps``. Returns rows.

    A trace shorter than ``steps`` means that episode ended early; the remaining
    rows are written as the idle action so the file covers the whole scored
    horizon. Pass ``steps=None`` to write exactly what you were handed — only
    useful for testing the reader's own tolerance.
    """
    path = Path(path)
    rows = 0
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([ID_FIELD, ACTION_FIELD])
        for seed in sorted(traces):
            actions = np.asarray(traces[seed], dtype=np.int64)
            if actions.ndim != 1:
                raise ValueError(
                    f"seed {seed}: expected actions of shape [steps], "
                    f"got {actions.shape}")
            if steps is not None:
                if actions.size > steps:
                    raise ValueError(
                        f"seed {seed}: {actions.size} actions for a "
                        f"{steps}-step horizon")
                actions = np.pad(actions, (0, steps - actions.size))
            for step, value in enumerate(actions):
                writer.writerow([row_id(seed, step), int(value)])
                rows += 1
    return rows


def read_submission(path: Path, n_flights: int, max_steps: int) -> dict:
    """Read a submission back into ``{seed: actions[steps]}``.

    Missing cells are filled with the idle action rather than rejected: a short
    trace means the submitting run ended early, which the replay handles by
    holding the sector. A *malformed* file is still an error — silence there
    would turn a broken submission into a quietly bad score.
    """
    path = Path(path)
    limit = max_action(n_flights)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != [ID_FIELD, ACTION_FIELD]:
            raise ValueError(
                f"expected header ['{ID_FIELD}', '{ACTION_FIELD}'], "
                f"got {reader.fieldnames}")
        cells = {}
        for line_no, record in enumerate(reader, start=2):
            seed, step = parse_row_id(record[ID_FIELD])
            if not 0 <= step < max_steps:
                raise ValueError(f"line {line_no}: step {step} outside "
                                 f"[0, {max_steps})")
            try:
                action = int(record[ACTION_FIELD])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_no}: action "
                                 f"{record[ACTION_FIELD]!r} is not an integer") from exc
            if not 0 <= action <= limit:
                raise ValueError(f"line {line_no}: action {action} outside "
                                 f"[0, {limit}]")
            cells[(seed, step)] = action

    if not cells:
        raise ValueError(f"{path} contains no action rows")

    traces = {}
    for seed in sorted({s for s, _ in cells}):
        steps = [t for s, t in cells if s == seed]
        actions = np.zeros(max(steps) + 1, dtype=np.int64)
        for (s, t), value in cells.items():
            if s == seed:
                actions[t] = value
        traces[seed] = actions
    return traces
