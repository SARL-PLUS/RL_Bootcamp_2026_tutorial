"""Export the Kaggle solution file: every scored aircraft's initial state.

    python -m competition.kaggle.export --out solution.csv
    python -m competition.kaggle.export --private --out solution_private.csv

Two different meanings of "private" meet in this file, so keep them apart:
``--private`` exports the **secret seed list** (organisers only, never uploaded
as the public solution), while ``--holdout-frac`` marks part of whatever list you
exported as ``Usage=Private`` — Kaggle's own hidden split of the *public* seeds,
which is what makes its final leaderboard differ from its live one.

The solution carries one row per (seed, step) so it aligns row-for-row with the
submission, and the whole sector's initial state is repeated on every row of a
seed — one column group per aircraft, ``f0_x … f4_hdg``. The metric reads it
only from the ``t00`` rows; the repetition is alignment padding, not information.

Exporting the scenarios — instead of regenerating them inside the metric — is
what keeps the metric honest. A second copy of the scenario generator living in
the metric file would be free to drift from the real one.
"""
import argparse
import csv
from pathlib import Path

from ..config import SCENARIO
from ..rollout import make_env
from ..seeds import PRIVATE_SEEDS, PUBLIC_SEEDS, load_seeds
from ..submission import row_id

#: Per-aircraft state, in the order ``Flight.get_5_tuple`` returns it.
STATE_FIELDS = ["x", "y", "alt", "vel", "hdg"]


def state_columns(n_flights: int) -> list[str]:
    return [f"f{i}_{field}"
            for i in range(n_flights) for field in STATE_FIELDS]


def scenario_states(seeds, scenario=SCENARIO) -> dict:
    """``{seed: [(x, y, alt, vel, hdg), ...]}`` as the env stands after reset()."""
    env = make_env(scenario)
    states = {}
    for seed in seeds:
        env.reset(seed=seed)
        states[seed] = [tuple(float(v) for v in f.get_5_tuple())
                        for f in env.sim.flights]
    return states


def split_usage(seeds, holdout_frac: float) -> dict:
    """``{seed: "Public" | "Private"}``, the back ``holdout_frac`` held out.

    Deterministic — the back of the sorted list, not a sample — so the file can
    be regenerated identically without recording a split anywhere.
    """
    ordered = sorted(seeds)
    n_private = int(round(len(ordered) * holdout_frac))
    private = set(ordered[len(ordered) - n_private:]) if n_private else set()
    return {seed: ("Private" if seed in private else "Public")
            for seed in ordered}


def write_solution(path: Path, seeds, scenario=SCENARIO, usage="Public",
                   holdout_frac: float = 0.0) -> int:
    states = scenario_states(seeds, scenario)
    split = (split_usage(states, holdout_frac) if holdout_frac
             else {seed: usage for seed in states})
    rows = 0
    with Path(path).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id"] + state_columns(scenario.n_flights) + ["Usage"])
        for seed in sorted(states):
            # repr() and not str(): the replay has to reproduce the environment
            # bit for bit, and a rounded initial position would put a floor
            # under how closely it ever could.
            flat = [repr(v) for state in states[seed] for v in state]
            for step in range(scenario.max_steps):
                writer.writerow([row_id(seed, step)] + flat + [split[seed]])
                rows += 1
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="solution.csv")
    p.add_argument("--private", action="store_true",
                   help="export the secret seed list instead")
    p.add_argument("--usage", default="Public",
                   choices=["Public", "Private", "Ignored"],
                   help="Kaggle leaderboard split for every row. Ignored when "
                        "--holdout-frac is given")
    p.add_argument("--holdout-frac", type=float, default=0.0,
                   help="fraction of the exported seeds to mark Usage=Private, "
                        "giving Kaggle a hidden split of its own. 0.4 is a "
                        "reasonable default; 0 marks every row --usage")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    seeds = load_seeds(PRIVATE_SEEDS if args.private else PUBLIC_SEEDS)
    if args.limit:
        seeds = seeds[: args.limit]
    if not 0.0 <= args.holdout_frac < 1.0:
        raise SystemExit("--holdout-frac must be in [0, 1)")
    rows = write_solution(Path(args.out), seeds, usage=args.usage,
                          holdout_frac=args.holdout_frac)
    print(f"wrote {rows} rows for {len(seeds)} seeds to {args.out}")
    if args.holdout_frac:
        n_private = int(round(len(seeds) * args.holdout_frac))
        print(f"  {len(seeds) - n_private} seeds Public, {n_private} Private")
    if args.private:
        print("this file contains the SECRET seeds — do not upload it as the "
              "public solution")


if __name__ == "__main__":
    main()
