#!/usr/bin/env python
"""Score policies on :mod:`envs.flight_4d` over a proper seed set.

    conda activate rlbootcamp
    python scripts/score_4d.py --agent noop --agent random --agent runs/v4d_s123/final_model.zip

Why this is not the training callback
-------------------------------------
The callback runs 20 episodes so it can run often. Twenty episodes could not
rank two agents in this project: on a 20-episode callback the macro run led its
control on both failures and congestion, and on 50 frozen seeds **the ranking
inverted**. The callback exists to tell you a run is alive, not which run is
better.

This scores 100 seeds by default, holds them fixed across agents, and ranks
lexicographically the way the competition does — **failures first**. That
ordering is load-bearing: an episode that ends in a mid-air accumulates fewer
conflict-steps purely by existing for less time, so ranking on congestion alone
makes crashing out a strategy.

Both scoring modes are reported. `idle_bias` makes idling the argmax at
initialisation, so a deterministic rollout of a weak policy is byte-identical to
doing nothing — three runs were written off as collapsed on exactly that
evidence and all three were still learning.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.flight_4d import Flight4DEnv, N_CLEARANCES                      # noqa: E402


def rollout(agent, seed: int, deterministic: bool, env_kwargs: dict) -> dict:
    env = Flight4DEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    while True:
        if agent == "noop":
            action = 0
        elif agent == "random":
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
        elif hasattr(agent, "predict"):
            action = int(agent.predict(obs, deterministic=deterministic)[0])
        else:
            raise TypeError(f"unusable agent: {agent!r}")
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    return info


def score(agent, seeds, deterministic: bool, env_kwargs: dict) -> dict:
    eps = [rollout(agent, s, deterministic, env_kwargs) for s in seeds]
    n = len(eps)
    return dict(
        failed=sum(int(e["collided"]) for e in eps),
        congestion=sum(e["congestion_total"] for e in eps),
        conflict_free=sum(int(e["congestion_total"] == 0) for e in eps) / n,
        exit_miss=float(np.mean([e["exit_miss"] for e in eps])),
        clearances=sum(e["clearances"] for e in eps),
        touched=sum(e["aircraft_involved"] for e in eps),
        n=n,
    )


def rank_key(s: dict) -> tuple:
    """Failures, then congestion, then the 4D miss (bucketed), then cost.

    The bucket exists for the same reason the competition's does: `exit_miss` is
    a float, so exact ties never happen and the efficiency terms would never
    break one.
    """
    return (s["failed"], s["congestion"], round(s["exit_miss"] / 0.01),
            s["clearances"], s["touched"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", action="append", default=None,
                   help="'noop', 'random', 'rule-based', or a checkpoint path")
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--seed0", type=int, default=900_000,
                   help="held away from the training callback's 100_000 block")
    p.add_argument("--hold-steps", type=int, default=5)
    args = p.parse_args()
    agents = args.agent or ["noop", "random"]

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    env_kwargs = dict(hold_steps=args.hold_steps)

    rows = []
    for spec in agents:
        if spec in ("noop", "random"):
            loaded, modes = spec, [True]
        elif spec == "rule-based":
            # The hand-written bar. Deterministic by construction, so one mode.
            from agents import Priority4DController
            loaded = Priority4DController(hold_steps=args.hold_steps)
            modes = [True]
        else:
            from stable_baselines3 import PPO
            loaded = PPO.load(spec, device="cpu")
            modes = [True, False]
        for det in modes:
            s = score(loaded, seeds, det, env_kwargs)
            # The RUN directory, not the file: every checkpoint is called
            # `final_model`, so `Path(spec).stem` labelled three different runs
            # identically and the table could not be read.
            name = (spec if spec in ("noop", "random", "rule-based")
                    else Path(spec).parent.name)
            if spec not in ("noop", "random", "rule-based"):
                name += "  [det]" if det else "  [sto]"
            rows.append((name, s))

    rows.sort(key=lambda r: rank_key(r[1]))
    print(f"\nFlight4DEnv  n+m=5, n~U(2,4)  |  {args.seeds} seeds "
          f"({args.seed0}..{args.seed0 + args.seeds - 1})\n")
    hdr = (f" {'#':>2}  {'agent':32s} {'failed':>7s} {'congestion':>11s} "
           f"{'cf':>6s} {'exit_miss':>10s} {'clearances':>11s} {'touched':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for i, (name, s) in enumerate(rows, 1):
        print(f" {i:>2}  {name:32s} {s['failed']:7d} {s['congestion']:11d} "
              f"{100*s['conflict_free']:5.0f}% {s['exit_miss']:10.3f} "
              f"{s['clearances']:11d} {s['touched']:8d}")
    print("\nranked on failures -> congestion -> exit_miss -> clearances -> touched")
    print("congestion is NOT comparable across rows with different failure counts:")
    print("an episode that ends early accumulates fewer conflict-steps by existing")
    print("for less time. Read it against the `failed` column, never alone.")


if __name__ == "__main__":
    main()
