#!/usr/bin/env python
"""Size the reward terms against each other, before spending a training run.

    conda activate rlbootcamp
    python scripts/calibrate_4d.py

Why this exists
---------------
The first weights on :mod:`envs.flight_4d` were chosen by taste and were badly wrong in a
way that was knowable in thirty seconds. ``w_safe=30`` against
``w_clearance=0.35`` is **86:1 in favour of acting**, and it buys that safety
every single step while a clearance is charged once. The trained policy duly
went to 26 clearances an episode and scored **worse than a random legal policy**
on the 4D objective.

Two asymmetries to check before every weight change:

1. **Per-step versus terminal.** All three cost terms are charged on every step;
   the mid-air charge is paid once and scales with the steps left unflown. Sizing
   a weight against it compares a rate to a lump sum.
2. **Per-event versus per-episode.** A conflict is worth ``w_safe`` *once*, but
   the clearance cost is paid *per clearance* — so the honest comparison is one
   conflict against the whole episode's clearance bill, not against one
   clearance.

The random-legal policy is the reference. Anything the trained agent does worse
than random on is not being priced, whatever the weight says.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.flight_4d import Flight4DEnv


def load_agent(spec: str, hold_steps: int):
    """Mirror of ``score_4d.py``'s dispatch, so both read the same string."""
    if spec in ("noop", "random"):
        return spec
    if spec == "rule-based":
        from agents import Priority4DController
        return Priority4DController(hold_steps=hold_steps)
    from stable_baselines3 import PPO
    return PPO.load(spec, device="cpu")


def measure(seeds: int, gamma: float, agent="random", deterministic: bool = True,
            **env_kwargs) -> dict:
    """Decompose the discounted return into its three cost terms.

    ``agent`` is ``"random"``, ``"noop"``, or anything exposing SB3's
    ``predict`` — the same objects :func:`load_agent` returns. The terms are
    read off ``env``'s own ``_severity``/``_deviation`` rather than re-derived,
    so a decomposition cannot drift from the reward it claims to decompose.
    """
    rows = []
    for seed in range(seeds):
        env = Flight4DEnv(**env_kwargs)
        obs, _ = env.reset(seed=seed)
        if hasattr(agent, "reset"):
            agent.reset()
        rng = np.random.default_rng(seed)
        safety = timely = action = 0.0
        step_r = []
        k = 0
        while True:
            if agent == "noop":
                act = 0
            elif agent == "random":
                act = int(rng.choice(np.flatnonzero(env.action_masks())))
            else:
                act = int(agent.predict(obs, deterministic=deterministic)[0])
            before = env.n_clearances
            obs, _r, term, trunc, info = env.step(act)
            g = gamma ** k
            safety += g * env.w_safe * env._severity()
            timely += g * (env.w_safe / env.n_flights) * float(env._deviation().sum())
            action += g * env.w_clearance * int(env.n_clearances > before)
            step_r.append(_r + env.collision_charge() if term else _r)
            k += 1
            if term or trunc:
                break
        worst = max(-sum(gamma ** (j - i) * step_r[j] for j in range(i, len(step_r)))
                    for i in range(len(step_r)))
        rows.append((k, env.n_clearances, safety, timely, action,
                     info["exit_miss"], info["congestion_total"], worst,
                     float(env.collided)))
    arr = np.array(rows)
    return dict(steps=arr[:, 0].mean(), clearances=arr[:, 1].mean(),
                safety=arr[:, 2].mean(), timely=arr[:, 3].mean(),
                action=arr[:, 4].mean(), exit_miss=arr[:, 5].mean(),
                congestion=arr[:, 6].mean(), worst_remaining=arr[:, 7].max(),
                failed=int(arr[:, 8].sum()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hold-steps", type=int, default=5)
    p.add_argument("--agent", action="append", default=None,
                   help="'noop', 'random', 'rule-based', or a checkpoint path; "
                        "repeatable. Adds a per-agent decomposition table below "
                        "the random-legal reference.")
    p.add_argument("--csv", type=Path, default=None,
                   help="also write the per-agent decomposition here")
    args = p.parse_args()

    env = Flight4DEnv()
    m = measure(args.seeds, args.gamma, hold_steps=args.hold_steps)

    print(f"random-legal policy, {args.seeds} seeds, gamma={args.gamma}\n")
    print(f"  episode length                    {m['steps']:8.1f} steps")
    print(f"  clearances issued                 {m['clearances']:8.1f}")
    print(f"  congestion                        {m['congestion']:8.1f}")
    print(f"  exit_miss                         {m['exit_miss']:8.3f}"
          "   <-- the bar a trained policy must beat")
    print()
    print("  discounted contribution of each term over a whole episode:")
    total = m["safety"] + m["timely"] + m["action"]
    for name, v in (("safety   (predicted severity)", m["safety"]),
                    ("timeliness (4D deviation)", m["timely"]),
                    ("action     (per clearance)", m["action"])):
        print(f"    {name:32s} {-v:10.2f}   {100*v/max(total,1e-9):5.1f}%")
    print()
    print(f"  one severity-1 infringement, per step   {env.w_safe:8.2f}")
    print(f"  whole fleet at max deviation, per step  "
          f"{env.w_safe / env.n_flights * env.n_flights:8.2f}   (equal, by design)")
    print(f"  one clearance                           {env.w_clearance:8.2f}")
    print()
    if m["action"] > m["safety"]:
        print("  WARNING: the action bill exceeds the safety bill. The policy will")
        print("           be reluctant to intervene at all.")
    if m["timely"] > m["safety"]:
        print("  WARNING: timeliness outweighs safety over an episode. Safety is")
        print("           supposed to be the dominant objective.")
    # Every term here is a cost, so ending early stops the meter. A terminal
    # charge below the cost of continuing pays the agent to crash.
    remaining = m["worst_remaining"]
    print(f"  mid-air charge at step 0                {env.collision_rate * env.w_safe * env.max_steps:8.2f}")
    print(f"  ... per remaining step                  {env.collision_rate * env.w_safe:8.2f}")
    print(f"  WORST cost of flying on, over all states {remaining:7.2f}")
    print(f"  (the mean is not the test: the distribution is skewed, and the")
    print(f"   agent only has to find ONE state where crashing is cheaper)")
    if env.collision_rate * env.w_safe * env.max_steps < remaining:
        print()
        print("  WARNING: dying is cheaper than flying on. Every term is a cost,")
        print("           so early termination stops the meter and the agent will")
        print("           learn to crash. Raise the terminal charge above the cost")
        print("           of continuing.")

    if not args.agent:
        return

    # --- per-agent decomposition -------------------------------------------
    # The same three terms, per policy, so "where does this agent's return
    # actually go?" is answerable without re-deriving anything. Costs are
    # printed POSITIVE (they are subtracted from the return) to keep the
    # columns readable; the CSV keeps the same convention.
    rows = []
    for spec in args.agent:
        agent = load_agent(spec, args.hold_steps)
        r = measure(args.seeds, args.gamma, agent, hold_steps=args.hold_steps)
        name = spec if spec in ("noop", "random", "rule-based") else Path(spec).stem
        rows.append((name, r))

    print(f"\n\n  per-agent decomposition, {args.seeds} seeds, "
          f"gamma={args.gamma}  (costs shown positive)\n")
    hdr = (f"    {'agent':22s} {'safety':>9s} {'timely':>9s} {'action':>9s} "
           f"{'total':>9s} {'failed':>7s} {'clear.':>7s} {'exit_miss':>10s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name, r in rows:
        tot = r["safety"] + r["timely"] + r["action"]
        print(f"    {name:22s} {r['safety']:9.2f} {r['timely']:9.2f} "
              f"{r['action']:9.2f} {tot:9.2f} {r['failed']:7d} "
              f"{r['clearances']:7.1f} {r['exit_miss']:10.3f}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["agent", "seeds", "gamma", "safety", "timely", "action",
                        "total", "failed", "steps", "clearances", "congestion",
                        "exit_miss"])
            for name, r in rows:
                w.writerow([name, args.seeds, args.gamma,
                            f"{r['safety']:.4f}", f"{r['timely']:.4f}",
                            f"{r['action']:.4f}",
                            f"{r['safety'] + r['timely'] + r['action']:.4f}",
                            r["failed"], f"{r['steps']:.2f}",
                            f"{r['clearances']:.2f}", f"{r['congestion']:.2f}",
                            f"{r['exit_miss']:.4f}"])
        print(f"\n  wrote {args.csv}")


if __name__ == "__main__":
    main()
