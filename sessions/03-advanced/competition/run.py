"""CLI: build a submission, replay one, or score several against each other.

    # roll an agent out on the public seeds and write a submission
    python -m competition.run submit --agent rule-based --out submission.csv

    # score a submission the way the public leaderboard does
    python -m competition.run replay --submission submission.csv

    # rank agents head to head (the reference bar)
    python -m competition.run compare --agent noop --agent rule-based

    # the private board: every submitted zip, on the secret seeds
    python -m competition.run --private score-zips --dir submissions/

    # turn a raw Kaggle leaderboard number back into failed/congestion/etc.
    python -m competition.run decode 40026885486301
"""
import argparse
import json
import sys
from pathlib import Path

from .config import SCENARIO, ScenarioConfig, SCORE_VERSION
from .score import aggregate, format_leaderboard
from .seeds import PRIVATE_SEEDS, PUBLIC_SEEDS, load_seeds


def _scenario(args):
    """The frozen scenario, unless the user asked for a different traffic mix.

    ``--n-lo/--n-hi`` exist for *development* — narrowing the per-episode draw
    of the converging traffic to see whether a policy can learn the easy case at
    all, and scoring it in the leaderboard's own units. The sector size stays
    fixed either way, so the spaces do not move and a checkpoint still loads. A
    run with either flag set is not comparable to the board and says so on every
    invocation.
    """
    lo, hi = SCENARIO.n_range
    lo = lo if args.n_lo is None else args.n_lo
    hi = hi if args.n_hi is None else args.n_hi
    if (lo, hi) == SCENARIO.n_range:
        return SCENARIO
    return ScenarioConfig(n_flights=SCENARIO.n_flights, n_range=(lo, hi),
                          max_steps=SCENARIO.max_steps,
                          hold_steps=SCENARIO.hold_steps)


def _seed_list(args):
    path = Path(args.seeds) if args.seeds else (
        PRIVATE_SEEDS if args.private else PUBLIC_SEEDS)
    seeds = load_seeds(path)
    return seeds[: args.limit] if args.limit else seeds


def cmd_submit(args):
    from .agents import load_agent
    from .rollout import rollout_agent
    from .submission import write_submission

    seeds = _seed_list(args)
    scenario = _scenario(args)
    agent = load_agent(args.agent, scenario)
    episodes, traces = [], {}
    for i, seed in enumerate(seeds, 1):
        kpis, actions = rollout_agent(agent, seed, scenario,
                                      deterministic=not args.stochastic)
        episodes.append(kpis)
        traces[seed] = actions
        if args.verbose:
            print(f"  [{i}/{len(seeds)}] seed {seed}: congestion={kpis.congestion} "
                  f"exit_miss={kpis.exit_miss:.4f} clearances={kpis.clearances}")

    rows = write_submission(Path(args.out), traces, scenario.max_steps)
    score = aggregate(episodes)
    print(f"\nwrote {rows} rows for {len(seeds)} seeds to {args.out}")
    print(json.dumps(score.summary(), indent=2))
    print("\nThis is your LOCAL score on the public seeds. The ranking that "
          "decides the trophies runs on seeds you have never seen.")


def cmd_replay(args):
    from .rollout import replay_actions
    from .submission import read_submission

    scenario = _scenario(args)
    traces = read_submission(Path(args.submission), scenario.n_flights,
                             scenario.max_steps)
    expected = set(_seed_list(args))
    missing = expected - set(traces)
    extra = set(traces) - expected
    if missing:
        raise SystemExit(f"submission is missing {len(missing)} required seeds, "
                         f"e.g. {sorted(missing)[:3]}")
    if extra and args.strict:
        raise SystemExit(f"submission contains {len(extra)} unexpected seeds")

    episodes = [replay_actions(traces[seed], seed, scenario)
                for seed in sorted(expected)]
    score = aggregate(episodes)
    print(json.dumps(score.summary(), indent=2))


def cmd_compare(args):
    from .agents import load_agent
    from .rollout import rollout_agent

    seeds = _seed_list(args)
    scenario = _scenario(args)
    scores = {}
    for spec in args.agent:
        agent = load_agent(spec, scenario)
        episodes = [rollout_agent(agent, s, scenario,
                                  deterministic=not args.stochastic)[0]
                    for s in seeds]
        scores[spec] = aggregate(episodes)
        if args.verbose:
            print(f"scored {spec}")
    print(format_leaderboard(scores))


def cmd_score_zips(args):
    """The private board. One zip, one process, one row.

    Every submission ships a directory called ``envs``; importing two of them
    into one interpreter would silently give the second team the first team's
    code. So each zip is scored by a fresh ``competition.ingest`` process and
    this function only collects what they print.
    """
    import subprocess
    from types import SimpleNamespace

    zips = sorted(Path(args.dir).glob("*.zip"))
    if not zips:
        raise SystemExit(f"no .zip files in {args.dir}")

    # Resolve the seed list here, not once per subprocess. A missing private
    # list is a host-side setup error, and reporting it fifteen times under a
    # heading about submissions would blame the teams for our own missing file.
    _seed_list(args)

    scores, rejected = {}, []
    for path in zips:
        cmd = [sys.executable, "-m", "competition.ingest", "--zip", str(path)]
        if args.private:
            cmd.append("--private")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if not args.no_contract:
            cmd.append("--contract")

        print(f"scoring {path.name} ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1])
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            rejected.append((path.name, (proc.stderr or proc.stdout or
                                         "no output").strip().splitlines()[-1]))
            continue

        name = result.get("team", path.stem)
        for warning in result.get("warnings", []):
            print(f"  ! {name}: {warning}")
        if not result.get("ok"):
            rejected.append((name, result.get("error", "unknown")))
            continue
        scores[name] = SimpleNamespace(**result["score"])

    print()
    if scores:
        print(format_leaderboard(scores))
    else:
        print("nothing scored.")
    if rejected:
        print("\nUNSCORED. The contract suite is the gate and we do not fix "
              "submissions on our side, but everything here is something the "
              "team can act on:")
        for name, why in rejected:
            print(f"  {name:<22} {why}")


def cmd_postmortem(args):
    from .agents import load_agent
    from .postmortem import award_trophies, format_report, run_postmortem

    seeds = _seed_list(args)
    scenario = _scenario(args)
    reports = {}
    for spec in args.agent:
        if args.verbose:
            print(f"post-mortem: {spec}")
        report = run_postmortem(load_agent(spec, scenario), seeds, name=spec,
                                k=args.extraction_k, verbose=args.verbose,
                                scenario=scenario)
        reports[spec] = report
        print()
        print(format_report(report))

    if len(reports) > 1:
        print("\n" + "=" * 60)
        print("TROPHIES")
        for trophy, winner in award_trophies(reports).items():
            print(f"  {trophy:<26} {winner}")
        print("\n  Black Box and Icarus are jury calls, not computed.")


def cmd_decode(args):
    """The packed score is not meant to be read, only sorted. This is the
    inverse of `kaggle.metric.pack`, for when someone squints at the
    leaderboard and asks what a number actually means.
    """
    from .kaggle.metric import EXIT_MISS_BUCKET, unpack

    parts = unpack(args.score)
    print(json.dumps(parts, indent=2))
    print(f"\napprox exit_miss_mean: "
          f"{parts['exit_miss_bucket'] * EXIT_MISS_BUCKET:.3f}")
    print("failed dominates congestion dominates exit_miss dominates "
          "clearances. A value sitting exactly on its BOUND_* constant "
          "(kaggle/metric.py) may be clamped, not measured.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default=None, help="explicit seed file")
    p.add_argument("--private", action="store_true",
                   help="use the secret seed list (organisers only)")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N seeds — for quick checks")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the argmax")
    p.add_argument("--n-lo", type=int, default=None,
                   help="override the lower end of the converging-traffic draw. "
                        "DEVELOPMENT ONLY — the leaderboard is frozen at "
                        f"n~U{SCENARIO.n_range}")
    p.add_argument("--n-hi", type=int, default=None,
                   help="override the upper end of the draw. DEVELOPMENT ONLY — "
                        f"the leaderboard is frozen at n~U{SCENARIO.n_range}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="roll an agent out and write submission.csv")
    s.add_argument("--agent", required=True, help="noop | rule-based | path/to/policy.zip")
    s.add_argument("--out", default="submission.csv")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("replay", help="score a submission.csv")
    s.add_argument("--submission", required=True)
    s.add_argument("--strict", action="store_true",
                   help="reject submissions carrying seeds outside the list")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("compare", help="rank several agents head to head")
    s.add_argument("--agent", action="append", required=True)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("score-zips",
                       help="score every submitted zip in a directory")
    s.add_argument("--dir", default="submissions",
                   help="directory of submission zips")
    s.add_argument("--no-contract", action="store_true",
                   help="skip the contract suite — for debugging only")
    s.set_defaults(func=cmd_score_zips)

    s = sub.add_parser("postmortem",
                       help="every diagnostic pass on one or more agents")
    s.add_argument("--agent", action="append", required=True)
    s.add_argument("--extraction-k", type=int, default=8,
                   help="stochastic rollouts per seed for the extraction search")
    s.set_defaults(func=cmd_postmortem)

    s = sub.add_parser("decode",
                       help="turn a raw Kaggle score back into its four KPIs")
    s.add_argument("score", type=float, help="the packed score, e.g. off the leaderboard")
    s.set_defaults(func=cmd_decode)

    args = p.parse_args()
    scenario = _scenario(args)
    print(f"Flight Challenge scoring v{SCORE_VERSION} — "
          f"{scenario.n_flights} aircraft, n~U{scenario.n_range}, "
          f"max_steps={scenario.max_steps}")
    if scenario is not SCENARIO:
        print(f"*** OFF-LEADERBOARD: the ranked scenario draws "
              f"n~U{SCENARIO.n_range}. These numbers are for development "
              f"only. ***")
    print()
    args.func(args)


if __name__ == "__main__":
    main()
