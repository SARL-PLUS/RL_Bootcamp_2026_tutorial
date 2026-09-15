"""Exercise 3/4 — benchmark checkpoints across injury severities.

Evaluates each given run on the healthy Ant, every 1-leg injury (4 variants)
and every 2-leg injury (6 variants), with identical episode seeds everywhere,
then reports the percentage drop relative to healthy.

Usage (from the "Crippled Ant" directory):
    # Ex 3 — zero-shot transfer of a healthy-trained policy:
    python scripts/transfer_benchmark.py --run healthy=runs/PPO_Ant-v5/<ts>

    # Ex 4 — compare zero-shot against a crippled-trained specialist:
    python scripts/transfer_benchmark.py \
        --run zero-shot=runs/PPO_Ant-v5/<ts> \
        --run specialist=runs/PPO_Ant-v5/<ts2> \
        --episodes 20

Writes benchmark.csv and benchmark.png to --out (default: benchmark/).
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.crippled_ant import N_LEGS
from utils.loading import (
    build_eval_env,
    eval_episodes,
    find_model_path,
    find_vecnormalize,
    load_model,
    load_run_config,
    vecnormalize_is_matched,
)


def scenarios():
    """(label, disabled_legs) pairs: healthy, all 1-leg, all 2-leg injuries."""
    yield "healthy", []
    for leg in range(N_LEGS):
        yield f"1 leg [{leg}]", [leg]
    for legs in itertools.combinations(range(N_LEGS), 2):
        yield f"2 legs {list(legs)}", list(legs)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="training run to benchmark; repeat for comparisons (Ex 4)",
    )
    p.add_argument("--model", default="best", help="'best', 'final' or a .zip path")
    p.add_argument("--episodes", type=int, default=20, help="episodes per scenario")
    p.add_argument("--seed", type=int, default=0, help="base episode seed")
    p.add_argument("--out", type=Path, default=Path("benchmark"))
    return p.parse_args()


def benchmark_run(label, run_dir, args):
    cfg = load_run_config(run_dir)
    model_path = find_model_path(run_dir, args.model)
    model = load_model(cfg, model_path)
    vecnorm = find_vecnormalize(model_path, Path(run_dir))
    if cfg.algo.normalize_obs and not vecnormalize_is_matched(model_path, vecnorm):
        print(f"  WARNING: no matching VecNormalize stats for {model_path.name}; "
              f"using {vecnorm.name if vecnorm else 'none'} — this understates the "
              "checkpoint. Prefer --model final.")
    rows = []
    for scenario, legs in scenarios():
        venv = build_eval_env(
            cfg,
            disabled_legs=legs,
            override_injury=True,
            vecnormalize_path=vecnorm,
        )
        returns = eval_episodes(model, venv, args.episodes, base_seed=args.seed)
        venv.close()
        rows.append(
            dict(
                model=label,
                scenario=scenario,
                n_disabled_legs=len(legs),
                mean_return=np.mean(returns),
                std_return=np.std(returns),
            )
        )
        print(f"  {scenario:<14} {np.mean(returns):8.1f} +/- {np.std(returns):.1f}")
    return rows


def main():
    args = parse_args()
    rows = []
    for spec in args.run:
        label, _, path = spec.rpartition("=")
        label = label or Path(path).parent.name
        print(f"[{label}] {path}")
        rows += benchmark_run(label, Path(path), args)

    df = pd.DataFrame(rows)
    # % drop vs. each model's own healthy score, averaged per severity
    healthy = df[df.scenario == "healthy"].set_index("model").mean_return
    df["drop_vs_healthy_pct"] = df.apply(
        lambda r: 100 * (1 - r.mean_return / healthy[r.model]), axis=1
    )

    summary = (
        df.groupby(["model", "n_disabled_legs"])[["mean_return", "drop_vs_healthy_pct"]]
        .mean()
        .round(1)
    )
    print("\n=== Mean return / % drop by injury severity ===")
    print(summary.to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "benchmark.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    models = df.model.unique()
    severities = sorted(df.n_disabled_legs.unique())
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        sub = df[df.model == m].groupby("n_disabled_legs").mean_return
        means = sub.mean()
        errs = sub.std().fillna(0)
        x = np.arange(len(severities)) + i * width
        ax.bar(x, means[severities], width, yerr=errs[severities], capsize=3, label=m)
    ax.set_xticks(np.arange(len(severities)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{s} disabled leg(s)" for s in severities])
    ax.set_ylabel("mean episode return")
    ax.set_title(f"Transfer benchmark ({args.episodes} episodes/scenario)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "benchmark.png", dpi=150)
    print(f"\nwrote {args.out}/benchmark.csv and benchmark.png")


if __name__ == "__main__":
    main()
