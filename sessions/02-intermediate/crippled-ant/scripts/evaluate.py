"""Evaluate a trained run, optionally on a different injury than it saw in training.

Usage (from the "Crippled Ant" directory):
    python scripts/evaluate.py --run runs/PPO_Ant-v5/<timestamp>
    python scripts/evaluate.py --run <run> --model final --episodes 20
    python scripts/evaluate.py --run <run> --disabled-legs 0        # zero-shot transfer
    python scripts/evaluate.py --run <run> --stochastic             # sample the policy
    python scripts/evaluate.py --run <run> --render                 # save an .mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.loading import (
    build_eval_env,
    eval_episodes,
    find_model_path,
    find_vecnormalize,
    load_model,
    load_run_config,
    vecnormalize_is_matched,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="training run directory")
    p.add_argument("--model", default="best", help="'best', 'final' or a .zip path")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="base seed; episode i uses seed+i")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--render", action="store_true", help="record videos to <run>/videos/")
    # Injury overrides: if any is given, the training config's injury is replaced.
    p.add_argument("--disabled-joints", type=int, nargs="*", default=None)
    p.add_argument("--disabled-legs", type=int, nargs="*", default=None)
    p.add_argument("--n-random-legs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_run_config(args.run)
    model_path = find_model_path(args.run, args.model)

    override = any(
        v is not None
        for v in (args.disabled_joints, args.disabled_legs, args.n_random_legs)
    )
    vecnorm = find_vecnormalize(model_path, args.run)
    if cfg.algo.normalize_obs and not vecnormalize_is_matched(model_path, vecnorm):
        print(f"WARNING: no matching VecNormalize stats for {model_path.name}; "
              f"falling back to {vecnorm.name if vecnorm else 'none'} — scores will "
              "understate this checkpoint. Prefer --model final.")
    venv = build_eval_env(
        cfg,
        disabled_joints=args.disabled_joints,
        disabled_legs=args.disabled_legs,
        n_random_legs=args.n_random_legs,
        override_injury=override,
        video_dir=args.run / "videos" if args.render else None,
        vecnormalize_path=vecnorm,
    )
    model = load_model(cfg, model_path)
    returns = eval_episodes(
        model,
        venv,
        n_episodes=args.episodes,
        base_seed=args.seed,
        deterministic=not args.stochastic,
    )
    mode = "stochastic" if args.stochastic else "deterministic"
    print(
        f"{cfg.algo.name} {model_path.name} | {args.episodes} episodes ({mode}) | "
        f"return {np.mean(returns):.1f} +/- {np.std(returns):.1f} "
        f"(min {np.min(returns):.1f}, max {np.max(returns):.1f})"
    )
    venv.close()


if __name__ == "__main__":
    main()
