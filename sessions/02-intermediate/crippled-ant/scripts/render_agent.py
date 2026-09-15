"""Render episodes of any trained checkpoint — optionally under an injury it
never saw in training (transferability / robustness check by eye).

Point it at a checkpoint .zip (best_model.zip, final_model.zip, or a periodic
checkpoints/rl_model_*_steps.zip). The script walks up from the .zip to find
the run's saved Hydra config, rebuilds the training environment, loads the
matching VecNormalize statistics, records each episode as an .mp4 and prints
the returns.

Usage (from the "Crippled Ant" directory):
    python scripts/render_agent.py runs/PPO_Ant-v5/<ts>/best_model/best_model.zip
    python scripts/render_agent.py <zip> --disabled-legs 0      # healthy policy, injured ant
    python scripts/render_agent.py <zip> --n-random-legs 1 --episodes 3
    python scripts/render_agent.py <zip> --stochastic --seed 7
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.loading import (
    ALGOS,
    build_eval_env,
    eval_episodes,
    find_run_dir,
    find_vecnormalize,
    load_run_config,
    vecnormalize_is_matched,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="path to a model .zip")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="base seed; episode i uses seed+i")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--algo", choices=["PPO", "SAC"], default=None,
                   help="only needed if no run config is found next to the checkpoint")
    p.add_argument("--out", type=Path, default=None,
                   help="video directory (default: <run>/videos, else ./renders)")
    # Injury overrides: if any is given, the training config's injury is replaced.
    p.add_argument("--disabled-joints", type=int, nargs="*", default=None)
    p.add_argument("--disabled-legs", type=int, nargs="*", default=None)
    p.add_argument("--n-random-legs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    run_dir = find_run_dir(args.checkpoint)
    if run_dir is not None:
        cfg = load_run_config(run_dir)
        algo_name = args.algo or cfg.algo.name
    else:
        # Arbitrary .zip without a run directory: fall back to the shipped
        # defaults (healthy Ant, 27-dim obs) — injury flags still apply.
        if args.algo is None:
            sys.exit("no .hydra/config.yaml found above the checkpoint; pass --algo")
        from hydra import compose, initialize_config_dir

        conf_dir = str(Path(__file__).resolve().parents[1] / "conf")
        with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
            cfg = compose(config_name="config", overrides=[f"algo={args.algo.lower()}"])
        algo_name = args.algo
        print("note: no run config found — using shipped conf/ defaults for the env")

    vecnorm = find_vecnormalize(args.checkpoint, run_dir)
    if cfg.algo.normalize_obs:
        if vecnorm is None:
            print("WARNING: this algo trains with VecNormalize but no stats .pkl was "
                  "found — rendered behaviour will not match training performance!")
        elif not vecnormalize_is_matched(args.checkpoint, vecnorm):
            print(f"WARNING: {vecnorm.name} holds END-of-training statistics but "
                  f"{args.checkpoint.name} is a mid-training checkpoint — the policy "
                  "will see differently-scaled observations than it was scored with. "
                  "Prefer final_model.zip, or a run trained after this warning existed "
                  "(best_model/vecnormalize.pkl is saved alongside best_model now).")

    override = any(
        v is not None
        for v in (args.disabled_joints, args.disabled_legs, args.n_random_legs)
    )
    tag = [args.checkpoint.stem]
    if args.disabled_legs is not None:
        tag.append("legs" + "-".join(map(str, args.disabled_legs)))
    if args.disabled_joints is not None:
        tag.append("joints" + "-".join(map(str, args.disabled_joints)))
    if args.n_random_legs is not None:
        tag.append(f"rand{args.n_random_legs}")
    if args.stochastic:
        tag.append("stoch")

    video_dir = args.out or (run_dir / "videos" if run_dir else Path("renders"))
    venv = build_eval_env(
        cfg,
        disabled_joints=args.disabled_joints,
        disabled_legs=args.disabled_legs,
        n_random_legs=args.n_random_legs,
        override_injury=override,
        video_dir=video_dir,
        video_prefix="_".join(tag),
        video_episodes=args.episodes,
        vecnormalize_path=vecnorm,
    )
    model = ALGOS[algo_name].load(args.checkpoint, device="cpu")

    returns = eval_episodes(
        model,
        venv,
        n_episodes=args.episodes,
        base_seed=args.seed,
        deterministic=not args.stochastic,
    )
    venv.close()  # flushes the last video

    mode = "stochastic" if args.stochastic else "deterministic"
    for i, r in enumerate(returns):
        print(f"episode {i} (seed {args.seed + i}, {mode}): return {r:.1f}")
    print(f"mean {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"videos written to {video_dir}/")


if __name__ == "__main__":
    main()
