"""Train the autoregressive policy on :mod:`envs.flight_4d`.

    conda activate rlbootcamp
    OMP_NUM_THREADS=1 python scripts/train_4d.py --steps 1500000 --out runs/v4d_s123

``OMP_NUM_THREADS=1`` is not optional when running more than one job. Torch
otherwise opens ~46 threads per process; two such processes on 16 cores measured
**31-39 fps** against **428 fps** pinned — an 11x difference that looks exactly
like a slow environment.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                                CheckpointCallback)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.flight_4d import (Flight4DEnv, N_CLEARANCES, N_FEATS,
                            N_GLOBALS)
from policies import AutoregressivePolicy
from callbacks_4d import AutoEntropyCallback, CurriculumCallback4D


def make_env(seed: int, **env_kwargs):
    def _f():
        env = Monitor(Flight4DEnv(**env_kwargs))
        env.reset(seed=seed)
        return env
    return _f


class KPICallback(BaseCallback):
    """Score on the KPIs, never on the training curve.

    Two runs in this project's history were called "learning" from a rising
    ``ep_len_mean`` while the policy had in fact collapsed to inaction. The
    training reward contains shaping the leaderboard does not; only these
    numbers are comparable to a baseline.
    """

    def __init__(self, every: int, episodes: int, env_kwargs: dict, verbose=0):
        super().__init__(verbose)
        self.every, self.episodes, self.env_kwargs = every, episodes, env_kwargs
        self._next = 0

    def _rollout(self, deterministic=True):
        env = Flight4DEnv(**self.env_kwargs)
        cong = miss = clr = 0.0
        failed = conflict_free = 0
        for k in range(self.episodes):
            obs, _ = env.reset(seed=100_000 + k)
            while True:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, _, term, trunc, info = env.step(int(action))
                if term or trunc:
                    break
            cong += info["congestion_total"]
            miss += info["exit_miss"]
            clr += info["clearances"]
            failed += int(info["collided"])
            conflict_free += int(info["congestion_total"] == 0)
        n = self.episodes
        return dict(congestion=cong, exit_miss=miss / n, clearances=clr,
                    failed=failed, conflict_free=conflict_free / n)

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.every
        # BOTH modes, always. `idle_bias` makes idling the argmax at
        # initialisation, so a deterministic rollout of an untrained policy is
        # byte-identical to doing nothing. Three runs in this project were
        # written off as "collapsed" on exactly that evidence and all three were
        # still learning. A scoring mode is part of the measurement.
        det = self._rollout(deterministic=True)
        sto = self._rollout(deterministic=False)
        for tag, k in (("det", det), ("sto", sto)):
            print(f"[eval] {self.num_timesteps:9d} steps [{tag}] "
                  f"| congestion {k['congestion']:5.0f} "
                  f"| conflict-free {100*k['conflict_free']:4.0f}% "
                  f"| failed {k['failed']:2d} "
                  f"| clearances {k['clearances']:5.0f} "
                  f"| exit_miss {k['exit_miss']:.3f}", flush=True)
            for key, v in k.items():
                self.logger.record(f"eval_{tag}/{key}", v)
        return True


def warmup_cosine(lr0: float, lr1: float, warmup_frac: float):
    def f(progress_remaining: float) -> float:
        done = 1.0 - progress_remaining
        if done < warmup_frac:
            return lr0 * done / max(warmup_frac, 1e-9)
        t = (done - warmup_frac) / max(1.0 - warmup_frac, 1e-9)
        return lr1 + 0.5 * (lr0 - lr1) * (1.0 + math.cos(math.pi * t))
    return f


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_500_000)
    p.add_argument("--out", default="runs/v4d")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--final-lr", type=float, default=3e-5)
    p.add_argument("--warmup-frac", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--idle-bias", type=float, default=2.5,
                   help="p(idle) = e^b/(e^b + n_flights); 2.5 gives ~0.71")
    # Every environment knob defaults to None and falls through to
    # Flight4DEnv's own default. Duplicating the numbers here has now silently
    # overridden the env TWICE — once for `hold_steps` and once for the reward
    # weights, each time launching a run that was not the run intended. One
    # source of truth, and the trainer only speaks when explicitly told to.
    p.add_argument("--hold-steps", type=int, default=None)
    p.add_argument("--w-safe", type=float, default=None)
    p.add_argument("--w-clearance", type=float, default=None)
    p.add_argument("--eval-freq", type=int, default=250_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--subproc", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=500_000,
                   help="timesteps between checkpoints. 0 disables. A 4M-step "
                        "run is 75+ minutes; without this an interruption at "
                        "3.9M leaves nothing on disk")
    # --- the settings a participant actually chooses between -----------------
    # Both default OFF, so the plain command is still the plain command and the
    # curriculum/entropy questions are ones you opt into having an opinion on.
    p.add_argument("--curriculum", action="store_true",
                   help="start on thin traffic (n pinned at --curriculum-n-start) "
                        "and widen the per-episode draw of n to the environment's "
                        "own n_range over --curriculum-frac of training. n+m is "
                        "fixed either way, so the spaces never change")
    p.add_argument("--curriculum-n-start", type=int, default=2)
    p.add_argument("--curriculum-frac", type=float, default=0.2,
                   help="fraction of training spent ramping. The rest is spent "
                        "at the scored density")
    p.add_argument("--auto-entropy", action="store_true",
                   help="tune ent_coef so the policy's entropy tracks a target "
                        "fraction of ln(n_actions), instead of holding --ent-coef "
                        "fixed. A fixed coefficient has failed in both directions "
                        "on this environment: too high inflates the clearance "
                        "rate until the budget truncates, too low collapses to idle")
    p.add_argument("--auto-entropy-target", type=float, default=0.35,
                   help="target entropy as a fraction of the maximum, ln(n_actions)")
    args = p.parse_args()

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {k: v for k, v in
                  dict(hold_steps=args.hold_steps, gamma=args.gamma,
                       w_safe=args.w_safe, w_clearance=args.w_clearance).items()
                  if v is not None}
    probe = Flight4DEnv(**env_kwargs)
    n_flights = probe.n_flights

    # Print what the ENVIRONMENT ended up with, never what the CLI asked for.
    # The header is the record of what actually ran.
    print(f"[env] Flight4DEnv n+m={n_flights} n~U(2,4) hold={probe.hold_steps} "
          f"w_safe={probe.w_safe} w_clr={probe.w_clearance} "
          f"coll_rate={probe.collision_rate} gamma={probe.gamma}", flush=True)
    print(f"[policy] autoregressive (aircraft -> clearance), masked, "
          f"hidden={args.hidden} idle_bias={args.idle_bias}", flush=True)

    maker = SubprocVecEnv if args.subproc else DummyVecEnv
    venv = maker([make_env(args.seed + i, **env_kwargs) for i in range(args.n_envs)])

    model = PPO(
        AutoregressivePolicy, venv,
        policy_kwargs=dict(n_flights=n_flights, n_feats=N_FEATS,
                           n_globals=N_GLOBALS, n_clearances=N_CLEARANCES,
                           hidden=args.hidden, idle_bias=args.idle_bias),
        learning_rate=warmup_cosine(args.lr, args.final_lr, args.warmup_frac),
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=0.95, clip_range=0.2,
        ent_coef=args.ent_coef, target_kl=args.target_kl,
        seed=args.seed, verbose=1, device="cpu",
        tensorboard_log=str(out / "tb"),
    )
    callbacks = [KPICallback(args.eval_freq, args.eval_episodes, env_kwargs)]
    if args.checkpoint_every > 0:
        # SB3 counts _on_step calls, one per vectorised step, so the divisor
        # converts the interval from timesteps into calls.
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(out / "checkpoints"), name_prefix="v4d"))
    if args.curriculum:
        callbacks.append(CurriculumCallback4D(
            args.steps, n_start=args.curriculum_n_start,
            ramp_frac=args.curriculum_frac, verbose=1))
    if args.auto_entropy:
        callbacks.append(AutoEntropyCallback(
            target_frac=args.auto_entropy_target, verbose=1))

    print(f"[train] steps={args.steps} n_envs={args.n_envs} "
          f"curriculum={args.curriculum} auto_entropy={args.auto_entropy} "
          f"target_kl={args.target_kl} ckpt_every={args.checkpoint_every}",
          flush=True)
    model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks))
    model.save(out / "final_model")
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
