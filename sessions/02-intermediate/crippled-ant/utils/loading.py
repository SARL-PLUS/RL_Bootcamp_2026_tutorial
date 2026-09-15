"""Load trained runs back for evaluation: config, model, normalisation stats.

A "run" is the directory written by scripts/train.py (it contains
.hydra/config.yaml). These helpers are shared by scripts/evaluate.py,
scripts/transfer_benchmark.py and the tests.
"""
import re
from pathlib import Path
from typing import Optional, Sequence

import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import make_ant

ALGOS = {"PPO": PPO, "SAC": SAC}


def load_run_config(run_dir: Path) -> DictConfig:
    cfg_path = Path(run_dir) / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"not a training run (no {cfg_path})")
    return OmegaConf.load(cfg_path)


def find_model_path(run_dir: Path, which: str = "best") -> Path:
    """Resolve 'best'/'final' (or an explicit .zip path) inside a run dir."""
    run_dir = Path(run_dir)
    candidates = {
        "best": run_dir / "best_model" / "best_model.zip",
        "final": run_dir / "final_model.zip",
    }
    path = candidates.get(which, Path(which))
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    return path


def find_run_dir(checkpoint: Path) -> Optional[Path]:
    """Walk up from a .zip until a directory containing .hydra/ is found."""
    for parent in Path(checkpoint).resolve().parents:
        if (parent / ".hydra" / "config.yaml").exists():
            return parent
    return None


def find_vecnormalize(checkpoint: Path, run_dir: Optional[Path]) -> Optional[Path]:
    """The normalisation statistics that belong to THIS checkpoint.

    A policy must be fed observations scaled the way they were when it was
    trained/scored. VecNormalize statistics move throughout training, so each
    checkpoint needs its own snapshot:

    - periodic checkpoints -> rl_model_vecnormalize_<n>_steps.pkl (written by
      CheckpointCallback with save_vecnormalize=True);
    - best_model.zip       -> best_model/vecnormalize.pkl (written by our
      SaveVecNormalize callback; SB3's EvalCallback does NOT save stats);
    - final_model.zip      -> <run>/vecnormalize.pkl.

    Falling back to the end-of-training stats for a mid-training checkpoint is
    a silent mismatch, so callers warn rather than pretend it is fine.
    """
    checkpoint = Path(checkpoint)

    m = re.match(r"(.+)_(\d+)_steps$", checkpoint.stem)
    if m:
        snapshot = checkpoint.parent / f"{m.group(1)}_vecnormalize_{m.group(2)}_steps.pkl"
        if snapshot.exists():
            return snapshot

    sibling = checkpoint.parent / "vecnormalize.pkl"   # best_model/
    if sibling.exists():
        return sibling

    if run_dir is not None and (run_dir / "vecnormalize.pkl").exists():
        return run_dir / "vecnormalize.pkl"
    return None


def vecnormalize_is_matched(checkpoint: Path, stats: Optional[Path]) -> bool:
    """False when `stats` are end-of-training stats for a mid-training model."""
    if stats is None:
        return False
    checkpoint, stats = Path(checkpoint), Path(stats)
    if stats.parent == checkpoint.parent:
        return True                                   # snapshot next to it
    return checkpoint.stem == "final_model"           # final <-> final


def build_eval_env(
    cfg: DictConfig,
    disabled_joints: Optional[Sequence[int]] = None,
    disabled_legs: Optional[Sequence[int]] = None,
    n_random_legs: Optional[int] = None,
    override_injury: bool = False,
    video_dir: Optional[Path] = None,
    video_prefix: str = "rl-video",
    video_episodes: Optional[int] = None,
    vecnormalize_path: Optional[Path] = None,
):
    """Single-env DummyVecEnv matching the training setup.

    By default the injury comes from the training config; pass
    ``override_injury=True`` to replace it (e.g. to test zero-shot transfer
    of a healthy-trained policy onto a crippled Ant). ``video_dir`` records
    every episode as an .mp4.
    """
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)
    if video_dir is not None:
        env_kwargs["render_mode"] = "rgb_array"
    if not override_injury:
        disabled_joints = list(cfg.env.disabled_joints)
        disabled_legs = list(cfg.env.disabled_legs)
        n_random_legs = cfg.env.n_random_legs

    def factory():
        env = make_ant(
            disabled_joints=disabled_joints,
            disabled_legs=disabled_legs,
            n_random_legs=n_random_legs,
            **env_kwargs,
        )
        if video_dir is not None:
            # cap the trigger so the auto-reset after the last evaluated
            # episode doesn't leave a stub video behind
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                episode_trigger=lambda ep: video_episodes is None or ep < video_episodes,
                name_prefix=video_prefix,
            )
        return env

    venv = DummyVecEnv([factory])
    if vecnormalize_path is not None and Path(vecnormalize_path).exists():
        venv = VecNormalize.load(str(vecnormalize_path), venv)
        venv.training = False
        venv.norm_reward = False
    return venv


def load_model(cfg: DictConfig, model_path: Path, device: str = "cpu"):
    return ALGOS[cfg.algo.name].load(model_path, device=device)


def eval_episodes(model, venv, n_episodes: int, base_seed: int = 0,
                  deterministic: bool = True) -> list[float]:
    """Evaluate over seeded episodes so different models/env variants see the
    exact same initial conditions (episode i always uses base_seed + i)."""
    returns = []
    for ep in range(n_episodes):
        venv.seed(base_seed + ep)
        obs = venv.reset()
        done, ep_return = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, _ = venv.step(action)
            done = bool(dones[0])
            # raw reward: build_eval_env sets norm_reward=False
            ep_return += float(reward[0])
        returns.append(ep_return)
    return returns
