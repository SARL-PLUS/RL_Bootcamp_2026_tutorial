"""Train PPO or SAC on the healthy or crippled MuJoCo Ant.

Usage (from the "Crippled Ant" directory):
    python scripts/train.py                          # Ex 1: PPO on healthy Ant-v5
    python scripts/train.py algo=sac                 # Ex 2: SAC on the same env
    python scripts/train.py env.disabled_legs=[0]    # Ex 4: specialist on crippled Ant
    python scripts/train.py env.n_random_legs=1      # Ex 5: domain randomisation
    python scripts/train.py -m +experiment=sweep_lr_batch   # Ex 6: hyperparameter grid

Each run writes to its own timestamped directory under runs/:
    .hydra/config.yaml   resolved configuration (read back by evaluate.py)
    tb/                  TensorBoard logs
    best_model/          best checkpoint according to EvalCallback
    checkpoints/         periodic checkpoints (+ VecNormalize stats)
    final_model.zip      model at the end of training
    vecnormalize.pkl     final normalisation statistics (if enabled)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import stable_baselines3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from envs import make_ant


class SaveVecNormalize(BaseCallback):
    """Save VecNormalize statistics whenever EvalCallback saves a new best.

    SB3's EvalCallback saves best_model.zip but not the normalisation
    statistics that produced its score. Pairing that checkpoint with the
    end-of-training stats feeds the policy differently-scaled observations
    than it was scored with — a silent performance loss. This keeps the two
    together.
    """

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            venv.save(str(self.save_path))
        return True


def make_train_env(cfg: DictConfig, seed: int, training: bool):
    """Vectorised (and optionally normalised) environment from the config."""
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)
    venv = make_vec_env(
        make_ant,
        n_envs=cfg.env.n_envs if training else 1,
        seed=seed,
        env_kwargs=dict(
            disabled_joints=list(cfg.env.disabled_joints),
            disabled_legs=list(cfg.env.disabled_legs),
            n_random_legs=cfg.env.n_random_legs,
            n_random_legs_max=cfg.env.n_random_legs_max,
            **env_kwargs,
        ),
    )
    if cfg.algo.normalize_obs or cfg.algo.normalize_reward:
        venv = VecNormalize(
            venv,
            training=training,
            norm_obs=cfg.algo.normalize_obs,
            norm_reward=cfg.algo.normalize_reward and training,
            gamma=cfg.algo.gamma,
        )
    return venv


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.train.seed
    set_random_seed(seed)
    env = make_train_env(cfg, seed=seed, training=True)
    # Evaluate on a separate env; EvalCallback keeps its VecNormalize
    # statistics in sync with the training env automatically.
    eval_env = make_train_env(cfg, seed=seed + 10_000, training=False)

    algo_kwargs = OmegaConf.to_container(cfg.algo, resolve=True)
    algo_cls = getattr(stable_baselines3, algo_kwargs.pop("name"))
    policy = algo_kwargs.pop("policy")
    policy_kwargs = algo_kwargs.pop("policy_kwargs")
    algo_kwargs.pop("normalize_obs"), algo_kwargs.pop("normalize_reward")
    # CLI overrides like algo.learning_rate=1e-4 can arrive as strings
    algo_kwargs["learning_rate"] = float(algo_kwargs["learning_rate"])

    model = algo_cls(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device=cfg.train.device,
        tensorboard_log=str(run_dir / "tb"),
        **algo_kwargs,
    )

    # SB3 callback frequencies count vec-env steps, so divide by n_envs to
    # express them in total environment steps as the config does.
    per_env = lambda freq: max(freq // cfg.env.n_envs, 1)
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            callback_on_new_best=SaveVecNormalize(
                run_dir / "best_model" / "vecnormalize.pkl"
            ),
            log_path=str(run_dir / "eval"),
            eval_freq=per_env(cfg.train.eval_freq),
            n_eval_episodes=cfg.train.n_eval_episodes,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=per_env(cfg.train.checkpoint_freq),
            save_path=str(run_dir / "checkpoints"),
            save_vecnormalize=True,
        ),
    ]

    model.learn(total_timesteps=cfg.train.total_timesteps, callback=callbacks)

    model.save(run_dir / "final_model")
    if isinstance(env, VecNormalize):
        env.save(str(run_dir / "vecnormalize.pkl"))

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg.train.n_eval_episodes, deterministic=True
    )
    print(f"Final eval: {mean_reward:.1f} +/- {std_reward:.1f}  ({run_dir})")
    env.close(), eval_env.close()
    return float(mean_reward)  # consumed by Hydra sweepers (e.g. Optuna)


if __name__ == "__main__":
    main()
