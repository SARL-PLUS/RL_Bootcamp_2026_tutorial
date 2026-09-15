"""The shipped Hydra configs must always compose — a broken YAML would only
surface mid-session otherwise."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from utils.loading import find_vecnormalize, vecnormalize_is_matched

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


def compose_cfg(*overrides):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def test_default_config_composes():
    cfg = compose_cfg()
    assert cfg.algo.name == "PPO"
    assert cfg.env.id == "Ant-v5"
    assert cfg.env.kwargs.include_cfrc_ext_in_observation is False


def test_sac_config_composes():
    cfg = compose_cfg("algo=sac")
    assert cfg.algo.name == "SAC"
    assert cfg.algo.normalize_obs is False  # off-policy: no VecNormalize


def test_injury_overrides():
    cfg = compose_cfg("env.disabled_legs=[0,2]", "env.n_random_legs=null")
    assert list(cfg.env.disabled_legs) == [0, 2]


def test_n_random_legs_max_override_composes():
    cfg = compose_cfg("env.n_random_legs_max=4")
    assert cfg.env.n_random_legs_max == 4
    assert cfg.env.n_random_legs is None  # the default; not this exercise's job to change


def test_sweep_experiment_composes():
    cfg = compose_cfg("+experiment=sweep_lr_batch")
    assert cfg.train.total_timesteps == 300_000


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_learning_rate_is_numeric(algo):
    cfg = compose_cfg(f"algo={algo}")
    assert float(cfg.algo.learning_rate) > 0


# --- VecNormalize statistics must follow their checkpoint ---------------
# A policy fed observations scaled differently from training silently
# underperforms, so pairing rules get their own tests.


def _fake_run(tmp_path):
    (tmp_path / ".hydra").mkdir()
    (tmp_path / ".hydra" / "config.yaml").write_text("algo: {}\n")
    (tmp_path / "vecnormalize.pkl").touch()
    (tmp_path / "final_model.zip").touch()
    (tmp_path / "best_model").mkdir()
    (tmp_path / "best_model" / "best_model.zip").touch()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "rl_model_1000_steps.zip").touch()
    return tmp_path


def test_final_model_pairs_with_run_level_stats(tmp_path):
    run = _fake_run(tmp_path)
    stats = find_vecnormalize(run / "final_model.zip", run)
    assert stats == run / "vecnormalize.pkl"
    assert vecnormalize_is_matched(run / "final_model.zip", stats)


def test_best_model_prefers_its_own_snapshot(tmp_path):
    run = _fake_run(tmp_path)
    snapshot = run / "best_model" / "vecnormalize.pkl"
    snapshot.touch()
    stats = find_vecnormalize(run / "best_model" / "best_model.zip", run)
    assert stats == snapshot
    assert vecnormalize_is_matched(run / "best_model" / "best_model.zip", stats)


def test_best_model_without_snapshot_is_flagged_as_mismatched(tmp_path):
    # legacy runs: EvalCallback saved best_model.zip but no stats with it
    run = _fake_run(tmp_path)
    ckpt = run / "best_model" / "best_model.zip"
    stats = find_vecnormalize(ckpt, run)
    assert stats == run / "vecnormalize.pkl"      # only fallback available
    assert not vecnormalize_is_matched(ckpt, stats)  # …and callers must warn


def test_periodic_checkpoint_uses_its_step_snapshot(tmp_path):
    run = _fake_run(tmp_path)
    ckpt = run / "checkpoints" / "rl_model_1000_steps.zip"
    snapshot = run / "checkpoints" / "rl_model_vecnormalize_1000_steps.pkl"
    snapshot.touch()
    stats = find_vecnormalize(ckpt, run)
    assert stats == snapshot
    assert vecnormalize_is_matched(ckpt, stats)
