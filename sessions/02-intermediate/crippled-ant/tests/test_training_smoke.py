"""End-to-end smoke tests: the actual scripts, tiny budgets (marked slow).

Guarantees the whole student workflow — train, evaluate, benchmark — runs
before anyone stands in front of a classroom with it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]

pytest.importorskip("mujoco")
pytestmark = pytest.mark.slow


def run(script, *args):
    result = subprocess.run(
        [sys.executable, str(PKG / "scripts" / script), *args],
        cwd=PKG,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    return result


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("runs") / "smoke"
    run(
        "train.py",
        "train.total_timesteps=2048",
        "env.n_envs=2",
        "algo.n_steps=256",
        "algo.batch_size=64",
        "train.eval_freq=1024",
        "train.checkpoint_freq=2048",
        "train.n_eval_episodes=1",
        f"hydra.run.dir={run_dir}",
    )
    return run_dir


def test_train_writes_expected_artifacts(tiny_run):
    assert (tiny_run / "final_model.zip").exists()
    assert (tiny_run / "best_model" / "best_model.zip").exists()
    assert (tiny_run / "vecnormalize.pkl").exists()  # PPO default normalises
    assert (tiny_run / ".hydra" / "config.yaml").exists()


def test_evaluate_runs_on_trained_model(tiny_run):
    out = run("evaluate.py", "--run", str(tiny_run), "--episodes", "1").stdout
    assert "return" in out


def test_evaluate_with_injury_override(tiny_run):
    run("evaluate.py", "--run", str(tiny_run), "--episodes", "1",
        "--disabled-legs", "0", "--stochastic")


def test_transfer_benchmark(tiny_run, tmp_path):
    out_dir = tmp_path / "benchmark"
    run(
        "transfer_benchmark.py",
        "--run", f"smoke={tiny_run}",
        "--episodes", "1",
        "--out", str(out_dir),
    )
    assert (out_dir / "benchmark.csv").exists()
    assert (out_dir / "benchmark.png").exists()


def test_render_agent_writes_video_with_injury_override(tiny_run, tmp_path):
    out = tmp_path / "vids"
    result = run(
        "render_agent.py",
        str(tiny_run / "best_model" / "best_model.zip"),
        "--disabled-legs", "0",
        "--episodes", "1",
        "--out", str(out),
    )
    videos = list(out.glob("*.mp4"))
    assert len(videos) == 1, result.stdout  # exactly one video, no reset stub
    assert "legs0" in videos[0].name
    assert "return" in result.stdout


def test_sac_train_smoke(tmp_path):
    run_dir = tmp_path / "sac_smoke"
    run(
        "train.py",
        "algo=sac",
        "train.total_timesteps=300",
        "env.n_envs=1",
        "algo.learning_starts=100",
        "algo.buffer_size=10000",
        "train.eval_freq=200",
        "train.checkpoint_freq=300",
        "train.n_eval_episodes=1",
        f"hydra.run.dir={run_dir}",
    )
    assert (run_dir / "final_model.zip").exists()
    assert not (run_dir / "vecnormalize.pkl").exists()  # SAC: no VecNormalize
