"""The private board: a submitted zip, scored on seeds it has never seen.

The property under test everywhere here is the same one: **only the action index
crosses the boundary**. A team's environment decides what their policy sees; it
never decides what they score. That is what makes it safe to run student code
against a ranking, and it is worth a test that says so out loud.
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from competition import _paths  # noqa: F401,E402
from competition.config import SCENARIO  # noqa: E402
from competition.ingest import (MANIFEST, Submission, SubmissionError,  # noqa: E402
                                _resolve, load_submission, read_manifest, unpack)
from competition.rollout import make_env, rollout_submission  # noqa: E402
from envs.flight_4d import Flight4DEnv  # noqa: E402


class FixedPolicy:
    """Plays a fixed cycle of actions. Deterministic, and free of torch."""

    def __init__(self, actions):
        self.actions, self.i = list(actions), 0

    def predict(self, obs, deterministic=True):
        action = self.actions[self.i % len(self.actions)]
        self.i += 1
        return np.array(action), None


class MangledReward(Flight4DEnv):
    """A legal redesign: same dynamics, wildly different reward."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward * 1000.0 - 7.0, terminated, truncated, info


class GivesUpEarly(Flight4DEnv):
    """Terminates at step 3. Ending early must not stop the scored clock."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated or self.step_idx >= 3, truncated, info


def _submission(env, policy, **kwargs):
    """A Submission with its env and policy pre-loaded — no zip, no torch."""
    sub = Submission(team="t", root=Path("."), model_path=Path("x"), **kwargs)
    sub._observed = env
    sub._policy = policy
    return sub


def _actions():
    """A trace that actually issues clearances rather than idling through."""
    return [0, 3, 0, 0, 11, 0, 0, 0, 8, 0, 0, 17, 0, 0]


# ----------------------------------------------------------- the boundary

def test_a_redesigned_reward_cannot_move_the_score():
    """The whole design in one assertion. Teams may shape the reward however
    they like — it reaches the leaderboard only through the behaviour it
    produces, and here the behaviour is pinned, so the score must not move."""
    stock = rollout_submission(
        _submission(make_env(), FixedPolicy(_actions())), seed=4242)[0]
    theirs = rollout_submission(
        _submission(MangledReward(**SCENARIO.as_env_kwargs()),
                    FixedPolicy(_actions())), seed=4242)[0]
    assert theirs == stock


def test_their_env_giving_up_does_not_shorten_the_scored_episode():
    """An episode that ends early accumulates less congestion purely by being
    shorter. That is why `failed_episodes` sorts first, and it would be a hole
    if a student env could end the *scored* episode by ending its own."""
    honest = rollout_submission(
        _submission(make_env(), FixedPolicy(_actions())), seed=4242)[0]
    quitter = rollout_submission(
        _submission(GivesUpEarly(**SCENARIO.as_env_kwargs()),
                    FixedPolicy(_actions())), seed=4242)[0]

    assert quitter.steps > 3, "the scored env stopped when theirs did"
    assert quitter.steps == honest.steps
    assert quitter.completed == honest.completed


def test_the_scored_env_is_ours_even_when_they_ship_one():
    """`observed_env` is theirs; the KPIs come from a second env we build."""
    sub = _submission(MangledReward(**SCENARIO.as_env_kwargs()),
                      FixedPolicy(_actions()))
    assert isinstance(sub.observed_env(), MangledReward)
    kpis, actions = rollout_submission(sub, seed=7)
    assert kpis.seed == 7 and len(actions) > 0


# ------------------------------------------------------------ unpacking

def _write_zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


MANIFEST_OK = "team: brick-with-wings\nalgo: PPO\ndeterministic: true\n"


def test_unpack_finds_the_manifest_at_either_depth(tmp_path):
    """`zip -r sub.zip .` puts files at the top; zipping a folder puts them one
    level down. Both are common and neither is worth failing a team over."""
    for name, files in (
        ("flat", {MANIFEST: MANIFEST_OK, "model/policy.zip": "x"}),
        ("nested", {f"sub/{MANIFEST}": MANIFEST_OK, "sub/model/policy.zip": "x"}),
    ):
        zip_path = _write_zip(tmp_path / f"{name}.zip", files)
        root = unpack(zip_path, tmp_path / f"out-{name}")
        assert (root / MANIFEST).exists()


def test_unpack_refuses_a_path_that_escapes_the_archive(tmp_path):
    zip_path = _write_zip(tmp_path / "evil.zip", {"../escaped.txt": "x"})
    with pytest.raises(SubmissionError, match="outside itself"):
        unpack(zip_path, tmp_path / "out")


@pytest.mark.parametrize("files,match", [
    ({"readme.txt": "hello"}, "no submission.yaml"),
    ({MANIFEST: MANIFEST_OK}, "no checkpoint"),
    ({MANIFEST: "algo: PPO\n", "model/policy.zip": "x"}, "'team' key"),
    ({MANIFEST: "[not a mapping]\n", "model/policy.zip": "x"}, "mapping"),
])
def test_a_bad_submission_says_what_is_wrong(tmp_path, files, match):
    """Every rejection has to be actionable — we are not debugging fifteen zips
    at a prizegiving."""
    zip_path = _write_zip(tmp_path / "bad.zip", files)
    with pytest.raises(SubmissionError, match=match):
        load_submission(zip_path, tmp_path / "out")


def test_a_good_submission_loads(tmp_path):
    zip_path = _write_zip(tmp_path / "ok.zip",
                          {MANIFEST: MANIFEST_OK + 'notes: "hi"\n',
                           "model/policy.zip": "x"})
    sub = load_submission(zip_path, tmp_path / "out")
    assert sub.team == "brick-with-wings"
    assert sub.env_entry_point is None       # stock environment
    assert sub.deterministic is True
    assert sub.model_path.exists()


def test_shipping_vecnormalize_warns_rather_than_failing(tmp_path):
    """Flight4DEnv already emits scaled observations, so the harness does not
    wrap the scored env. Silently ignoring the file would be the bad outcome."""
    zip_path = _write_zip(tmp_path / "vn.zip",
                          {MANIFEST: MANIFEST_OK, "model/policy.zip": "x",
                           "model/vecnormalize.pkl": "x"})
    sub = load_submission(zip_path, tmp_path / "out")
    assert any("vecnormalize" in w for w in sub.warnings)


@pytest.mark.parametrize("spec", ["envs.flight_4d:Flight4DEnv",
                                  "envs.flight_4d.Flight4DEnv"])
def test_entry_points_resolve_in_both_spellings(spec):
    assert _resolve(spec) is Flight4DEnv


@pytest.mark.parametrize("spec,match", [
    ("MyFlightEnv", "needs a module and a class"),
    ("envs.nope:Thing", "could not import"),
    ("envs.flight_4d:Nope", "has no"),
])
def test_a_bad_entry_point_says_which_half_is_wrong(spec, match):
    with pytest.raises(SubmissionError, match=match):
        _resolve(spec)


# ------------------------------------------------- policy_entry_point

def test_policy_entry_point_skips_the_checkpoint_requirement(tmp_path):
    """A hand-written controller has no SB3 zip to ship — requiring one would
    quietly break the rules' own promise that the network is the team's to
    choose."""
    zip_path = _write_zip(
        tmp_path / "handwritten.zip",
        {MANIFEST: MANIFEST_OK +
         "policy_entry_point: agents.rule_based_4d:Priority4DController\n"})
    sub = load_submission(zip_path, tmp_path / "out")
    assert sub.policy_entry_point == "agents.rule_based_4d:Priority4DController"
    assert not sub.model_path.exists(), "no checkpoint was shipped, or required"


def test_policy_entry_point_is_instantiated_with_no_arguments_and_predicts():
    """The frozen scenario's defaults are also `Priority4DController`'s
    constructor defaults, so a no-argument build is the whole contract — a
    policy that wants the scenario reads `competition.config.SCENARIO` itself.
    """
    sub = _submission(make_env(), None,
                      policy_entry_point="agents.rule_based_4d:Priority4DController")
    policy = sub.policy()
    assert policy.predict.__self__.__class__.__name__ == "Priority4DController"

    obs, _ = sub.observed_env().reset(seed=4242)
    action, _ = policy.predict(obs, deterministic=True)
    assert action in range(1 + 7 * SCENARIO.n_flights)


def test_a_submission_without_either_checkpoint_or_entry_point_is_rejected(tmp_path):
    zip_path = _write_zip(tmp_path / "empty.zip", {MANIFEST: MANIFEST_OK})
    with pytest.raises(SubmissionError, match="policy_entry_point"):
        load_submission(zip_path, tmp_path / "out")


# ------------------------------------------------ the boundary, end to end

def _shipped_envs_zip(tmp_path, sabotage: str) -> Path:
    """A zip that ships the WHOLE ``envs/`` package with one edit, plus a
    torch-free policy via ``policy_entry_point`` — the shape a real team's
    submission has, so the test exercises the real import path."""
    import shutil

    root = tmp_path / "team"
    shutil.copytree(_paths.AIRTRAFFIC / "envs", root / "envs",
                    ignore=shutil.ignore_patterns("__pycache__"))
    src = root / "envs" / "flight_4d.py"
    text = src.read_text()
    assert sabotage in text
    src.write_text(text.replace(sabotage, "        return 0  # SABOTAGED\n"))
    (root / "policy").mkdir()
    (root / "policy" / "__init__.py").write_text("")
    (root / "policy" / "idle.py").write_text(
        "import numpy as np\n"
        "class Idle:\n"
        "    def predict(self, obs, deterministic=True):\n"
        "        return np.array(0), None\n")
    (root / MANIFEST).write_text(
        "team: saboteur\n"
        "env_entry_point: envs.flight_4d:Flight4DEnv\n"
        "policy_entry_point: policy.idle:Idle\n")
    zip_path = tmp_path / "saboteur.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in root.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(root))
    return zip_path


def test_a_shipped_envs_package_cannot_change_what_is_scored(tmp_path):
    """The failure this pins: ``ingest`` puts the zip's root first on
    ``sys.path`` *before* ``competition.rollout`` is imported, so a naive
    ``from envs import Flight4DEnv`` there resolved to the TEAM's copy and
    every KPI was read from an environment the team wrote. The generic
    contract suite is green for this zip — it never compares against our
    physics — so the reference environment is the only thing standing between
    a rewritten ``_count_congestion`` and the top of the board.

    Run in a fresh process, exactly as ``score-zips`` does, because the bug is
    an import-order bug and cannot be reproduced in a process that has already
    imported the harness."""
    import json
    import subprocess

    zip_path = _shipped_envs_zip(
        tmp_path, sabotage="        f = self.sim.flights\n        c = 0\n")
    proc = subprocess.run(
        [sys.executable, "-m", "competition.ingest", "--zip", str(zip_path),
         "--contract", "--limit", "3", "--out", str(tmp_path / "sub.csv")],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["ok"], result
    assert "contract" in result                       # the gate ran, and passed

    # An idle policy on the first three public seeds, scored on OUR environment.
    from competition.agents import load_agent
    from competition.rollout import rollout_agent
    from competition.score import aggregate
    from competition.seeds import PUBLIC_SEEDS, load_seeds
    seeds = load_seeds(PUBLIC_SEEDS)[:3]
    ours = aggregate([rollout_agent(load_agent("noop"), s)[0] for s in seeds]).summary()
    assert ours["congestion_total"] > 0
    assert result["score"]["congestion_total"] == ours["congestion_total"]
    assert result["score"]["failed_episodes"] == ours["failed_episodes"]

    # --out wrote a Kaggle-shaped trace for exactly those seeds.
    from competition.submission import read_submission
    traces = read_submission(tmp_path / "sub.csv", SCENARIO.n_flights,
                             SCENARIO.max_steps)
    assert sorted(traces) == sorted(seeds)
    assert all(len(t) == SCENARIO.max_steps for t in traces.values())
    assert result["submission"] == {"path": str(tmp_path / "sub.csv"),
                                    "rows": 3 * SCENARIO.max_steps, "seeds": 3}


def test_out_is_refused_on_the_private_seeds(tmp_path):
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "competition.ingest", "--zip", "x.zip",
         "--private", "--out", "sub.csv"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "PUBLIC" in proc.stderr
