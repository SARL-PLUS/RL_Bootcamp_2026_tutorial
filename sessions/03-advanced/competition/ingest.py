"""Take a submitted zip apart, far enough to score it on the private seeds.

    # score one, printing JSON (this is what the batch runner spawns)
    python -m competition.ingest --zip submissions/brick-with-wings.zip --private

    # …with the contract suite first, which is how it is actually run
    python -m competition.ingest --zip <path> --private --contract

    # a TEAM's own dry run: the same gate, on the public seeds, and the action
    # trace written out as their Kaggle submission (the only way onto the
    # public board for a policy that observes an environment of its own)
    python -m competition.ingest --zip <team>.zip --contract --out submission.csv

The zip is the *private* board's submission format. The public board takes an
action-trace CSV instead, because Kaggle can only replay actions — see
``submission.py``. Here we have the policy itself, which is the whole point: we
run it on seeds it has never seen, and those seeds never leave this machine.

What is loaded, and what is deliberately not
--------------------------------------------
``submission.yaml`` names the team, the checkpoint and (optionally) a custom
environment class. If ``env_entry_point`` is given, the student's environment is
what their **policy observes**; if it is omitted, they observe the stock one.

The competition rules say "everything else is yours: observation space, action
space, reward function, ... network architecture, training regime" — which
promised more than an SB3 checkpoint could actually redeem. ``policy_entry_point``
closes that gap: a class with a SB3-shaped ``predict(obs, deterministic)``, built
by hand rather than trained, is instantiated with **no arguments** in place of
``PPO.load(model/policy.zip)``. Set it and ``model/policy.zip`` is not required.

**The score never comes from their environment.** Their env produces
observations, their policy chooses an action, and that action is stepped through
*our* ``Flight4DEnv``, which is where every KPI is read. Two consequences worth
being explicit about:

* A student cannot gain from an environment that models the sector wrongly. A
  wrong environment feeds their policy misleading observations and scores
  *worse*. There is nothing to defend against, so this file does not try to.
* Their reward, their termination shaping and their info dict are never
  consulted. Only the action index crosses the boundary.

Our environment also decides when the episode is over. A student env that ends
early does not end the scored episode — the remaining steps are idled, exactly
as a short action trace is on the public board, so ending early stays a penalty
rather than a way to stop accumulating congestion.

Isolation
---------
Every submission ships a directory called ``envs``. Import two of them into one
interpreter and the second team silently gets the first team's code. So each zip
is scored in **its own process** and this module is the entry point for one;
``competition.run score-zips`` is the batch runner that spawns them.
"""
import argparse
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import _paths  # noqa: F401  (import side effect: sys.path)
from .config import SCENARIO

REQUIRED_MODEL = Path("model") / "policy.zip"
MANIFEST = "submission.yaml"


class SubmissionError(Exception):
    """A submission we cannot score, with a reason to send back to the team."""


@dataclass
class Submission:
    """One unpacked zip, ready to roll out."""
    team: str
    root: Path
    model_path: Path
    algo: str = "PPO"
    deterministic: bool = True
    notes: str = ""
    env_entry_point: str | None = None
    policy_entry_point: str | None = None
    warnings: list = field(default_factory=list)
    _policy: object = None
    _observed: object = None

    def observed_env(self, scenario=SCENARIO):
        """What the policy observes — theirs if they shipped one, else stock.

        Built once and reset per episode: ``reset(seed=)`` carrying nothing over
        is a property the contract suite pins, so rebuilding it 100 times would
        only be slower.
        """
        if self._observed is None:
            if self.env_entry_point is None:
                from .rollout import make_env
                self._observed = make_env(scenario)
            else:
                cls = _resolve(self.env_entry_point)
                self._observed = cls(**scenario.as_env_kwargs())
        return self._observed

    def policy(self):
        """The checkpoint — theirs via ``policy_entry_point`` if they shipped
        code instead of a network, else the SB3 zip. Imported lazily — torch is
        heavy, and the batch runner reads manifests long before it loads
        anything.

        A ``policy_entry_point`` class is instantiated with **no arguments**:
        the frozen scenario's defaults are also ``Priority4DController``'s
        constructor defaults (``n_flights=5, max_steps=50, hold_steps=5``), so
        a policy that wants them can read ``competition.config.SCENARIO``
        itself rather than trust us to inject the right kwargs for a class we
        have never seen.
        """
        if self._policy is None:
            if self.policy_entry_point is None:
                from stable_baselines3 import PPO
                self._policy = PPO.load(str(self.model_path), device="cpu")
            else:
                cls = _resolve(self.policy_entry_point)
                self._policy = cls()
        return self._policy


def _resolve(entry_point: str):
    """``envs.my_env:MyFlightEnv`` or ``envs.my_env.MyFlightEnv`` -> the class."""
    import importlib

    spec = entry_point.replace(":", ".")
    module_name, _, attr = spec.rpartition(".")
    if not module_name:
        raise SubmissionError(
            f"env_entry_point {entry_point!r} needs a module and a class, "
            f"e.g. envs.my_env:MyFlightEnv")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SubmissionError(
            f"could not import {module_name!r} from the submission: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise SubmissionError(
            f"{module_name!r} has no {attr!r}") from exc


def unpack(zip_path: Path, workdir: Path) -> Path:
    """Extract into ``workdir/<stem>`` and return the root that holds the files.

    A zip made with ``zip -r sub.zip .`` has the files at the top; one made by
    zipping a folder has them one level down. Both are common and neither is
    worth failing a team over, so we look for the manifest and take its parent.
    """
    zip_path, workdir = Path(zip_path), Path(workdir)
    if not zip_path.exists():
        raise SubmissionError(f"{zip_path} does not exist")
    root = workdir / zip_path.stem
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                target = (root / name).resolve()
                if not str(target).startswith(str(root.resolve())):
                    raise SubmissionError(
                        f"{zip_path.name} contains a path outside itself "
                        f"({name!r}); refusing to extract it")
            zf.extractall(root)
    except zipfile.BadZipFile as exc:
        raise SubmissionError(f"{zip_path.name} is not a readable zip") from exc

    found = sorted(root.rglob(MANIFEST), key=lambda p: len(p.parts))
    if not found:
        raise SubmissionError(
            f"{zip_path.name} has no {MANIFEST}. It is the one required file — "
            f"see the competition rules for the four keys it needs.")
    return found[0].parent


def read_manifest(root: Path) -> dict:
    import yaml

    text = (root / MANIFEST).read_text()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SubmissionError(f"{MANIFEST} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SubmissionError(f"{MANIFEST} must be a mapping of keys to values")
    if not data.get("team"):
        raise SubmissionError(f"{MANIFEST} needs a 'team' key")
    return data


def load_submission(zip_path: Path, workdir: Path) -> Submission:
    """Unpack, validate, and put the submission's own code on the import path."""
    root = unpack(zip_path, workdir)
    data = read_manifest(root)
    policy_entry_point = data.get("policy_entry_point") or None

    model_path = root / REQUIRED_MODEL
    if policy_entry_point is None and not model_path.exists():
        candidates = sorted(root.rglob("*.zip"))
        raise SubmissionError(
            f"no checkpoint at {REQUIRED_MODEL}" +
            (f" (found {candidates[0].relative_to(root)} — move it there)"
             if candidates else "") +
            f". If your policy is not an SB3 checkpoint, set "
            f"'policy_entry_point' in {MANIFEST} instead.")

    warnings = []
    if (root / "model" / "vecnormalize.pkl").exists():
        warnings.append(
            "vecnormalize.pkl shipped but not applied — Flight4DEnv already "
            "emits observations in [-1, 1], and the harness does not wrap the "
            "scored env. If your policy needs it, say so and we will score it "
            "by hand.")

    # The submission's own modules must win over ours for `envs`/`policy` to
    # resolve to theirs. This is why one process scores exactly one zip.
    sys.path.insert(0, str(root))

    return Submission(
        team=str(data["team"]),
        root=root,
        model_path=model_path,
        algo=str(data.get("algo", "PPO")),
        deterministic=bool(data.get("deterministic", True)),
        notes=str(data.get("notes", "")),
        env_entry_point=data.get("env_entry_point") or None,
        policy_entry_point=policy_entry_point,
        warnings=warnings,
    )


def run_contract_suite(submission: Submission) -> tuple[bool, str]:
    """The gate: red means unscored, with no manual fixes on our side.

    Runs `airtraffic/tests/test_env_contract.py` against whatever the submission
    asks us to observe. A team that shipped no environment is running our own,
    so the suite is green by construction and we say so rather than spending
    thirty seconds proving it again.
    """
    if submission.env_entry_point is None:
        return True, "stock environment — contract suite not re-run"

    import os
    import subprocess

    airtraffic = Path(_paths.AIRTRAFFIC)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-x",
         str(airtraffic / "tests" / "test_env_contract.py")],
        cwd=airtraffic, capture_output=True, text=True,
        env={**os.environ,
             "RLB_ENV_ENTRY_POINT": submission.env_entry_point,
             "RLB_SUBMISSION_ROOT": str(submission.root)},
    )
    if proc.returncode == 0:
        return True, (proc.stdout.strip().splitlines() or ["passed"])[-1]

    # `-x` stops at the first failure, and the first failure is the one worth
    # sending back. A count of failures tells a team nothing they can act on;
    # the assertion that fired tells them what their environment broke.
    output = (proc.stdout or proc.stderr).splitlines()
    for line in output:
        if line.startswith("E   ") or line.startswith("FAILED"):
            return False, line.strip()
    return False, (output or ["no output"])[-1].strip()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True)
    p.add_argument("--workdir", default=None,
                   help="where to extract (default: a temp dir)")
    p.add_argument("--private", action="store_true",
                   help="score on the secret seed list")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--contract", action="store_true",
                   help="run the contract suite first; a red suite is unscored")
    p.add_argument("--out", default=None, metavar="submission.csv",
                   help="also write the action trace as a Kaggle submission — "
                        "the route to the public board for a policy that "
                        "observes its own environment, which `competition.run "
                        "submit` cannot roll out")
    args = p.parse_args()
    if args.out and args.private:
        p.error("--out writes a Kaggle submission, and Kaggle scores the "
                "PUBLIC seeds; drop --private")

    import tempfile
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp())

    result = {"zip": args.zip}
    try:
        submission = load_submission(Path(args.zip), workdir)
        result.update(team=submission.team, algo=submission.algo,
                      env_entry_point=submission.env_entry_point,
                      policy_entry_point=submission.policy_entry_point,
                      notes=submission.notes, warnings=submission.warnings)

        if args.contract:
            ok, detail = run_contract_suite(submission)
            result["contract"] = detail
            if not ok:
                raise SubmissionError(f"contract suite failed: {detail}")

        from .rollout import rollout_submission
        from .score import aggregate
        from .seeds import PRIVATE_SEEDS, PUBLIC_SEEDS, load_seeds

        seeds = load_seeds(PRIVATE_SEEDS if args.private else PUBLIC_SEEDS)
        if args.limit:
            seeds = seeds[: args.limit]
        episodes, traces = [], {}
        for seed in seeds:
            kpis, actions = rollout_submission(submission, seed)
            episodes.append(kpis)
            traces[seed] = actions
        result["score"] = aggregate(episodes).summary()
        if args.out:
            from .submission import write_submission
            rows = write_submission(Path(args.out), traces, SCENARIO.max_steps)
            result["submission"] = {"path": args.out, "rows": rows,
                                    "seeds": len(traces)}
        result["ok"] = True
    except SubmissionError as exc:
        result.update(ok=False, error=str(exc))

    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
