import importlib
import os
import sys
from pathlib import Path

import pytest

#: Point the whole suite at your own class without editing this file:
#:
#:     RLB_ENV_ENTRY_POINT=envs.my_env:MyFlightEnv pytest tests/test_env_contract.py
#:
#: The competition harness sets this when it checks a submitted environment, so
#: what runs there is exactly what you can run here.
ENV_ENTRY_POINT = "RLB_ENV_ENTRY_POINT"

#: …and this, to the unpacked submission, whose own ``envs/`` must win over
#: ours. Both inserts happen *before* anything imports ``envs``: once a package
#: is in ``sys.modules``, path order stops mattering and the submission's
#: modules resolve to the shipped ones instead. The submission goes in last so
#: it lands ahead.
SUBMISSION_ROOT = "RLB_SUBMISSION_ROOT"

# make `envs`, `agents`, `policies` and `utils` importable when running pytest
# from this directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.environ.get(SUBMISSION_ROOT):
    sys.path.insert(0, os.environ[SUBMISSION_ROOT])

try:
    from envs import Flight4DEnv  # noqa: E402
except ImportError:  # a submission shipped an envs/ that does not export it
    Flight4DEnv = None


def _env_class():
    """Flight4DEnv, unless ``$RLB_ENV_ENTRY_POINT`` names something else."""
    entry_point = os.environ.get(ENV_ENTRY_POINT)
    if not entry_point:
        if Flight4DEnv is None:
            raise ImportError(
                "no Flight4DEnv importable, and no $RLB_ENV_ENTRY_POINT naming "
                "a replacement. A submitted envs/ package must either export "
                "Flight4DEnv or name its class in submission.yaml.")
        return Flight4DEnv
    module_name, _, attr = entry_point.replace(":", ".").rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def env_factory(**overrides):
    """The environment under contract.

    ADVANCED SESSION: when you subclass or modify Flight4DEnv, run the suite
    against YOUR class — the whole thing must stay green for your submission to
    be scored in the competition. Either set ``$RLB_ENV_ENTRY_POINT`` as above,
    or edit this factory directly. The harness uses the first form.
    """
    kwargs = dict(n_flights=5, max_steps=50)
    kwargs.update(overrides)
    return _env_class()(**kwargs)


@pytest.fixture
def make_env():
    return env_factory
