"""Put the shared AirTraffic package on the import path — and pin the reference
environment so that nothing a submission ships can shadow it.

The competition harness lives beside the advanced session; the environment it
scores lives at the repository root because Sessions #2 and #3 share it.

Two ways to reach that environment, deliberately separate:

* ``import envs`` — the public name. Fine for the harness's own tooling, and it
  is also what a submitted zip's ``envs/`` package is *meant* to shadow, so the
  team's policy observes the environment it was trained on.
* :func:`reference_envs` — the **scored** environment, loaded from this
  repository's own files under a private module name. ``sys.path`` order cannot
  reach it. This is what :mod:`competition.rollout` builds the KPIs from, and it
  is the reason "only the action index crosses the boundary" is true rather than
  merely intended: the first version bound ``make_env`` to whatever ``envs``
  resolved to at import time, which — inside :mod:`competition.ingest`, where the
  zip's root goes on the path first — was the team's copy.
"""
import sys
from pathlib import Path

AIRTRAFFIC = Path(__file__).resolve().parents[1] / "airtraffic"

if str(AIRTRAFFIC) not in sys.path:
    sys.path.insert(0, str(AIRTRAFFIC))

#: The private name the scored environment package is loaded under.
REFERENCE_MODULE = "rlb_reference_envs"


def reference_envs():
    """The repository's own ``envs`` package, immune to ``sys.path`` shadowing.

    Loaded once, by file path, under :data:`REFERENCE_MODULE`; the package's
    relative imports resolve inside it. Re-registering ``Flight4DEnv-v0`` with
    gymnasium is harmless (a warning at most) and is silenced here.
    """
    module = sys.modules.get(REFERENCE_MODULE)
    if module is not None:
        return module

    import importlib.util
    import warnings

    package = AIRTRAFFIC / "envs"
    spec = importlib.util.spec_from_file_location(
        REFERENCE_MODULE, package / "__init__.py",
        submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[REFERENCE_MODULE] = module
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)
    return module
