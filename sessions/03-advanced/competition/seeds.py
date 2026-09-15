"""Seed lists for the two leaderboards.

Public seeds ship with the repository: students roll out on them, submit the
action trace, and Kaggle replays it. Private seeds are generated from a salt that
is **not** committed, so nobody can train against the ranking that decides the
trophies.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC_SEEDS = HERE / "seeds_public.txt"
PRIVATE_SEEDS = HERE / "seeds_private.txt"

#: Seeds must fit what Gymnasium accepts for reset(seed=...).
SEED_MODULUS = 2 ** 31 - 1


def derive_seeds(salt: str, count: int) -> list[int]:
    """Deterministically derive `count` distinct seeds from `salt`.

    Reproducible from the salt alone, so the private list can be regenerated
    without ever being committed.
    """
    seeds, i = [], 0
    seen = set()
    while len(seeds) < count:
        digest = hashlib.sha256(f"{salt}:{i}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big") % SEED_MODULUS
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
        i += 1
    return seeds


def write_seeds(path: Path, seeds: list[int], header: str) -> None:
    lines = [f"# {header}", f"# {len(seeds)} seeds"]
    lines += [str(s) for s in seeds]
    path.write_text("\n".join(lines) + "\n")


def load_seeds(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Public seeds ship with the repo; private seeds "
            f"are generated with `python -m competition.make_seeds --salt ...` "
            f"and are deliberately not committed."
        )
    seeds = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seeds.append(int(line))
    if not seeds:
        raise ValueError(f"{path} contains no seeds")
    return seeds


def load_public() -> list[int]:
    return load_seeds(PUBLIC_SEEDS)


def load_private() -> list[int]:
    return load_seeds(PRIVATE_SEEDS)
