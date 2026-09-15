"""Generate a seed list.

    python -m competition.make_seeds --which public
    python -m competition.make_seeds --which private --salt "$COMPETITION_SALT"

The private list is derived from a salt and is gitignored, so it can be
regenerated from the salt alone but never leaks by being committed.
"""
import argparse

from .seeds import PRIVATE_SEEDS, PUBLIC_SEEDS, derive_seeds, write_seeds

PUBLIC_SALT = "rl-bootcamp-2026-public"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--which", choices=["public", "private"], required=True)
    p.add_argument("--salt", default=None,
                   help="required for the private list; must be kept secret")
    p.add_argument("--count", type=int, default=100)
    args = p.parse_args()

    if args.which == "public":
        seeds = derive_seeds(PUBLIC_SALT, args.count)
        write_seeds(PUBLIC_SEEDS, seeds,
                    "PUBLIC competition seeds — students train and submit against these")
        print(f"wrote {len(seeds)} public seeds to {PUBLIC_SEEDS}")
    else:
        if not args.salt:
            p.error("--salt is required for the private list, and must not be "
                    "the public salt or committed anywhere")
        if args.salt == PUBLIC_SALT:
            p.error("the private salt must differ from the public one")
        seeds = derive_seeds(args.salt, args.count)
        write_seeds(PRIVATE_SEEDS, seeds,
                    "PRIVATE competition seeds — DO NOT COMMIT OR DISTRIBUTE")
        print(f"wrote {len(seeds)} private seeds to {PRIVATE_SEEDS}")
        print("reminder: seeds_private.txt is gitignored. Keep the salt safe — "
              "it is the only thing needed to regenerate this list.")


if __name__ == "__main__":
    main()
