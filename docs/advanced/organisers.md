---
title: Running the competition
---

# Running the competition — organiser runbook

Every command, in the order you run it. This page is *what to type*;
[the Kaggle leaderboard](kaggle.md) is *what to click*.

!!! info "Assumed throughout"
    ```bash
    conda activate rlbootcamp
    cd sessions/03-advanced
    ```
    Every `competition.*` command runs from there — the package is
    `competition`, not `airtraffic/competition`. The AirTraffic scripts run one
    level down, in `airtraffic/`.

!!! danger "The one secret"
    `$COMPETITION_SALT` expands into the seed list that decides the trophies.
    It is not in the repository and must not be. Everything else here is
    reproducible from a clean clone; this is the single piece of state you carry.

---

## Phase 1 — before the session

### 1. Regenerate the private seeds

```bash
export COMPETITION_SALT='…'        # from your password manager, not from here
python -m competition.make_seeds --which private --salt "$COMPETITION_SALT"
```

Writes `competition/seeds_private.txt`, which is gitignored. Derived from the
salt alone, so it never has to be stored — and never has to be committed.

The public list already ships. You only regenerate it if you deliberately want a
different one, which invalidates every number in the repo:

```bash
python -m competition.make_seeds --which public          # salt is built in
```

### 2. Sanity-check the harness

```bash
python -m pytest competition/tests -q                    # 80 tests, ~8 s
cd airtraffic && python -m pytest -q && cd ..            # 207 tests, ~7 s
```

`competition/tests/test_kaggle_metric.py` is the load-bearing one: it asserts
the Kaggle metric and the real environment agree exactly. If it is red, the
leaderboard would be wrong silently.

### 3. Build the Kaggle files

```bash
# the solution: initial states + Kaggle's own public/private split
python -m competition.kaggle.export --holdout-frac 0.4 --out solution.csv
#   -> wrote 5000 rows for 100 seeds
#      60 seeds Public, 40 Private

# the sample submission: the brick is a valid all-zero submission
python -m competition.run submit --agent noop --out sample_submission.csv
```

### 4. Build the benchmark rows

```bash
python -m competition.run submit --agent rule-based --out submission_rulebased.csv
python -m competition.run submit \
    --agent airtraffic/checkpoints/ppo_4M_curriculum.zip \
    --out submission_ppo.csv
```

Upload all three as **sandbox submissions**; tick `Benchmark` on `noop` and
`rule-based`. [Which, and why](kaggle.md#sandbox-submissions-what-to-put-there).

### 5. Set the competition up on Kaggle

[The full walkthrough is on its own page](kaggle.md#part-2-setting-it-up-organisers)
— the metric notebook, the settings panel, and the **Leaderboard Sort Order**
toggle that defaults to the wrong direction.

### 6. Confirm the local bar

```bash
python -m competition.run compare --agent noop --agent rule-based \
    --agent airtraffic/checkpoints/ppo_4M_curriculum.zip
```

Expected on the 100 public seeds, verbatim (score v2.0.0, ~90 s):

```text
Flight Challenge scoring v2.0.0 — 5 aircraft, n~U(2, 4), max_steps=50

 #  submission             failed  congestion   exit miss  clearances  touched
------------------------------------------------------------------------------
 1  airtraffic/checkpoints/ppo_4M_curriculum.zip     10         176      0.1288        2089      477
 2  rule-based                 20          96      0.5700         727      231
 3  noop                      100         752      0.0000           0        0

ranked on safety -> timeliness -> efficiency (score v2.0.0, 100 episodes)
```

(A checkpoint is named by its path, which overruns the column. Cosmetic.)

If those numbers have moved, something in the environment changed and every
figure in the decks needs re-deriving before anyone teaches from them. They are
**not** the numbers in `slides/data/air_4d_board.csv`, which scores a different
held-out block — see [the two seed blocks](../reference/tooling.md#session-3-airtraffic).

---

## Phase 2 — during the session

Mostly nothing. The public board runs itself. Two things worth having ready:

```bash
# a quick head-to-head while someone is at your elbow (20 seeds, seconds)
python -m competition.run --limit 20 compare --agent rule-based --agent <their.zip>

# video of a specific episode, for the projector
cd airtraffic
python scripts/render_4d.py --agent rule-based --agent <their.zip> --seed 900000
```

!!! tip "When a participant's local score disagrees with Kaggle"
    Nine times in ten the file is stale or was built against a different seed
    list. Have them run the replay, which is the same code Kaggle runs:

    ```bash
    python -m competition.run replay --submission submission.csv
    ```

    If replay agrees with Kaggle and disagrees with what `submit` printed, the
    submission was built from a different checkpoint than they think.

---

## Phase 3 — after submissions close

### 1. Collect the zips

Teams deliver one zip each to the shared
[Drive folder](https://drive.google.com/drive/folders/1Mrv5aZ70LDN5goMj0k67XFGK84LZtBOI?usp=sharing)
— a one-time drop at the sprint deadline, not a running submission like
Kaggle. Announce the deadline explicitly and close the folder to further
uploads once it passes, so "which version is the real one" is never a
question.

Download everything into one local directory, any filename — the team name
is read from `submission.yaml`, not the filename:

```bash
mkdir -p submissions/          # gitignored
# download every file from the Drive folder into submissions/
```

!!! tip "A `--limit` pass catches a missed deadline early"
    If a team's zip is a `--contract` failure they never re-uploaded, you want
    to know that from the room, not from the leaderboard. See §2 below.

### 2. Run the private board

```bash
python -m competition.run --private score-zips --dir submissions/
```

This is the ranking that decides the trophies. Per zip, in its own process: the
contract suite as a gate, then 100 secret seeds. Unscored submissions are listed
underneath with the assertion that failed.

!!! warning "Do a `--limit` pass first"
    ```bash
    python -m competition.run --private --limit 10 score-zips --dir submissions/
    ```
    Ten seeds tells you which zips are *broken* in about a minute. Fixing a
    rejection is a conversation with a team, and you want that conversation
    before the full run, not after it.

### 3. Debug a rejected submission

`score-zips` reports the reason; `ingest` shows the whole picture for one zip:

```bash
python -m competition.ingest --zip submissions/team-x.zip --private --contract \
    --workdir /tmp/team-x
```

It prints JSON — the manifest it read, any warnings, and either the score or the
error. `--workdir` keeps the unpacked files so you can look at them.

The most common rejection by far is a submitted `envs/` package whose module
imports the stock `envs.flight_4d`; that resolves inside *their* package and
fails. [The rule, and the fix](competition.md#what-you-submit).

### 4. The post-mortem and the trophies

```bash
python -m competition.run --private postmortem \
    --agent noop --agent rule-based --agent submissions/team-x/model/policy.zip
```

Five diagnostic passes and the trophies that follow from them. *Black Box* and
*Icarus* are jury calls and are deliberately not computed.

!!! note "`postmortem` takes checkpoints, not zips"
    It predates the zip format. Point it at the `model/policy.zip` inside an
    unpacked submission — `ingest --workdir` is how you get one. A team whose
    policy needs their own environment to be meaningful is not covered by this
    path; score them with `score-zips` and read the post-mortem as indicative.

### 5. Video for the ceremony

```bash
cd airtraffic
python scripts/render_4d.py --agent <winner.zip> --agent rule-based \
    --seed 900000 --ext mp4
```

---

## Complete command reference

Every flag below is checked against `--help`. Flags marked
:material-cog: are organiser-only.

### `competition.run` — global flags

These come **before** the subcommand: `python -m competition.run --private compare …`

| Flag | Default | What it does |
|---|---|---|
| `--private` :material-cog: | off | use `seeds_private.txt` instead of the public list |
| `--seeds FILE` | — | an explicit seed file, overriding both |
| `--limit N` | all | use only the first N seeds — for quick checks |
| `--stochastic` | off | sample actions instead of taking the argmax |
| `--n-lo N` / `--n-hi N` | frozen | narrow the converging-traffic draw. **Development only** — every line of output says so |
| `-v`, `--verbose` | off | per-seed progress |

### `competition.run` — subcommands

| Command | Flags |
|---|---|
| `submit` | `--agent` *(required)* — `noop`, `rule-based`, or a path to an SB3 `.zip`; `--out` (default `submission.csv`) |
| `replay` | `--submission` *(required)*; `--strict` — reject a file carrying seeds outside the list |
| `compare` | `--agent` *(required, repeatable)* |
| `postmortem` | `--agent` *(required, repeatable)*; `--extraction-k N` (default `8`) — stochastic rollouts per seed for the best-extraction pass |
| `score-zips` :material-cog: | `--dir` (default `submissions`); `--no-contract` — skip the gate, **debugging only** |

### `competition.make_seeds` :material-cog:

| Flag | Default | What it does |
|---|---|---|
| `--which public\|private` | *(required)* | which list to write |
| `--salt S` | — | **required for `private`**, and must stay secret |
| `--count N` | `100` | how many seeds. Changing it invalidates every number in the repo |

### `competition.kaggle.export` :material-cog:

| Flag | Default | What it does |
|---|---|---|
| `--out FILE` | `solution.csv` | where to write |
| `--holdout-frac F` | `0.0` | fraction marked `Usage=Private` — Kaggle's own split. `0.4` is the recipe |
| `--private` | off | export the **secret** seed list. Never upload this file |
| `--usage U` | `Public` | mark every row `Public`, `Private` or `Ignored`. Ignored when `--holdout-frac` is set |
| `--limit N` | all | first N seeds only |

### `competition.ingest` :material-cog:

| Flag | Default | What it does |
|---|---|---|
| `--zip PATH` | *(required)* | the submission to score |
| `--contract` | off | run the contract suite first; red means unscored |
| `--private` | off | score on the secret seed list |
| `--workdir DIR` | a temp dir | where to unpack — pass one to keep the files |
| `--limit N` | all | first N seeds only |

### Environment variables

| Variable | Read by | What it does |
|---|---|---|
| `COMPETITION_SALT` :material-cog: | you, passed to `make_seeds --salt` | expands into the private seed list |
| `RLB_ENV_ENTRY_POINT` | `airtraffic/tests/conftest.py` | point the contract suite at a class: `envs.my_env:MyFlightEnv` |
| `RLB_SUBMISSION_ROOT` | `airtraffic/tests/conftest.py` | put an unpacked submission ahead of ours on the import path. Set by the harness |
| `OMP_NUM_THREADS=1` | torch | **not optional** when running more than one training job — an 11× difference |

### Running the contract suite by hand

The gate, invoked exactly as the harness invokes it:

```bash
cd sessions/03-advanced/airtraffic
RLB_ENV_ENTRY_POINT=envs.my_env:MyFlightEnv \
RLB_SUBMISSION_ROOT=/path/to/unpacked/submission \
    python -m pytest tests/test_env_contract.py -q
```

Omit both variables and it checks the stock environment. This is also the
command to give a participant who wants to check their own environment before
handing it in.

---

## Files, and which of them are secret

| Path | Committed? | Notes |
|---|:--:|---|
| `competition/seeds_public.txt` | yes | 100 published seeds. Participants train against these |
| `competition/seeds_private.txt` | **no** | regenerate from the salt; gitignored |
| `solution.csv` | no | gitignored; upload to Kaggle |
| `sample_submission.csv`, `submission_*.csv` | no | gitignored |
| `submissions/` | no | gitignored; the zips and what they unpack into |
| [Drive folder](https://drive.google.com/drive/folders/1Mrv5aZ70LDN5goMj0k67XFGK84LZtBOI?usp=sharing) | — | not part of the repo at all; where teams drop their zip, and where `submissions/` is downloaded from |
| `airtraffic/checkpoints/ppo_4M_curriculum.zip` | **yes** | deliberately — the board must be reproducible from a fresh clone |

## Related

- [The Kaggle leaderboard](kaggle.md) — the UI half of Phase 1.
- [Competition rules](competition.md) — what participants are told.
- [Scripts and tooling](../reference/tooling.md) — every runnable thing in the repo.
