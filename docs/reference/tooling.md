---
title: Scripts and tooling
---

# Scripts and tooling — everything runnable in the repo

The session pages cover the handful of commands you need to *do the tutorial*.
This page is the complete inventory: every script, what it is for, and who is
expected to run it. Reach for it when you are looking for a tool you half
remember, or when you are setting up the competition.

!!! info "Who runs what"
    :material-account: **Participants** — part of a session.
    :material-school: **Instructors** — reference material and answer keys.
    :material-cog: **Organisers** — competition and deck infrastructure.

Everything assumes the `rlbootcamp` environment is active. Paths are relative to
the repository root unless a block says otherwise.

---

## Session #2 — Crippled Ant

`cd sessions/02-intermediate/crippled-ant` first; the scripts resolve their
config relative to that directory.

| Script | Who | What it does |
|---|:--:|---|
| `scripts/train.py` | :material-account: | PPO/SAC on `Ant-v5` and its crippled variants. Hydra-configured, so every value in `conf/` is a command-line override and `-m` turns comma-separated values into a sweep. Full flag list: [running the Crippled Ant](../intermediate/running.md). |
| `scripts/evaluate.py` | :material-account: | Evaluate a finished run, optionally under an injury it never trained on — this is the zero-shot transfer measurement. `--render` saves an mp4, `--stochastic` samples instead of taking the mean action. |
| `scripts/transfer_benchmark.py` | :material-account: | Exercises 3 and 4. Scores each named run on the healthy Ant, all four one-leg injuries and all six two-leg injuries, on identical episode seeds, and reports the drop against healthy. Writes `benchmark.csv` and `benchmark.png`. |
| `scripts/render_agent.py` | :material-account: | Video of any checkpoint. Walks up from the `.zip` to find the run's Hydra config, rebuilds the training environment and loads the matching `VecNormalize` statistics — so it renders what was actually trained, not a guess. |

---

## Session #3 — AirTraffic

`cd sessions/03-advanced/airtraffic` first.

| Script | Who | What it does |
|---|:--:|---|
| `scripts/train_4d.py` | :material-account: | PPO on `Flight4DEnv` with the autoregressive masked policy. Every flag is documented in [training](../advanced/training.md). Always prefix `OMP_NUM_THREADS=1`. |
| `scripts/score_4d.py` | :material-account: | **The one you actually need.** Rolls agents over a fixed held-out seed block and prints the KPIs the competition ranks on, with `noop`, `random` and `rule-based` in the same table. `--agent` repeats; `--seeds` sets how many (default 100), `--seed0` where the block starts. See [evaluation](../advanced/evaluation.md). |
| `scripts/render_4d.py` | :material-account: | Episodes to video. The environment declares no render modes on purpose, so this drives it from outside and replays through `CollisionCourseSimulator`'s shared geometry. Output lands in `renders/` (gitignored). |
| `scripts/calibrate_4d.py` | :material-account: | Sizes the reward terms against each other **before** you spend a training run. Run this whenever you touch a weight — the original weights were 86:1 in favour of acting and it was knowable in thirty seconds. `--csv` writes the figure data. |
| `scripts/callbacks_4d.py` | :material-school: | Not a CLI. The training callbacks — `AutoEntropyCallback` (tunes `ent_coef` so entropy tracks a target fraction of `ln(n_actions)`) and `CurriculumCallback4D` (widens the per-episode traffic draw over training). Imported by `train_4d.py`. |

!!! note "Two different seed blocks, and they are not interchangeable"
    `score_4d.py` defaults to 100 seeds from `--seed0 900000`. The competition
    harness uses `competition/seeds_public.txt`, a different list entirely. Both
    are honest held-out blocks; numbers from one do not transfer to the other.
    Whenever you quote a KPI, quote the seed list with it.

---

## Session #3 — the competition harness

`cd sessions/03-advanced` first — the package is `competition`, not
`airtraffic/competition`.

| Command | Who | What it does |
|---|:--:|---|
| `python -m competition.run submit` | :material-account: | Roll an agent out on the public seeds and write `submission.csv` — the action trace the public leaderboard is built on. `--agent` takes `noop`, `random`, `rule-based`, or a path to an SB3 `.zip` — the same names `score_4d.py` takes. |
| `python -m competition.run replay` | :material-account: | Score a `submission.csv` exactly the way Kaggle will. Run it before uploading. |
| `python -m competition.run compare` | :material-account: | Rank several agents head to head in one table. `--limit 20` for a quick check. |
| `python -m competition.run postmortem` | :material-account: | Every diagnostic pass on one or more agents, and the trophies that follow. See below. |
| `python -m competition.run decode` | :material-account: | Turn a raw number off the Kaggle leaderboard back into `failed`/`congestion`/`exit_miss_bucket`/`clearances` — the exact inverse of `kaggle.metric.pack`, for when the packed score needs reading rather than sorting. |
| `python -m competition.run score-zips` | :material-cog: | **The private board.** Scores every submitted zip in a directory on the secret seeds, runs the contract suite as a gate, and prints one leaderboard. Each zip gets its own process — every team ships a directory called `envs`, and two of them in one interpreter would silently give the second team the first team's code. |
| `python -m competition.ingest` | :material-account: :material-cog: | One zip, printing JSON. What `score-zips` spawns; organisers run it to debug a single rejected submission, and **participants run it as their dry run** — `--contract` for the gate, `--out submission.csv` to write the Kaggle trace from a policy that observes its own environment (`run submit` only knows the stock one). |
| `python -m competition.make_seeds` | :material-cog: | Regenerate a seed list from a salt. The private list is derived from `$COMPETITION_SALT` and deliberately never committed, so it can be reproduced without ever leaking. |
| `python -m competition.kaggle.export` | :material-cog: | Write the Kaggle solution file: every scored aircraft's initial state, plus the `Usage` column that splits Kaggle's live board from its final one. Full runbook: [the Kaggle leaderboard](../advanced/kaggle.md). |

Global flags on `competition.run`: `--private` (the secret seed list),
`--seeds FILE`, `--limit N`, `--stochastic`, and `--n-lo/--n-hi` to narrow the
traffic draw. The last two are **development only** and every run that uses them
says so on every line of output.

!!! tip "Every flag of every command, in the order you would run them"
    The [organiser runbook](../advanced/organisers.md) is the full reference —
    the three phases of running the competition, a complete flag table per
    command, the environment variables, and which files are secret.

### The modules behind the CLI

Not scripts, but this is where to look when a number surprises you.

| Module | What lives there |
|---|---|
| `competition/config.py` | The frozen scenario and `SCORE_VERSION`. Changing anything here voids the board, which is why it is one file rather than scattered CLI defaults. |
| `competition/score.py` | The KPIs, the aggregation, and the lexicographic rank key. We own this; you own your reward. |
| `competition/rollout.py` | `rollout_agent` (needs the agent, so needs torch), `replay_actions` (numpy only, because it has to run in Kaggle's sandbox) and `rollout_submission` (a zip's policy observing its env, ours scoring). The scored class comes from `_paths.reference_envs()` — the repository's `envs` loaded by file path, so a shipped `envs/` cannot shadow it. |
| `competition/submission.py` | The action-trace CSV: write, read, validate. |
| `competition/agents.py` | Loading the thing being scored. Imports SB3 lazily so the replay path stays dependency-light. |
| `competition/perturb.py` | The wrappers that drive the post-mortem passes, each presenting the same `predict()` interface as what it wraps. |
| `competition/ingest.py` | Unpacking, validating and loading a submitted zip. Also documents the boundary: a team's environment supplies observations, never KPIs. |
| `competition/kaggle/metric.py` | The custom Kaggle metric — a deliberate *second* implementation of the rules, pandas and numpy only. `tests/test_kaggle_metric.py` is the only thing stopping it drifting from the first. |

### The post-mortem passes

```bash
python -m competition.run --private postmortem \
    --agent noop --agent rule-based --agent submissions/team-x/policy.zip
```

| Pass | The question it asks |
|---|---|
| **do-nothing** | Does it beat the brick? Below that line is worse than not being there. |
| **stochastic** | How much of the score survives sampling instead of `argmax`? |
| **next-best** | Swap every decision for the runner-up. How much rested on thin margins? |
| **ε-random** | Corrupt actions with ε ∈ {0.05, 0.1, 0.25}. A robustness curve. |
| **best-extraction** | *k* stochastic rollouts per seed, keep the best. How good is it when we try? |

Heuristics skip the passes that need a distribution and say so, rather than
reporting a fabricated one — which is also why the *Iron Stomach* trophy
excludes them.

---

## Tests

Every package carries its own `pytest.ini`, so run them from inside it:

```bash
cd sessions/03-advanced && python -m pytest competition/tests -q   # 80 tests, ~8 s
cd sessions/03-advanced/airtraffic && python -m pytest -q
cd sessions/02-intermediate/crippled-ant && python -m pytest -q
```

Two suites are load-bearing rather than routine:

- **`airtraffic/tests/test_env_contract.py`** — a submitted environment must pass
  this to be scored at all. Red means unscored, with no manual fixes on our side.
  Point it at any class without editing `conftest.py`:
  `RLB_ENV_ENTRY_POINT=envs.my_env:MyFlightEnv python -m pytest tests/test_env_contract.py`
  — which is exactly how the harness invokes it.
- **`competition/tests/test_kaggle_metric.py`** — asserts the Kaggle metric and
  the real environment agree **exactly** on every KPI. The metric is a second
  implementation of the rules living in a sandbox that cannot import the first,
  so drift is the standing risk and equivalence is the standing test.

## Related

- [Environment API](environment-api.md) · [Reward design](reward-design.md) · [Rule-based controller](rule-based-controller.md)
- [The Kaggle leaderboard](../advanced/kaggle.md) · [Competition rules](../advanced/competition.md)
