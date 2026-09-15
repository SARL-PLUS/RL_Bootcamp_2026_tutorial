# Flight Challenge — scoring harness

We own the score. Students own their environment and their policy.

```
competition/
├── config.py        # the frozen scenario + SCORE_VERSION — changing these voids the board
├── score.py         # KPIs, aggregation, lexicographic ranking
├── rollout.py       # rollout_agent (needs the agent) / replay_actions (numpy only)
├── submission.py    # the action-trace CSV: write, read, validate
├── agents.py        # noop | random | rule-based | path to an SB3 .zip
├── seeds.py         # public + private seed lists
├── make_seeds.py    # regenerate either list from a salt
├── ingest.py        # a submitted zip: unpack, validate, load, score one
├── postmortem.py    # the diagnostic passes + trophy computation
├── perturb.py       # next-best / eps-random agent wrappers
├── run.py           # CLI: submit / replay / compare / postmortem
├── seeds_public.txt # 100 seeds, shipped
└── kaggle/
    ├── metric.py    # the custom evaluation metric — upload this to Kaggle
    └── export.py    # writes the solution CSV (scenario initial states)
```

## Use

```bash
cd sessions/03-advanced

# rank the reference agents (quick check: --limit 20)
python -m competition.run --limit 20 compare --agent noop --agent rule-based

# build a submission from an agent
python -m competition.run submit --agent rule-based --out submission.csv

# score one the way the public leaderboard does
python -m competition.run replay --submission submission.csv

# a team that changed its environment: the zip, flown in lockstep with ours on
# the public seeds, and the trace written out as their Kaggle submission
python -m competition.ingest --zip my-team.zip --contract --out submission.csv

# organisers: the ranking that decides the trophies
python -m competition.make_seeds --which private --salt "$COMPETITION_SALT"
python -m competition.run --private compare --agent rule-based

# the private board: every submitted zip, contract-gated, on the secret seeds
python -m competition.run --private score-zips --dir submissions/

# every diagnostic pass, and the trophies that follow from them
python -m competition.run --private postmortem \
    --agent noop --agent rule-based --agent submissions/team-x/policy.zip
```

## The private board

Kaggle takes an action trace, so its seeds must be published — you cannot submit
actions for a seed you were never given. The board that decides the trophies
therefore takes the **policy** instead: a zip, our machine, seeds that never
leave it.

`ingest.py` unpacks one and `run.py score-zips` batches them. Three decisions
worth knowing:

**A team's environment supplies observations, never KPIs.** Their env and ours
run in lockstep on the same seed; only the action index crosses. Their reward,
termination shaping and info dict are never read. So a wrong environment cannot
buy a better score — it feeds their policy worse information and scores worse,
honestly. There is nothing to defend against, and the code does not pretend to.
The one thing this *does* depend on is that "ours" really is ours:
`_paths.reference_envs()` loads the repository's `envs` by file path under a
private module name, because `ingest` puts the zip's root first on `sys.path`
and a plain `import envs` there resolves to the team's copy.
`test_a_shipped_envs_package_cannot_change_what_is_scored` pins it, in a fresh
process, with a zip whose `_count_congestion` returns zero.

**Our env decides when the episode ends.** If theirs gives up at step 3, ours
idles out the horizon. `failed_episodes` sorts first precisely so ending early
is a penalty; it must not also be a way to stop accumulating congestion.

**One process per zip.** Every submission ships a directory called `envs`.
Import two of them into one interpreter and the second team silently gets the
first team's code — a scoring bug that would be invisible in the output.

## The post-mortem

A leaderboard position is one measurement of one policy under one set of
conditions. These passes ask what the position cannot:

| Pass | Question |
|---|---|
| **do-nothing** | Does it beat the brick? Below that line is worse than not being there. |
| **stochastic** | How much of the score survives sampling instead of argmax? |
| **next-best** | Swap every decision for the runner-up. How much rested on thin margins? |
| **eps-random** | Corrupt actions with probability ε ∈ {0.05, 0.1, 0.25}. A robustness curve. |
| **best-extraction** | k stochastic rollouts per seed, keep the best. How good is it when we try? |

Three things the implementation is careful about:

- **ε perturbation is per aircraft, not per step.** The action factorises across
  flights, so corrupting the whole vector at once would be a far blunter
  instrument than the curve is meant to be.
- **Heuristics skip the passes that need a distribution**, and say so, rather
  than reporting a fabricated one. Their stochastic gap is zero because they have
  nothing to sample from — which is why `Iron Stomach` excludes them. A trophy
  for robustness must not go to something that was never perturbed.
- **`best_extraction` seeds torch explicitly.** Sampling draws on the global RNG;
  without that, "best of k" changes between runs and cannot be quoted.

`award_trophies` computes what follows from numbers. **Black Box** (best QA
writeup) and **Icarus** (boldest failure) are deliberately absent — they are jury
calls, and pretending a metric decides them would be worse than admitting a human
does.

### A finding worth expecting

Random perturbation makes the do-nothing baseline *safer*: at ε=0.25 its
congestion drops from 253 to 146 over 8 seeds. Random clearances scatter aircraft
across flight levels, and scattered aircraft do not conflict. Any participant who
reads "my agent got safer when I added noise" as proof of robustness has found
this effect, not a good policy — which makes it worth showing them on purpose.

## How the ranking works

Lexicographic, every component lower-is-better:

```
(failed_episodes, congestion_total, timeliness_bucket, clearances_total, involved_total)
 └ safety ──────────────────────┘  └ timeliness ────┘  └ efficiency ─────────────────┘
```

Three design decisions worth knowing before you change anything:

**`failed_episodes` is first, and that placement is load-bearing.** An episode
that ends early — invalid-action overflow, or the clearance budget running out —
accumulates fewer congestion events purely by existing for less time. Rank on
safety alone and crashing out becomes a strategy. Note this is *not* the
`truncated` flag: the environment sets that both at the horizon and when the
clearance budget is exhausted, so the step counter is the only unambiguous test.

**Timeliness is bucketed** (`TIMELINESS_BUCKET`). It is a float, so exact ties
never occur, so efficiency would never break one and the third objective would be
decoration. Rounding to a resolution finer than anyone can control for restores
real ties.

**Replay must stay numpy-only.** The public leaderboard runs `replay_actions`
inside Kaggle's metric sandbox, where torch is not available. `agents.py` imports
SB3 lazily for exactly this reason, and a test asserts the replay path never
pulls torch into `sys.modules`.

## Why an action trace and not a score

A submission is the action taken at every (seed, step, aircraft). There is no
aggregate number in the file to inflate: the score comes from re-flying the
actions through the frozen simulator. A trace that was not produced by a good
policy does not replay like one.

`rollout_agent` and `replay_actions` are verified to agree to the last digit of
`timeliness_mean` — that equality is what makes the public board trustworthy.

## Reference numbers

All 100 public seeds, deterministic, score v2.0.0:

```
 #  submission             failed  congestion   exit miss  clearances  touched
------------------------------------------------------------------------------
 1  ppo_4M_curriculum          10         176      0.1288        2089      477
 2  rule-based                 20          96      0.5700         727      231
 3  random                     76         667      0.7012        2947      497
 4  noop                      100         752      0.0000           0        0
```

The brick is perfectly on time and perfectly cheap — it never touches anything —
and loses anyway, because safety is compared first. That is the ranking working.

These are **not** the numbers in `slides/data/air_4d_board.csv`, which scores a
different held-out block. Both are honest; quote the seed list with the number.


## Kaggle setup (organisers)

```bash
cd sessions/03-advanced

# the solution: initial states + Kaggle's own public/private split of the
# published seeds (the SECRET seeds are a separate list and stay off Kaggle)
python -m competition.kaggle.export --holdout-frac 0.4 --out solution.csv
#   -> 5000 rows, 1.8 MB; 60 seeds Public, 40 Private

# the sample submission: the brick is a valid all-zero submission
python -m competition.run submit --agent noop --out sample_submission.csv
```

Then in the Kaggle competition settings: upload `solution.csv` as the solution
file and `sample_submission.csv` as the sample, upload
`competition/kaggle/metric.py` as the custom evaluation metric, and set the
leaderboard to **lower is better**.

The full runbook — the creation form, the screenshots to take, and the dry run
to do before publishing — is `docs/advanced/kaggle.md`.

### Submissions cover the whole horizon, always

`write_submission` idles every trace out to `max_steps`. Locally the padding is
invisible (the replay idles through the tail either way), but the solution file
is exported at the full horizon and the metric **rejects** a submission missing
any row. Without the padding every real submission was refused, because every
real agent ends at least one episode early — the brick ends nearly all of them.
`test_score_accepts_traces_from_episodes_that_ended_early` is the regression.

### Why the scenarios live in the solution file

`metric.py` cannot import this package — Kaggle's metric sandbox has no
gymnasium, no matplotlib and no bootcamp code. So the metric is necessarily a
**second implementation** of the rules, and the standing risk is that it drifts
from the first one.

Shipping the aircraft's initial states in the solution file removes the largest
piece of that risk: the metric never regenerates a scenario, it re-flies one it
was handed. What remains — clearance application, straight-line motion,
congestion detection, exit deviation — is pinned by
`tests/test_kaggle_metric.py`, which asserts the metric and the real environment
agree on every KPI across four action distributions and the rule-based
controller. **If you change the environment's dynamics or KPIs, that test is what
tells you the metric went stale.**

The solution repeats each aircraft's initial state on all 50 step-rows so it
aligns row-for-row with the submission. The metric reads it only from the `t00`
rows; the rest is alignment padding.

### One float, four objectives

Kaggle sorts a single number. The competition ranks lexicographically, so
`pack()` folds the top four components into one float: each is clamped to a
documented bound and weighted by the product of all lower-priority ranges, so a
difference in a higher-priority objective always dominates any possible
difference below it. The span stays under `2**53`, which makes the packed value
exact in float64 — verified by test, because a silent precision loss here would
reorder the board without any error.

The fifth component, **distinct aircraft touched**, does not fit that budget. It
breaks ties only on the private leaderboard. Two submissions identical on the
four public components tie on Kaggle; they will not tie at the ceremony.
