---
title: The Kaggle leaderboard
---

# The Kaggle leaderboard

The public board runs on Kaggle all session. You roll your agent out on the
**published** seeds, submit the actions it chose, and Kaggle re-flies them
through the frozen simulator and scores the result. There is no number in the
file to inflate.

The [competition rules](competition.md) say what you are ranked on. This page is
the mechanics: how to produce a submission, how to upload it, and — at the
bottom — how an organiser sets the whole thing up.

!!! example "This year's competition"
    <https://www.kaggle.com/t/a0a74b5818114812a2b076b172dfe197> — the join
    link. It is unlisted, so this is the only way in; there is no search
    result and no public listing to find it by.

!!! info "This page is the public board only — the trophies are decided elsewhere"
    Kaggle is where you **publish a result and compare it live** against the
    brick and the rule-based baseline: self-service, all session, upload
    `submission.csv` as often as you like, watch your row move. It is a
    scoreboard, not the verdict.

    The verdict is a separate delivery: your zip, dropped once in the
    [shared Drive folder](https://drive.google.com/drive/folders/1Mrv5aZ70LDN5goMj0k67XFGK84LZtBOI?usp=sharing)
    at the sprint deadline, and **scored offline, by us, on our machine** —
    not on Kaggle at all. That is not just about the seeds being secret.
    Every team's zip can carry its *own* environment code, and there is no
    safe way to run twenty different `envs/` packages inside one shared
    Kaggle notebook — so each zip is unpacked and scored in its own process,
    one at a time, locally. Two different deliveries, two different rhythms,
    two different reasons — see [the deliverables table](design-sprint.md#deliverables).

---

## Part 1 — Submitting (participants)

### 1. Build the file

```bash
conda activate rlbootcamp
cd sessions/03-advanced

python -m competition.run submit \
    --agent airtraffic/checkpoints/ppo_4M_curriculum.zip \
    --out submission.csv
```

`--agent` takes `noop`, `random`, `rule-based`, or a path to any SB3 `.zip` —
the same four names `score_4d.py` takes — and it rolls that agent out on the
**stock** `Flight4DEnv`. If your policy observes an
environment you changed, build the file from your zip instead (see
[the custom-environment route](#if-you-changed-the-environment) below). The
command prints your local score as it goes:

```json
{
  "version": "2.0.0",
  "n_episodes": 100,
  "failed_episodes": 10,
  "congestion_total": 176,
  "exit_miss_mean": 0.12875161496894288,
  "clearances_total": 2089,
  "aircraft_involved_total": 477,
  "rank_key": [10, 176, 26, 2089, 477]
}
```

The file is 5 001 lines — one header plus 100 seeds × 50 steps — and looks like
this:

```csv
id,action
s12146695_t00,1
s12146695_t01,1
```

!!! info "Why every seed carries all 50 rows"
    Most episodes end early: the brick ends nearly all of them in a mid-air
    inside a dozen steps. The rows past that point are written as the idle
    action so the file covers the whole scored horizon — Kaggle's solution file
    is exported at the full horizon and rejects a submission that does not match
    it. Locally the padding is invisible; the replay idles through the tail
    either way.

#### Every flag, while you are here

Global flags go **before** the subcommand: `python -m competition.run --limit 20 submit ...`.

| Flag | On | What it does |
|---|---|---|
| `--limit N` | any | only the first `N` public seeds — a quick check, **not** a valid upload (Kaggle wants all 100) |
| `--stochastic` | `submit`, `compare` | sample actions instead of taking the argmax; what the *Iron Stomach* trophy measures |
| `--seeds FILE` | any | an explicit seed file instead of the public list |
| `--n-lo / --n-hi` | any | narrow the converging-traffic draw — **development only**; every line of output is stamped off-leaderboard |
| `-v` | any | per-seed progress |
| `--private` | any | the secret list — organisers only, the file is not in the repository |
| `submit --agent A --out F` | | `noop` \| `random` \| `rule-based` \| path to an SB3 `.zip`; default `submission.csv` |
| `replay --submission F --strict` | | `--strict` also rejects a file carrying seeds outside the list (missing seeds are always an error) |
| `compare --agent A [--agent B ...]` | | one ranked table, same rank key as the board |
| `postmortem --agent A [...] --extraction-k K` | | every diagnostic pass; `K` stochastic rollouts per seed for best-extraction (default 8) |
| `decode SCORE` | | a leaderboard number back into its four KPIs |

`python -m competition.ingest --zip Z [--contract] [--limit N] [--workdir D] [--out F]`
is the zip's counterpart: the same gate the private board runs, on the public
seeds, and with `--out` the action trace written as a Kaggle submission.

#### If you changed the environment

`submit` cannot roll out a policy that reads observations from an environment
of your own — it only knows the stock one. Zip your submission as for the
[private board](competition.md#build-the-zip) and let the harness fly it in
lockstep with ours:

```bash
python -m competition.ingest --zip my-team.zip --contract --out submission.csv
```

Your environment observed, ours scored, the trace written at the full horizon.
Steps 2 and 3 are then the same.

### 2. Check it yourself first

```bash
python -m competition.run replay --submission submission.csv
```

This is the same replay Kaggle runs. If the numbers here differ from what
`submit` printed, something is wrong with the file and not with your agent —
say so rather than uploading it.

### 3. Upload

Open [the competition page](https://www.kaggle.com/t/a0a74b5818114812a2b076b172dfe197)
→ **Submit Prediction** → drop `submission.csv` in →
add a description (use it: "relative obs, 2M steps" is worth more to you in
three hours than "submission 7") → **Submit**.

Scoring takes a few seconds. **Lower is better** — the score is the four ranked
objectives packed into one float, so a difference in safety dominates any
possible difference in timeliness, which dominates any difference in
efficiency. You cannot trade one for another; that is the whole point.

Teams of two; **20 submissions per team per day**. Submit the rule-based agent
first — it takes two minutes, puts a row with your name on the board, and
proves the whole pipeline before anything you trained is ready.

!!! tip "What a good number looks like"
    On the 60 public-split seeds, the three reference agents score:

    | agent | Kaggle score | what it is |
    |---|---:|---|
    | PPO, 4M steps | `1.20 × 10¹³` | the shipped checkpoint |
    | rule-based | `3.60 × 10¹³` | the bar you have to clear |
    | noop | `2.40 × 10¹⁴` | the brick |

    The magnitudes look absurd because the number is a packed lexicographic
    key, not a score anyone reads directly. Only the **ordering** means
    anything. Read your actual KPIs from `competition.run replay` — or, if all
    you have is the number off the leaderboard, decode it directly:

    ```bash
    python -m competition.run decode 40026885486301
    ```
    ```json
    {
      "failed": 10,
      "congestion": 176,
      "exit_miss_bucket": 26,
      "clearances": 2089
    }
    ```

    `kaggle.metric.unpack` is the exact inverse of `pack` — a mixed-radix
    decode, not an approximation — so this reads back precisely what went in,
    except `exit_miss_bucket` (rounded to the nearest 0.005 before packing;
    the command prints an approximate mean alongside it). A component sitting
    exactly on its `BOUND_*` constant may have been clamped rather than
    measured — worth checking before reading it as a real count.

### 4. Read the board honestly

Kaggle splits the published seeds into a live half and a hidden half of its own,
so your position moves once at the end even on the public board. And **that is
not the ranking that decides the trophies** — the private board runs your zip on
our machine, on 100 seeds you have never seen. Expect all three numbers to
disagree; the gap between them is the most useful thing you will take home.

---

## Part 2 — Setting it up (organisers)

!!! warning "Do this before the session, not during it"
    Every step below has been run end to end from a clean checkout, and the
    metric has been checked against the four constraints Kaggle's notebook
    enforces. What is *not* verified is the competition-creation form itself,
    which moves between editions. Budget an hour.

    If you read one line on this page, read the **Leaderboard Sort Order** row
    below. It defaults to *higher is better*, and ours is not.

### What you upload

Three files, all produced from this repository:

| File | Produced by | Size |
|---|---|---|
| `solution.csv` | `competition.kaggle.export` | 5 000 rows, 1.8 MB |
| `sample_submission.csv` | a `noop` submission — all-zero actions | 5 001 rows, 92 KB |
| `metric.py` | committed at `competition/kaggle/metric.py` | — |

```bash
conda activate rlbootcamp
cd sessions/03-advanced

# 1. the solution: every scored aircraft's initial state, with Kaggle's split
python -m competition.kaggle.export --holdout-frac 0.4 --out solution.csv
#    -> wrote 5000 rows for 100 seeds
#       60 seeds Public, 40 Private

# 2. the sample submission: the brick, which is a valid submission of all zeros
python -m competition.run submit --agent noop --out sample_submission.csv
```

!!! danger "`--private` and `--holdout-frac` are different things"
    `--holdout-frac` splits the **published** seeds into Kaggle's own live and
    final halves. `--private` exports the **secret** seed list, which decides
    the trophies and must never be uploaded anywhere. If you run
    `export --private` by accident, the tool says so on the way out — but the
    file is still on your disk, so delete it.

### What Kaggle's private split actually is

It is a **chronological** holdout, not an informational one: 40% of the
*published* seeds, hidden from view until the competition closes. Every one of
them is in `competition/seeds_public.txt`, committed in this repository, and a
participant can score all 100 locally whenever they like.

**It could not work any other way.** A submission here is an action trace keyed
by seed — to produce the rows for a seed you must roll out on it, and to roll
out on it you must know it. A seed nobody can submit for is a seed Kaggle cannot
score. So an action-trace competition cannot have secret seeds at all.

That is precisely why the trophies are decided by *your zip, our machine*: we run
the policy ourselves, so the secret seeds never have to leave our side. The two
mechanisms are not redundant, they answer different questions:

| | Kaggle's private split | Our private board |
|---|---|---|
| Seeds | 40 published ones | 100 secret ones |
| Protects against | reading the final rank early | training against the ranked seeds |
| Run by | Kaggle, on an action trace | us, on the participant's policy |

!!! question "Is the Kaggle split worth having, then?"
    It buys one real thing: the rank moves once at the close, which is an honest
    small lesson about trusting a leaderboard you tuned against. It costs the
    live board 40% of its signal.

    `0.4` is a defensible middle. Lower it toward `0.2` if you would rather the
    live board be less noisy; set `0` and Kaggle simply has no final split.
    **If a solution file is already uploaded and verified, leave it alone** — the
    difference is marginal and re-uploading a solution is not free.

### Creating the competition

!!! example "Already done for 2026"
    The competition exists at the join link above. The steps below are the
    runbook for creating it — read them to know what is already set, or to
    recreate the competition for a future edition.

1. **kaggle.com/competitions → Create a Community Competition.** Set the host to
   the bootcamp organisation if you have one; your own account works otherwise.
2. **Visibility.** Choose *unlisted* and share the join link with the room. A
   public listing invites the internet into a teaching exercise.
3. **Data.** Upload `solution.csv` as the solution file and
   `sample_submission.csv` as the sample. Kaggle reads the public/private split
   from the `Usage` column and **strips that column before the metric sees it**.
4. **Evaluation.** The custom metric is a **Metric Notebook**, not a file
   upload — see the settings below.
5. **Settings.** Submissions per day: 20 is generous and stops nobody. Team
   size: 2, matching the design-sprint pairs. End date: the evening of day 3.
6. **Description.** Point at [the rules](competition.md) rather than restating
   them; a second copy of the scoring rules is a second thing to keep in sync.

!!! warning "The Data tab also wants an actual Dataset — separate from Solution/Sample"
    Beyond the solution file and sample submission, Kaggle's launch checklist
    for a Community Competition wants a **Dataset** attached to the Data tab in
    its own right — a platform requirement, not something this repository's
    harness produces or needs. Upload `seeds_public.txt` and
    `sample_submission.csv` there: both are already public, and together they
    let someone browsing Kaggle see the exact scenario list and submission
    format before cloning anything.

    **Rewrite the pre-filled "Dataset Description" before publishing.** It
    ships seeded with boilerplate about `train.py`, `evaluate.py`,
    `default.yaml` and `viz.py` — none of which exist in this repository. Left
    as-is, it actively contradicts [the rules](competition.md) and Part 1 of
    this page. Replace it with a short pointer to the submission flow above:
    build `submission.csv` with `competition.run submit`, verify it with
    `replay`, upload it under **Submit Predictions**. There is no code/zip
    upload for the public leaderboard — that only happens for the private
    board, offline.

### The metric notebook

Paste the whole of `competition/kaggle/metric.py` into the notebook. It is
already self-contained — pandas, numpy and the standard library, nothing else —
which is the reason it is a second implementation of the rules rather than an
import of the first.

Then, in the panel on the right:

| Field | Set it to | Why |
|---|---|---|
| **Name** | *(the notebook's own title)* | Saving renames the metric to match the notebook title, so title the notebook first. |
| **Description** | 255 characters, see below | Shown next to the metric. |
| **Category** | `Other` | Nothing in the list describes a packed lexicographic key. |
| **Leaderboard Sort Order** | **turn "Higher score is better" OFF** | :material-alert: **The one setting that silently inverts the competition.** It defaults to *on*. Our score is lower-is-better. |
| **Pass Complete Submission** | either — `Off` is fine | Verified both ways: the metric scores only the seeds present in the solution it was handed, so the toggle does not change a single score. |

```text title="Description, 246 characters"
Air traffic conflict resolution. Participants submit the action taken at every
(seed, step); the metric re-flies each episode and ranks lexicographically on
failures, congestion, 4D exit miss and clearances, packed into one float.
Lower is better.
```

!!! danger "Do not follow the template's `del solution[row_id_column_name]`"
    Kaggle's starter metric deletes the id column, because most metrics only
    need the columns either side of it. **Ours needs it**: the id *is* the data
    — `s0000042_t07` names the seed and the step, and the replay cannot be
    reconstructed without both. Our `score()` never deletes it.

!!! info "What Kaggle does before your metric runs"
    - It **aligns** solution and submission on the id column, so row order in
      the uploaded file does not matter. Verified: shuffling the submission
      changes nothing.
    - It **removes the `Usage` column**, then calls the metric once per split.
    - It validates the submission's rows against `sample_submission.csv` first,
      which is the other reason every seed must carry the full 50-step horizon.

!!! tip "Save & Validate runs the examples"
    The `score()` docstring carries **twelve doctests** — two worked scores and
    two rejection messages, on a two-aircraft head-on sector small enough to
    read. They are executable documentation of the submission format, and they
    run in both contexts (`test_score_docstring_is_the_host_facing_description`
    checks that they still pass and that the docstring stays under Kaggle's
    8 000-character cap).

    That docstring is also **what a host sees when selecting the metric**, so it
    leads with the ranking and the direction rather than with implementation
    notes.

!!! note "The 30-minute scoring limit is not close"
    Metric notebooks are capped at 30 minutes per scoring run. Ours takes
    **0.04 s** for a 60-seed split and 0.02 s for a 40-seed one — the replay is
    pure numpy over 5 000 rows. There is no reason to optimise it and no reason
    to fear a timeout.

### The dry run — do this before you publish

Everything the metric does can be checked locally, and it should be, because a
metric that is wrong on Kaggle is wrong silently.

```bash
cd sessions/03-advanced
python -m pytest competition/tests -q          # 80 tests, ~8 s
```

Then score all three reference agents through the *metric file itself*, not
through the harness:

```python title="dry_run.py"
import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location("km", "competition/kaggle/metric.py")
km = importlib.util.module_from_spec(spec)
spec.loader.exec_module(km)

solution = pd.read_csv("solution.csv")
for name in ("ppo", "rulebased", "noop"):
    submission = pd.read_csv(f"submission_{name}.csv")
    for usage in ("Public", "Private"):
        rows = solution[solution["Usage"] == usage].drop(columns=["Usage"])
        print(name, usage, km.score(rows, submission.copy(), "id"))
```

Verified on 2026-09-03, score v2.0.0, against the committed checkpoint. The KPI
columns are the harness's own, over **all 100** published seeds; the two Kaggle
columns are the packed score over the 60-seed and 40-seed splits respectively,
which is why they are not derivable from each other:

| agent | failed | congestion | `exit_miss` | clearances | touched | Kaggle public | Kaggle private |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO 4M | 10 | 176 | 0.1288 | 2089 | 477 | `1.20 × 10¹³` | `2.80 × 10¹³` |
| rule-based | 20 | 96 | 0.5700 | 727 | 231 | `3.60 × 10¹³` | `4.40 × 10¹³` |
| noop | 100 | 752 | 0.0000 | 0 | 0 | `2.40 × 10¹⁴` | `1.60 × 10¹⁴`  |

The ordering is the same on both splits and matches the harness's lexicographic
ranking. That agreement is the check; the magnitudes are not meant to be read.

Three things that dry run confirms, each of which would have broken the board:

- **The metric and the harness agree exactly** on all four packed components,
  for all three agents. Not to a tolerance — bit for bit.
- **The metric never imports torch.** Kaggle's sandbox has no torch, no
  gymnasium and no bootcamp code; a test asserts the replay path keeps it out of
  `sys.modules`.
- **Each split scores alone.** Handing the metric only the `Private` rows scores
  only those seeds, against the same full submission.

!!! warning "The KPIs above are the *public* seeds, not the board in the deck"
    The checkpoint's README quotes `failed 1` and `exit_miss 0.087` — that is a
    different held-out block of 100 seeds. Both are honest; they are not the
    same seeds, and neither pins a failure rate. Quote the seed list along with
    the number, every time.

### Confirmed live

The pipeline was verified on the real competition with a sandbox submission of
the shipped checkpoint. Kaggle returned:

| | Kaggle returned | Computed locally |
|---|---|---|
| **Public Score** | `12012163199400.000` | `12012163199400` |
| **Private Score** | `28014722826928.000` | `28014722826928` |

Exact, to the digit, on both splits. That single upload confirms the metric
notebook runs, the solution file parses, the `Usage` split is applied, and the
horizon padding satisfies Kaggle's row check — the four things that could each
have failed silently.

It also answers a question this repository could not: **the board renders the
score at full precision**, all fourteen digits plus three decimals. The packed
key is a lexicographic ordering, so two submissions differing only in
clearances — the lowest-weighted component — will still show as different
numbers rather than collapsing to a tie. Worth one glance at the *public*
leaderboard view when the first benchmark row lands, since that is a different
view from the host's sandbox panel.

### Sandbox submissions — what to put there

*Settings → Evaluation Metric → Sandbox Submissions.* A sandbox submission is
scored exactly like a real one but stays private, **unless you tick
`Benchmark`**, which puts it on the leaderboard as a host row.

Upload all three reference agents. Tick `Benchmark` on two of them:

| Upload | Benchmark? | Why |
|---|:--:|---|
| `submission_noop.csv` | **yes** | The brick. The participation bar, and the row the *Brick with Wings* trophy is measured against. It belongs on the board where everyone can see it. |
| `submission_rulebased.csv` | **yes** | The bar that matters. "Beat the brick, then beat the heuristic" is the framing of the whole session, and it lands harder as a row than as a sentence. |
| `submission_ppo.csv` | *recommend no* | Your regression check that the metric still works. See below. |

```bash
cd sessions/03-advanced
python -m competition.run submit --agent noop        --out submission_noop.csv
python -m competition.run submit --agent rule-based  --out submission_rulebased.csv
python -m competition.run submit --agent airtraffic/checkpoints/ppo_4M_curriculum.zip \
    --out submission_ppo.csv
```

!!! question "Should the trained agent be a benchmark too?"
    A judgement call, and it goes either way.

    **Against** (the recommendation): `ppo_4M_curriculum.zip` ships in the
    repository, so benchmarking it puts a row on the board that anyone can match
    in two minutes by submitting a file we handed them. The first thing the room
    learns is then "the top row is free", which is not the lesson.

    **For:** the deck's honest finding is that *the learner does beat the
    heuristic here* — the first environment in the bootcamp where that holds. If
    the board shows only `noop` and `rule-based`, a room that never gets a
    trained agent above the heuristic may leave with the opposite impression.

    A middle path: keep it as a private sandbox row during the sprint, and tick
    `Benchmark` in the last hour so the closing discussion has it on screen.

!!! tip "The sandbox is also where to take screenshot #5"
    Truncate a file — `head -2000 submission.csv > broken.csv` — and upload it
    as a sandbox submission. You get the participant-visible rejection message
    without putting a broken row anywhere public.

### On the day

- Have `submission.csv` for `rule-based` ready to upload as a **starter row**,
  so the board is not empty when the room opens. It is also the bar, which
  makes it the right first entry.
- The public seed list is committed, so anyone can tune against it. That is
  intended. The private ranking is what the trophies follow.
- Regenerate the private seeds from the salt on the day, and keep the salt out
  of the repository:
  ```bash
  python -m competition.make_seeds --which private --salt "$COMPETITION_SALT"
  ```

---

## Screenshots to take

The Advanced deck now carries a three-frame Kaggle walkthrough, and each frame
has a screenshot slot that is **already wired up**: save the PNG as
`slides/figures/<name>.png` and it appears on the next `make advanced`. Until
then the deck builds with a labelled placeholder naming the missing file, so
nothing is silently blank.

| # | Filename | Used by | What to capture | Why it earns a slide |
|---:|---|---|---|---|
| 1 | `kaggle_join.png` | — | The competition landing page, **Join Competition** button visible, rules tab in shot | The one thing everybody has to do first, and the one that silently blocks a submission if skipped |
| 2 | `kaggle_data_tab.png` | **deck** | The **Data** tab showing `sample_submission.csv` with its first rows expanded | Shows the exact two-column format before anyone writes code |
| 3 | `kaggle_submit_dialog.png` | — | The **Submit Prediction** drop zone, with a file selected and the description box filled in with something real | Where people get stuck; also models the habit of describing a submission |
| 4 | `kaggle_leaderboard.png` | **deck** | The public leaderboard with the three reference rows on it (noop, rule-based, PPO), **before** the room submits | The bar, visible, with the brick at the bottom — "beat the brick" made concrete |
| 5 | `kaggle_score_error.png` | **deck** | A deliberately broken submission's error message — truncate a file and upload it | Turns "read the error" into something they have already seen once |
| 6 | `kaggle_my_submissions.png` | — | The **My Submissions** tab with several entries and their descriptions | Makes the point that the board is a log, not a single attempt |
| 7 | `kaggle_metric_settings.png` | — | The metric notebook's settings panel, with **Leaderboard Sort Order** switched off | Organiser-facing: the one toggle that inverts the whole competition, captured in the state it should be in |

The three marked **deck** are the ones with slots waiting. Numbers 1, 3 and 6 are
worth taking at the same time — they cost nothing extra while you are in the UI,
and they are what you will want if anyone asks for a written walkthrough.

!!! tip "Take #5 on purpose, before the session"
    Cut a submission down with `head -2000 submission.csv > broken.csv` and
    upload it. The metric answers *"Submission is missing 3000 of 5000 required
    rows. Rebuild it with `python -m competition.run submit`."* — that message
    exists precisely so a participant can fix it without asking, and it is worth
    showing them once while nothing is at stake.

Take them against the real competition once it exists, in **light mode**, at a
window width of about 1400 px, and crop to the content — the decks are 16:9 and
a full-screen 4K capture reduces to unreadable mush.

## Related

- [Organiser runbook](organisers.md) — every command, in the order you run it.
- [Competition rules](competition.md) — what the score means.
- [The design sprint](design-sprint.md) — what you change to move it.
- [Evaluation](evaluation.md) — scoring locally, before you ever submit.
- [Scripts and tooling](../reference/tooling.md) — every command in the repo.
