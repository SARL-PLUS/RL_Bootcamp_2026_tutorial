---
title: Changelog
---

# Changelog

All notable changes to the **tutorial code and its documentation** are recorded
here. The two are versioned together — a code change ships with the matching docs
update in the same version.

The version lives in the repo-root [`VERSION`](https://github.com/SARL-PLUS/RL_Bootcamp_2026_tutorial/blob/main/VERSION)
file and is shown in the docs header.

## Versioning policy

We use a [SemVer](https://semver.org/)-style `MAJOR.MINOR.PATCH`:

| Bump | When | Who |
|---|---|---|
| **PATCH** `0.0.0 → 0.0.1` | Docs edits, small fixes, refactors with no API change. | Routine, applied with the change. |
| **MINOR** `0.0.0 → 0.1.0` | New runnable feature (e.g. Crippled-Ant wrapper, Hydra sweep, action masking) or an API change. | Routine, applied with the change. |
| **MAJOR** `0.x → 1.0.0` | "Everything works and is tested." | **Granted by Leander only.** |

While the project is pre-`1.0.0`, expect things to move.

---

## [Unreleased]

Nothing outstanding.

## [0.35.0] — 2026-09-15

### Added — `random` is an agent the harness knows

`competition.run` accepted `noop`, `rule-based` or a checkpoint path while
`scripts/score_4d.py` also accepted `random`, so `--agent random` on the harness
fell through to `PPO.load("random")` and died on *No such file: random.zip*.
`agents.RandomLegalController` now ships beside the brick — uniform over the
mask read off the observation, never an invalid action, seeded from each
episode's first observation so its row is reproducible and its deterministic
and stochastic post-mortem passes are the same pass. The four names are now
the same in both tools, and an unknown name says so instead of hunting for a
zip. On the 100 public seeds: 76 failed, 667 congestion, `exit_miss` 0.701,
2947 clearances.

## [0.34.0] — 2026-09-15

### Fixed — the scored environment really is ours now

`competition.ingest` puts a submitted zip's root first on `sys.path` before it
imports `competition.rollout`, and `rollout` bound its scored environment with
a plain `from envs import Flight4DEnv`. For any zip that shipped an `envs/`
package — i.e. exactly the submissions the lockstep design exists for — that
resolved to the **team's** copy, so every KPI on the private board would have
been read from an environment the team wrote. The generic contract suite
does not compare against our physics, so a rewritten `_count_congestion`
sailed through the gate. Found while checking the deck's "only the action
index crosses the boundary" claim against the code.

`competition/_paths.py` now exposes `reference_envs()`, which loads the
repository's own `envs` by file path under a private module name;
`rollout.make_env` and `submission.N_CLEARANCES` read from it, so nothing on
`sys.path` can shadow the scored environment. The existing test built its
submission in-process and never hit the import path;
`test_a_shipped_envs_package_cannot_change_what_is_scored` now runs `ingest`
in a fresh process on a zip whose environment reports zero congestion, and
asserts it scores what the stock environment says.

### Added — `competition.ingest --out submission.csv`

`competition.run submit` rolls a checkpoint out on the stock environment, so a
team that redesigned its observation had no route to the public board short
of writing its own loop. `ingest --out` writes the action trace of a zip flown
in lockstep — the team's environment observed, ours scored — on the public
seeds, padded to the full horizon as Kaggle requires. Refused with
`--private`, since a private trace cannot be uploaded anywhere.

### Changed — the participant docs now carry the whole pipeline

[The Kaggle page](advanced/kaggle.md) lists every flag of `competition.run`
and `competition.ingest` in one table, the custom-environment route, and the
team/submission limits; [the rules](advanced/competition.md) gain a
step-by-step "build the zip" and state the policy explicitly: **Kaggle is how
you submit, the zip is how we check** — if the top of the board is close, the
teams involved are asked for code and zips.

## [0.33.0] — 2026-09-14

### Added — two new Session #3 decks, and the action vocabulary one clip at a time

`slides/advanced-intro.tex` (ninety minutes) and `slides/advanced-challenge.tex`
(twenty to thirty) replace the `advanced` + `litmus` pair for the session
without removing either — both older decks still build unchanged, and the
Makefile now knows all five.

The intro deck opens with **why this problem**: Eurocontrol's long-term
outlook (`make_figures_intro.py`, numbers transcribed from the cited
publication), the controller as the bottleneck, and the two SESAR 3 JU
projects the package grew out of — TADA and ASTRA — before stating plainly what
`airtraffic` removes from the real problem (routes, headings, pilot latency,
fuel) and the one secondary objective it keeps in place of fuel. The simulator
and MDP module then walks the constants, the conflict rule against the mid-air
rule, the observation, the mask, termination, the reward and the ranking; the
baseline policy gets its own frame — *idle, or which aircraft; then which
clearance*, the two-stage choice the environment sees as one flat index; the
litmus module introduces `Priority4DController` pedantically (assign, steer,
come home; why it refuses the macro; its one knob); and the last module is
about what the current solution still gets wrong — with the first-generation
measurements (the injection probe, the macro training run) labelled as such
rather than passed off as `Flight4DEnv` results. The old appendices survive,
grouped, under `\appendix`.

`scripts/render_4d.py` gained **`--n-flights`** and eight **single-clearance
demos** (`fl_inc`, `fl_dec`, `fl_inc2_t`, `fl_dec2_t`, `spd_dn_t`, `spd_up_t`,
`resume`, plus `noop`), all on seed 2 at `n_flights=2` — the same converging
pair as the handbook's three-resolution demos, with the bystanders removed —
each issued to aircraft 0 at step 14. Their outcomes are on
[the environment page](advanced/airtraffic.md#the-action-vocabulary-one-clip-at-a-time)
as GIFs and a table, in the deck as clips, and pinned by
`tests/test_render_4d.py` so the numbers cannot drift silently.

### Fixed — the render's KPI-bar labels

The safety and timeliness bars' labels were rotated y-labels on sliver-wide
axes, so they landed over the main plot on one side and over the neighbouring
bar on the other. They are now x-labels under each bar, with the tick labels
pushed outward (safety reads on its left, timeliness on its right). Every
committed GIF and staged poster/filmstrip was regenerated.

Also noticed while building: a Beamer frame whose body *starts* with a brace
group (`{\scriptsize ... \begin{tabular}`) has that group parsed as the frame
**subtitle**, and the table vanishes without a warning. The older
`advanced.tex` has one such frame (*Pick one track*); it is left as it was, per
the decision not to touch the old decks, but the new decks open those groups
with `\begingroup` instead and `slides/README.md` now says why.

## [0.32.1] — 2026-09-04

### Added — GitHub Pages plumbing, and the competition's join link

`.github/workflows/docs.yml` builds the handbook with `mkdocs build --strict`
and deploys it through `actions/upload-pages-artifact` /
`actions/deploy-pages` — GitHub's "Deploy from GitHub Actions" Pages mode,
the same one `rl-bootcamp-setup` already uses. No `gh-pages` branch is ever
written to; going public later is a two-step flip (repo visibility, then
Pages source in Settings), not a new deploy mechanism. Dormant until both
are set — until then it just double-checks the handbook builds on every push.

The 2026 competition's join link
(<https://www.kaggle.com/t/a0a74b5818114812a2b076b172dfe197>) is now on
[the Kaggle leaderboard page](advanced/kaggle.md), which is also where the
link belongs — everywhere else that sends someone toward Kaggle already
links to that page rather than repeating the URL.

### Changed — sharper on why the trophies aren't scored on Kaggle

The public/private split was explained as "different seeds" in three places
([kaggle.md](advanced/kaggle.md), [competition.md](advanced/competition.md),
[design-sprint.md](advanced/design-sprint.md)) without saying why the private
run has to happen offline at all. It's not only the seeds: every zip can ship
its own environment code, and a shared Kaggle notebook has no way to run
twenty different `envs/` packages without them colliding — the reason
`score-zips` already gives each zip [its own
process](advanced/organisers.md#2-run-the-private-board). All three pages now
say so.

## [0.32.0] — 2026-09-03

The Kaggle leaderboard is set up and verified end to end, the standalone
syllabus files are gone, and every runnable thing in the repository is now
listed in one place.

### Fixed — every real submission would have been rejected by Kaggle

`competition.run submit` wrote one row per step **actually flown**, but the
solution file is exported at the full 50-step horizon and `kaggle/metric.py`
rejects a submission missing any `(seed, step)` row. Every real agent ends at
least one episode early — the brick ends nearly all of them in a mid-air inside
a dozen steps — so the refusal would have hit **every** submission, including
the reference ones.

Measured before the fix, against the 100 published seeds:

| agent | rows written | rows required | verdict |
|---|---:|---:|---|
| PPO 4M | 4 789 | 5 000 | refused, 211 missing |
| rule-based | 4 238 | 5 000 | refused, 762 missing |
| noop | 1 162 | 5 000 | refused, 3 838 missing |

`write_submission` now idles every trace out to `max_steps`. The padding is
score-neutral — `replay_actions` already idled through the tail, and
`read_submission` already filled missing cells with the idle action — so no
number changes; the file simply covers what it claims to.

The gap existed because every test built traces that happened to run the full
horizon. `test_score_accepts_traces_from_episodes_that_ended_early` closes it,
using the harness's own `legal_traces`, which stops when the episode does.

### Added — `docs/advanced/kaggle.md`, and the dry run behind it

The competition was set up and scored locally, all the way through the real
metric file. Verified 2026-09-03, score v2.0.0, against the committed
`ppo_4M_curriculum.zip`:

| agent | failed | congestion | `exit_miss` | clearances | touched |
|---|---:|---:|---:|---:|---:|
| PPO 4M | 10 | 176 | 0.1288 | 2 089 | 477 |
| rule-based | 20 | 96 | 0.5700 | 727 | 231 |
| noop | 100 | 752 | 0.0000 | 0 | 0 |

Three things the dry run confirms, each of which would have broken the board:
the metric and the harness agree **exactly** on all four packed components for
all three agents; the metric never pulls torch into `sys.modules`; and handing
the metric one `Usage` split scores only that split's seeds.

The page carries both halves — how a participant builds and uploads a
submission, and the organiser runbook for creating the competition.

### Changed — the metric is a Kaggle *notebook*, and it now meets that contract

The custom metric is pasted into a **Metric Notebook**, not uploaded as a file,
and Kaggle enforces four things the repository cannot see: a function named
`score`, its first three parameters fixed, **type annotations on every
argument**, and a finite non-null `float` return. All four held already;
`test_score_satisfies_kaggles_metric_notebook_contract` now asserts them, so a
refactor cannot break one and have it surface on the leaderboard instead.

`score()`'s docstring is **what a host sees when selecting the metric**, capped
at 8 000 characters. It was one line. It now leads with the ranking and the
direction, documents the submission format, and carries **twelve doctests** — two
worked scores on a two-aircraft head-on sector, and two rejection messages. The
examples print their errors rather than raising, so they pass identically whether
the file is imported as a module or pasted into a cell, where the exception's
qualified name would differ. Verified in both contexts.

Three behaviours were measured rather than assumed:

- **`Pass Complete Submission` does not matter.** On or off, the score is
  identical to the last digit — the metric reads only the seeds present in the
  solution it was handed.
- **Row order does not matter.** Kaggle aligns on the id column; shuffling the
  submission changes nothing, because the metric sorts internally.
- **The 30-minute scoring limit is not close.** 0.04 s for a 60-seed split.

### Verified — on the real competition, not just locally

A sandbox submission of the shipped checkpoint returned
`12012163199400.000` public and `28014722826928.000` private — **exactly** the
values computed locally, on both splits. One upload confirms the four things
that could each have failed silently: the metric notebook runs, the solution
file parses, the `Usage` split is applied, and the horizon padding satisfies
Kaggle's row check.

It also settles a question this repository could not answer: the board renders
the packed key at **full precision**, so submissions differing only in the
lowest-weighted component still show as different numbers. The open caveat about
display rounding is removed rather than softened.

### Changed — three numbers, not two leaderboards

`competition.md` described a public Kaggle board and our private one. With
Kaggle's own `Usage` split there are three scores, and conflating two of them
would have been the confusing kind of wrong. The page now separates them and
says plainly that **Kaggle's private split is not secret**: every seed it scores
ships in `seeds_public.txt`.

It cannot be otherwise, and the reason is structural — a submission is an action
trace keyed by seed, so producing rows for a seed requires knowing it. An
action-trace competition cannot have secret seeds at all, which is exactly why
the trophies are decided by the participant's zip on our machine. Kaggle's split
hides the final rank from *view*; ours hides the seeds from *training*.

The one setting that matters is **Leaderboard Sort Order**, which defaults to
*higher is better* and has to be turned off. It is flagged at the top of the
organiser section and has a screenshot slot of its own.

The template's suggested `del solution[row_id_column_name]` must **not** be
copied: for this metric the id *is* the data — `s0000042_t07` names the seed and
the step, and the replay cannot be reconstructed without both.

### Added — `--holdout-frac` on the solution exporter

Kaggle wants a public/private split of its own, and the exporter could only mark
every row the same way. `--holdout-frac 0.4` marks the back 40% of the exported
seeds `Usage=Private`, deterministically, so the file regenerates identically
without a split being recorded anywhere.

This is **not** the seed list that decides the trophies. `--private` exports
that one, and it never goes near Kaggle; the exporter's docstring now says so at
the top, because the two words meet in one file.

### Added — `docs/reference/tooling.md`

Every runnable thing in the repository, grouped by session and marked with who
is expected to run it. Written because eleven scripts appeared nowhere in the
handbook — the whole Kaggle path, the post-mortem, `make_seeds`, and the entire
slide pipeline among them.

Writing it turned up that most of `solutions/` no longer runs: several files
read `SCENARIO.n` and `SCENARIO.m`, which `ScenarioConfig` has not had since the
`Flight4DEnv` rewrite. The page says so rather than listing them as tooling.

### Removed — `tutorial/TUTORIAL_SYLLABUS_{ADVANCED,INTERMEDIATE}.md`

Both files, and the two handbook pages that embedded them. The Advanced one
opened with a banner admitting it described a lecture structure that no longer
existed, and its body still referenced `um_flight_env.py`, a `MultiDiscrete`
action space, `MaskablePPO`, a five-tuple observation and an Easy/Medium/Hard
evaluation suite — none of which are true of `Flight4DEnv`.

What was worth keeping was rewritten against the current environment as
**[the design sprint](advanced/design-sprint.md)**: the four tracks, the MDP
pitfalls table, the sprint format and the deliverables. Track E (masking) and
Track F (permutation equivariance) are not tracks any more — both ship as
defaults — and the session guide already covered them.

Track D gained a correction the syllabus never had: changed dynamics are a
**training-only** intervention, because the scored run uses our frozen simulator.

The `mkdocs-include-markdown-plugin` dependency went with them; those two pages
were its only users.

### Changed — the Advanced deck's Kaggle walkthrough

`Two leaderboards` told participants to score the public board with
`scripts/score_4d.py` on seeds 900000–900099. The public board has not worked
that way since the harness landed: it replays a `submission.csv` built on
`competition/seeds_public.txt`.

Three appendix frames added — the three-command submission flow, the reference
rows on the board, and what a rejection looks like — with a new `\shot` macro
whose missing-file box names the screenshot to take. Body pages unchanged at 20;
the deck is 74 pages.

### Added — the private board actually exists now

`competition.md` described unzipping a submission, loading the team's
environment class and running the contract suite. **None of it had code.** The
harness could score a bare SB3 checkpoint passed as `--agent path.zip` and
nothing else: no zip handling, no `submission.yaml`, no `env_entry_point`, and
`rollout.make_env` was hardcoded to our environment with the comment *"Always
ours, never the student's"* — which directly contradicted the published rules.

`competition/ingest.py` closes it, plus `competition.run score-zips` as the
batch runner. The design decision that makes it safe to run student code
against a ranking:

**A team's environment supplies observations, never KPIs.** Their env and ours
run in lockstep on the same seed, and only the action index crosses. Their
reward, their termination shaping and their info dict are never read. So a wrong
environment cannot buy a better score — it feeds their policy worse information
and scores worse, honestly earned. There is nothing to defend against and the
code does not pretend to.

Verified three ways on the same seeds, all returning `4 / 38 / 0.1541 / 455 /
98`: the pre-existing `--agent checkpoint.zip` path, a zip using the stock
environment, and a zip shipping its own environment whose reward is multiplied
by 1000 and offset by −7. The last one is the point — a mangled reward passes
the contract suite and changes nothing.

Also fixed along the way:

- **Our env decides when the episode ends.** A student env that terminates early
  no longer shortens the scored episode; the sector is idled to the horizon, so
  giving up stays a penalty rather than a way to stop accumulating congestion.
- **One process per zip.** Every submission ships a directory called `envs`;
  importing two of them into one interpreter would silently give the second team
  the first team's code, and produce a plausible wrong number.
- **The contract suite can be aimed without editing `conftest.py`.**
  `RLB_ENV_ENTRY_POINT=envs.my_env:MyFlightEnv pytest tests/test_env_contract.py`
  is what a team runs and what the harness runs. Getting the path precedence
  right needed both env vars set *before* anything imports `envs` — once a
  package is in `sys.modules`, path order stops mattering.
- **Rejections are actionable.** `-x` stops at the first failure and the harness
  reports the assertion that fired, not a count. "24 failed" tells a team
  nothing at a prizegiving.

### Changed — the submission format matches the harness that reads it

`inference.py` is gone from the spec: the manifest says everything needed, so
there was no loader for a team to write and nothing read the file.
`vecnormalize.pkl` now produces a visible warning instead of being silently
ignored — `Flight4DEnv` emits observations already scaled, and the harness does
not wrap the scored env.

The trap worth naming: a submitted `envs/` **replaces** ours on the import path,
so a file in it doing `from envs.flight_4d import Flight4DEnv` looks inside the
team's own package and fails. Copy the whole directory and use a relative
import. This is the most likely way for a submission to be rejected and it is
entirely avoidable, so it is called out in the rules with the command to check
it.

### Added — `docs/advanced/organisers.md`, the runbook

Every competition command in the order an organiser runs it, across the three
phases: seeds and Kaggle files before the session, the two things worth having
ready during it, and the private board, post-mortem and video after it closes.

Then a complete flag reference — `competition.run`'s global flags and all five
subcommands, `make_seeds`, `kaggle.export`, `ingest`, the four environment
variables, and which files are secret. Every flag in it was checked against
`--help` rather than transcribed from memory, which turned up `make_seeds
--count` and `replay --strict` as previously undocumented.

The split with the Kaggle page is deliberate: this one is *what to type*, that
one is *what to click*.

Verified by running the documented Phase 3 end to end against a throwaway salt —
`make_seeds --which private`, then `score-zips --private` over two real zips on
100 genuinely unseen seeds. Both scored identically, which is the
reward-independence property holding on a second seed set.

### Fixed — a missing private seed list blamed the teams

`score-zips` resolved the seed list inside each subprocess, so a host who had
not run `make_seeds` got the same `FileNotFoundError` repeated once per
submission, listed under a heading about the contract gate. It now resolves the
list once, before spawning anything. The rejection heading no longer claims
every rejection is a contract failure either — a bad zip or a missing manifest
is not.

### Added — `docs/planning/simplification_log.md`

Six places where the code is heavier than the job needs, or carries something
that no longer runs. Internal, excluded from the site.

### Added — `competition.run decode`

The packed score explains itself to nobody: `1.20 × 10¹³` carries no obvious
relationship to "10 failed episodes." `kaggle.metric.unpack` inverts `pack`
exactly — it is a mixed-radix encoding, so decoding is `divmod`, not a fit —
and `python -m competition.run decode <score>` puts that on the command line:

```json
{
  "failed": 10,
  "congestion": 176,
  "exit_miss_bucket": 26,
  "clearances": 2089
}
```

`exit_miss_bucket` is lossy one way — `pack` already rounded it to the nearest
`EXIT_MISS_BUCKET` before packing, so `unpack` can return the bucket and an
approximate mean but not the exact value that went in. A component sitting
exactly on its `BOUND_*` constant may be clamped rather than measured; the
docstring says so rather than let a saturated value read as a real count.

Kaggle's leaderboard has no second column to put a friendlier number in — this
is the closest a Metric Notebook competition gets, and it costs nothing on the
scoring path: `pack` and `score` are untouched, `unpack` is a new pure function
next to `pack`. Its doctest is collected by its own test, mirroring how
`score`'s is, and `test_pack_unpack_round_trip` checks the inverse holds across
200 random draws over the full bound space, not just the one worked example.

## [0.31.5] — 2026-09-03

The handbook is reorganised by **difficulty rather than by topic**. Three
session tabs, each self-contained and colour-coded, with Home and Setup in front
and Reference behind.

### Changed — one tab per session, each a self-contained container

Thirteen pages moved. The docs were organised by *kind* — Environments /
Running / Sessions — and each kind held exactly one page per session, so
re-parenting was a **transpose**: whole files moved, no page was split.

| Was | Now |
|---|---|
| `sessions/fundamentals.md`, `environments/maze.md` | `fundamentals/` |
| `sessions/intermediate.md`, `environments/crippled-ant.md`, `running/crippled-ant.md`, `sessions/syllabus-intermediate.md` | `intermediate/` |
| `sessions/advanced.md`, `environments/airtraffic.md`, `running/airtraffic-{training,evaluation}.md`, `running/baseline.md`, `sessions/competition.md`, `sessions/syllabus-advanced.md` | `advanced/` |

`reference/` is unchanged and `changelog.md` joins it. The Advanced tab carries
seven pages, so it is grouped in the nav rather than listed flat.

Every internal link was rewritten by resolving each target against its *old*
directory, mapping through the move table, and recomputing the relative path
from the new one. All moves preserved directory depth, so `../assets/…` links
were unaffected.

`changelog.md` needed a distinction: its **backticked prose paths are historical
record** and are left alone, but its **markdown links must resolve** or the
published page is broken. Ten links were repointed; the prose was not.

### Added — colour-coded tabs

`docs/overrides/main.html` sets the tab colour from `page.url` — green for
Fundamentals, amber for Intermediate, red for Advanced, and the site's blue for
Home, Setup and Reference. Pure Jinja, no JavaScript, and it survives the
light/dark toggle because it redefines the same CSS variables
`docs/stylesheets/extra.css` already drives everything from.

**Contrast was measured, not assumed.** Every value clears 4.5 against its own
surface in both schemes. The brand amber `#e09f3e` scores **2.22** on a white
page and is unusable as link text; the light-mode amber here is `#9a6700`
(4.87).

The colour is applied in two files — the tab in `main.html`, the site-map pill
in `extra.css` — and they must stay in sync. Both say so.

### Changed — the home page is now a site map

The four old cards were organised by kind and two of them pointed into
`environments/` and `running/`, which no longer exist as tabs. In their place: a
"do this first" pair, then **three colour-coded session cards** whose colour is
the difficulty and matches the tab you land in, then a new **"Understanding the
codebase"** section mapping repository paths to tabs.

### Fixed — two bugs that only rendering caught

`{ .rlb-f }` attr_list on a raw `<span>` does not apply. It rendered as
**literal visible text** on the page, and the difficulty pills had no background
at all — white text on transparent. Separately, Material renders `grid cards` as
`div > ul > li`, so the original `.rlb-map .rlb-f` selectors matched nothing.

The fix puts the colour class on the pill itself, so the **difficulty colour
cannot be broken by reordering the cards**; only the decorative left border
stays positional, and the CSS says so.

## [0.31.4] — 2026-09-03

Three decks instead of two, and the Advanced deck stops being a four-hour
lecture. Slides become a tour with a pointer to the handbook; the room gets its
time back.

### Added — `slides/litmus.tex`

Module 6 of the Intermediate deck becomes a standalone auxiliary deck that runs
alongside Session #3. It was a Session #3 argument wearing a Session #2 timeslot:
every number on it is measured on `Flight4DEnv`, the environment Session #3
designs, and Session #2 had no time to do it justice.

Session #2 therefore **ends 30 minutes earlier** — the timetable now runs
0:00–2:30 and closes on "Wrap-up, and what Session #3 asks of you". Its body
drops **38 → 31 pages**. Learning objective #4 changes from "apply the litmus
test" to "say what you would need to measure before claiming a learned policy
beat anything", which is the part Session #2 can actually deliver. The
AirTraffic command block in the appendix cheat sheet shrinks to the three
commands that still make sense before you have met the environment.

### Changed — `advanced.tex` re-scoped from four hours to thirty minutes

Body **46 → 20 pages** (14 frames, 5 sections). It is now a tour of the baseline
solution: the contract, the sector, the four constants, the demonstration clips,
the observation and action spaces, termination vs. truncation, the reward, and
reading the training curves. Every frame points at the handbook rather than
restating it.

Five frames are new. The **observation** and **action** spaces are TikZ diagrams
— the flat 103-vector split into five aircraft blocks, two globals and the 36
inline mask bits, with the thirteen per-aircraft features grouped by what they
are *for*; and the action space as an aircraft × clearance grid showing
`1 + 7*aircraft + clearance` with idle as index 0. The reward frames are built
on the figures added in 0.31.3.

**Nothing was lost.** 29 frames moved into seven labelled appendix sections
(`Extra --- eleven mistakes`, `Extra --- the competition`, and so on), so a
presenter can jump to any of them during questions or the sprint. The appendix is
now 43 frames, which is large — but an appendix is not presented linearly, and
labelled sections make it navigable.

Three frames were **deleted** rather than moved, per the standing rule that
duplicates and stale mechanics go: the old opener and closing punchline (both
replaced), and the **4-hour timetable**, which describes a schedule that no
longer exists and would be misinformation if kept.

### Fixed — five new frames were overfull, and two diagrams were clipped

Authored at 33–57 pt overfull. Trimmed by **removing boxes, not shrinking type**
— merging paired callouts into one, demoting a block to a grey footnote, cutting
the TensorBoard list from four items to three. Both decks' bodies are now clean
at the 18 pt threshold.

The observation diagram initially ran its feature list off the right edge, losing
`severity`, `t_conflict` and `active`; the fix wraps it into four role-grouped
lines in a box that sizes itself to its content. The action diagram had the idle
cell colliding with the row labels and its caption clipped at the slide edge.

### Note on measuring overfull boxes

The `Overfull \vbox` list is only meaningful when a deck is built **inside
`slides/`**. Without `figures/` and `media/` present, `\fig` and `\clipbox` fall
back to placeholder boxes of a different height and the list changes — which
briefly made a pre-existing box look like a regression. `intermediate` carries
three long-standing boxes over threshold (46.1, 29.4, 28.3 pt) and `advanced` one
(49.5 pt, in an appendix frame); the README now records this instead of claiming
the decks are clean.

### Follow-up created

`tutorial/TUTORIAL_SYLLABUS_ADVANCED.md` still describes the old lecture-heavy
Modules 1–7 on 4-hour timings. It now carries a prominent note saying the deck
was re-scoped, but it needs rewriting to match — a content decision, not a
mechanical edit.

## [0.31.3] — 2026-09-03

Plots and renders, so the reference pages stop asking anyone to picture a
trade-off from a table of numbers. Four figures, four demonstration clips, and a
palette validator, because "is this colourblind-safe?" is a question with a
computable answer.

### Added — the two-aircraft demonstration clips

`scripts/render_4d.py` gains a scripted mode: `--agent split|macro|speed` replays
a fixed `(step, aircraft, clearance)` list, plus `--n-range` to pin the traffic
draw and `--ext gif`. All three run on seed 2 at `n_range=(2,2)`, where flights 0
and 1 sit co-level at FL374 and close to `d_cpa` = 1.0 m at step 18.

The three resolutions **trade against each other**, which is worth more than
three clean wins:

| demo | outcome | congestion | clearances | `exit_miss` |
|---|---|---:|---:|---:|
| do-nothing | dies at step 18 | 3 | 0 | 0.000 |
| split the levels | survives | 0 | 4 | 0.000 |
| one macro | survives | 2 | 1 | 0.000 |
| offset the speeds | survives | 5 | 2 | 0.035 |

The scripted controller is deliberately open-loop: these clips exist to show
what *one* clearance does to the geometry, and a reactive controller would make
it impossible to say which decision produced which part of the outcome. The
fixed step numbers are an approximation and the docs say so — one aircraft may
be commanded per step and a temporary manoeuvre's return leg runs on a timer the
caller does not control, so timing a resolution is part of the problem.

`docs/environments/airtraffic.md` gains the clips and, before them, a table
explaining **how to read a render** — tracks, projections, congestion rings, the
hotspot square, the exit-deviation tick, the two KPI bars and the clearance
markers.

### Added — four reference figures

`slides/scripts/make_figures_4d.py` writes PDF to `slides/figures/` and PNG to
`docs/assets/figures/` in one call, so a figure cannot be current in one place
and stale in the other: the reward decomposition, the design ratios, the
severity cliff, and the `RECOVERY_SEP_FACTOR` trade-off curve.

The sweep is drawn as a **trade-off curve, not a dual-axis chart**. Both
quantities are costs, so they go on two axes of one plane and the frontier is
readable directly — down-left is better on both, and 7.0 is visibly the last
point inside the random-legal reference lines on both columns.

### Changed — two figures that did not survive contact with the data

The severity **surface** became a **line with an annotated cliff**. A heatmap
renders the exponential fade but hides what matters: `predict_conflicts` omits a
pair at or beyond `ALT_MIN_SEP`, so severity is *deleted*, not decayed. A
discontinuity needs a form that shows it.

The mid-air-charge figure was **dropped**. It plotted the charge against
`calibrate_4d.py`'s "worst cost of flying on" and appeared to show that a late
crash is cheap — a category error, since `worst_remaining` is a maximum over all
states and cannot be drawn as a horizontal line against a per-step curve. The
shipped check is two scalars, and two scalars are a sentence, not a chart. The
reasoning sits in the script so nobody re-adds it.

### Added — `slides/scripts/validate_palette.py`

A port of the six accessibility checks (lightness band, chroma floor, CVD
separation, tritan, normal-vision floor, contrast) using the same OKLab metric
and Machado transforms as the JS original, because this machine has no JS
runtime and the check must be computed rather than eyeballed.

It immediately earned itself: the repo's existing amber `#e09f3e` **fails
contrast** against a light surface — 2.22 against a floor of 3.0. The new
three-series figures use `#1f6feb / #c2801f / #a3195b`, which passes all six.
Darkening the amber instead was tried and rejected: it collapses CVD separation
against the existing green, the classic red-green convergence.

**Not fixed here:** `#e09f3e` is still used by `make_figures.py` and
`make_figures_mistakes.py`, which carry the same contrast failure. Repainting
the existing figures is a wider change and is filed, not done.

### Changed — the macro section of the controller reference

The argument was sound and the setup was missing. It now states what the macro
mechanically *is* (`_T` = auto-reverting on a `hold_steps` timer), why it is
tempting (one clearance against three, because `RESUME` is charged and the
auto-revert is not), why two levels is the minimum that resolves anything
(`DALT` 12 against `ALT_MIN_SEP` 24), the failure rate as a rate (50% against
random's 34%, rather than 20-in-40 against 34-in-100), what "just-in-time lead"
means, and why 8 aircraft were never commanded at all — the one-clearance-
per-step budget.

`flight_4d.py` calls the macro "the minimum effective vertical intervention"
while the controller page calls it worse than random. Both are true and they
describe different halves of one instruction; the environment now says so and
points at the controller, so the two files stop appearing to disagree.

### Added — a drift guard

`make_figures_4d.py` mirrors seven environment constants rather than importing
the airtraffic package, so it runs from the repo root against committed CSVs.
`test_figure_constants_match_env` asserts the copies still agree, and skips if
`slides/` is absent from a filtered export.

## [0.31.2] — 2026-09-03

The reward reference stops describing an environment that no longer exists.
`docs/reference/reward-design.md` was written against the retired
first-generation env at `n=5, m=5` — ten aircraft, a signature `Flight4DEnv`
does not have — and carried a survival bonus that was removed in 0.19.0. It is
now measured on `Flight4DEnv` at its own defaults, `n+m=5` with `n ~ U(2,4)`.

### Changed — `reward-design.md` rewritten and re-measured

Every number is new, from 40 seeds at `gamma=0.99`. The headline result
**inverts**: on the retired environment do-nothing scored *best* (+59.68),
because the survival bonus paid it 100 points it could not lose. On
`Flight4DEnv` do-nothing is the worst policy on the board by nearly a factor of
three (127.51 against the controller's 47.25), while remaining perfectly on time
and perfectly efficient — which is a far better teaching row, because it shows
two of the three objectives can be maxed by doing nothing at all.

Also now documented, and previously absent: the severity function's vertical
credit falling out of the physics rather than being chosen (one level removes
39%, the second the remaining 61%); the design ratios as the actual design
statement; and the mid-air charge's margin over the worst cost of flying on
(450 against 170.55), which is what stops a cost-only reward from paying the
agent to crash.

### Changed — potential-based shaping is now filed as an extension, not a description

It was described as what the reward *is*. It is Track C of the design sprint —
something a student may add. The section now says so, keeps the measured
shaping-scale sweep and the step-function trap (both `DALT`=12 and
`ALT_MIN_SEP`=24 are unchanged, so that trap is still live), and states the
trade the environment accepted by not using it.

### Added — the "detour versus endpoint" fix, which was never written down

The first-generation environment charged the timeliness deviation *right now*,
every step, so the reward integrated the whole detour: 9.963 penalised against
0.039 scored over 100 seeds. The reward said *never leave your plan* while the
ranking said *leave it freely, but come back*.

`Flight4DEnv` does not fix this by moving the charge to the horizon, which is
what the old page implied. It changes the quantity — `_exit_deviation()` returns
where the aircraft *will* cross if it holds its current speed. A detour that is
later recovered projects back to target and costs nothing; only an unrecoverable
deviation stays on the bill. The two regimes (projection before the boundary,
recorded fact after) are documented with the phantom-delay bug that conflating
them caused.

### Added — `calibrate_4d.py --agent` and `--csv`

`measure()` took no agent and hard-coded a random-legal rollout. It now accepts
`"noop"`, `"random"`, `"rule-based"` or a checkpoint path — the same dispatch
`score_4d.py` uses, so both read the same strings — and can write the per-agent
decomposition to CSV. The terms are still read off `env`'s own
`_severity`/`_deviation` rather than re-derived, so a decomposition cannot drift
from the reward it claims to decompose. Output at `slides/data/air_reward.csv`,
which Phase 2's reward plots will consume.

### Not changed, on inspection

Three flagged survival-bonus mentions turned out to be correctly framed already:
`slides/advanced.tex:630` presents it as removed mistake #1, `:656` explicitly
states its absence in `Flight4DEnv`, and `TUTORIAL_SYLLABUS_ADVANCED.md:344` is
a historical note about a past training run. All three describe it as history or
as absent, not as current behaviour, so none were touched.

## [0.31.1] — 2026-09-03

A documentation-review pass, ahead of a larger docs and slides refactor. Two of
the three published Session #2 policies were not actually in the repository, and
`Flight4DEnv` contradicted itself about whether its reward is potential-based.
No behaviour changes.

### Fixed — two Session #2 checkpoints were silently excluded from git

`.gitignore`'s un-ignore exception for the Crippled Ant checkpoints was
`checkpoints/*.zip`, one directory level too shallow to match the actual layout,
`checkpoints/<run>/final_model.zip`. `ppo_specialist_3M` and
`ppo_randomised_range_3M` were therefore never committed, while their
`.hydra/config.yaml` and `vecnormalize.pkl` *were* — so a fresh clone got two
populated-looking checkpoint directories with no policy inside them.
`ppo_healthy_3M` escaped only because it had been force-added.

The exception now uses `**`, and both missing policies are committed. The
airtraffic checkpoint was unaffected: it sits directly in `checkpoints/`, where
the single-level pattern matched.

### Fixed — the reward was documented as potential-based shaping, and is not

`Flight4DEnv.step()` computes three direct per-step costs plus a terminal crash
charge. There is no `gamma*Phi(s') - Phi(s)` term anywhere in the environment,
and the comment at the reward itself says so explicitly, calling the direct-cost
form *"a deliberate choice"*.

Four other places in the same module still described the retired shaping design
— the module docstring's design point 3, two comments on the reward weights
referring to `Phi_safe` / `Phi_4D` and a `gamma^T` discount that no longer
applies, and a `# potentials` section header over what are plain cost functions.
`docs/environments/airtraffic.md` and `scripts/calibrate_4d.py` had inherited the
same claim.

This mattered beyond tidiness: potential-based shaping *provably* cannot move
the optimal policy, and the docs were extending that guarantee to weights which
genuinely can. Anyone retuning `w_safe` or `w_clearance` on the strength of it
was working from a false premise. All six now describe the direct-cost design
and state the trade it accepts. `docs/reference/environment-api.md` already
described this correctly and was used as the reference.

### Fixed — `hold_steps` docstring disagreed with the signature

The module docstring said `hold_steps` "defaults to 8 (16 s)"; the constructor
has defaulted to `5` (10 s). Only the prose was wrong.

### Fixed — version strings that had drifted

`mkdocs.yml`'s `DOCS_VERSION` fallback still read `0.17.0` despite claiming to
mirror the repo-root `VERSION`, and the home page advertised `0.6.0`. The
fallback now mirrors, and the home page points at this changelog so it cannot
drift again.

The two decks' `\institute` lines still read `VERSION 0.29.0`. Those are
**provenance stamps** — the release the quoted numbers were measured at — not
mirrors of the current version, so they are deliberately left alone until the
board is re-measured.

### Added — internal planning documents

`docs/planning/tutorial_refactor_plan.md` (the compiled review, in six
dependency-ordered phases) and `docs/planning/volunteer_checklist.md` (five
standalone verification tasks). Both live under `docs/planning/`, which
`exclude_docs` keeps out of the student handbook.

## [0.31.0] — 2026-08-28

Domain randomisation gets a second, working attempt; the specialist gets a
fair budget; the rule-based controller's recovery order stops being an
accident of flight index; and `Flight4DEnv` gets the render path it did not
have — congestion visibility carried forward from the retired environment,
plus the flight plan and two live KPI bars that renderer never had.

### Fixed — `Priority4DController`'s `RESUME` order was arbitrary

`_recover()` picked among simultaneously-eligible aircraft by ascending flight
index (`np.flatnonzero`'s own order), which carries no information about the
problem. It now resumes in **assignment order** — the same earliest-conflict-
first ordering `_assign` already computed and stored — so the aircraft the
controller decided mattered first also comes home first. The board and
competition numbers are unchanged (`14 failed / 90 congestion / 667
clearances` on `score_4d.py`'s seeds; `20 / 96 / 727` on the competition's) —
simultaneous eligibility is rare enough on these seeds not to move the
aggregate, but the design was wrong regardless of whether it showed up.
Pinned by a new synthetic test, since the natural seeds don't reliably exhibit
the tie.

### Added — `scripts/render_4d.py`, and the render path it needed

`Flight4DEnv` still declares no render modes by design — hard masking and a
flat action space were the problem, not visualisation, and the environment
stays free of matplotlib. The new script drives it from the outside: step it,
record what happened, replay through `CollisionCourseSimulator`'s shared
geometry. The congestion styling (dashed-while-separated, solid-once-not,
severity-weighted, a fading trail) is carried over verbatim from the retired
environment's own renderer, which solved the same visibility problem once
already — an almost-invisible `alpha=0.1` fixed circle, replaced.

New on top of that inheritance, none of it in the retired renderer:

- **The flight plan itself.** A solid track behind each aircraft, one colour
  per segment for the level actually flown, and a dashed projection ahead —
  one colour for the level held now — running exactly as far as the aircraft
  has left to fly. Headings are never commanded, so that projection is not a
  guess; it is where the aircraft goes if nothing else touches it.
- **Two KPI bars**, safety and timeliness, on a green-red scale (low is good),
  reading straight off `_severity()`/`_deviation()` — the same methods the
  reward uses, not a re-derivation from positions that could quietly drift
  from what the policy is actually scored on.
- **A per-aircraft deviation tick**, the same green-red scale, above each
  flight's altitude/speed label.
- Smaller clearance markers, larger heading arrows, a square plot (the data
  aspect is pinned; it no longer stretches to fill whatever box the KPI bars
  and colourbar leave behind), and the legend outside the axes.

`sessions/03-advanced/airtraffic/renders/seed900000_n3m2_*.mp4` — noop,
random, rule-based and the reference PPO checkpoint, all the same scenario —
feed the posters and filmstrips both decks now embed via `make_media.py`; the
old clips were the *retired* environment's, quietly showing the wrong sector
for a deck that had already moved on.

### Fixed — Session #2's specialist and domain-randomisation checkpoints

**The specialist now gets the same 3M-step budget as the generalist it is
compared against.** The committed 1M-step specialist tied the transferred
zero-shot policy on its own injury (1112 vs 1119) at a third of the training;
give it the full budget and it *dominates*: **3978 vs 1119, a 3.6× margin** —
and healthy collapses further, not less, **93.7** against the 1M version's
335. More training on one failure mode sharpened the dependency on that
failure existing; it did not buy generalisation.

**Domain randomisation over severity was tried at `n_random_legs_max=4`
first, and it collapsed.** With four legs disabled there is barely anything
left to actuate, so near-stillness is close to optimal — and the policy
generalised that strategy to *every* severity, scoring **worse than
do-nothing even healthy** (`-150`, every band flat and near zero). The
training curve did not show this: `ep_rew_mean` sat around 300 with `ep_len`
500-600, reading as mediocre-but-functional. Only scoring against fixed
scenarios — not the mixed training distribution — exposed it. Narrowed to
`n_random_legs_max=1` (severity uniform on `{0, 1}`, keeping 0 in the training
distribution — the actual design fix `n_random_legs=1` never had), it trained
into a real result: **wins 2-legs-dead outright** (973 vs 919 generalist, 722
specialist) and lands within 2% of the specialist's own 1-leg aggregate (1220
vs 1247), for a healthy floor of 787 — below the do-nothing line, and the
honest price of the robustness bought above it.

Both checkpoints are published at `sessions/02-intermediate/crippled-ant/
checkpoints/{ppo_specialist_3M,ppo_randomised_range_3M}/`, laid out as run
directories so `transfer_benchmark.py` and `render_agent.py` take them
directly. `slides/data/ant_transfer.csv` and `fig_specialist`/`fig_randomised`
are regenerated from the real 20-episode benchmark; nothing here is
transcribed by hand into a slide that could drift from it.

## [0.30.0] — 2026-08-28

The pass that retires the first-generation environment. One environment, one
action space, one policy, one board — and every number on a slide traceable to a
committed CSV.

### Removed — the first-generation environment leaves the session tree

`LegacyFlightEnv` and everything built only for it moved out of
`sessions/03-advanced/airtraffic/` and into `solutions/advanced/`, which is now
explicitly **legacy example code** rather than a second live curriculum:

- `envs/um_flight_env.py` -> `solutions/advanced/legacy_env.py`
- `policies/order_invariant.py` -> `solutions/advanced/legacy_policy.py`
- `agents/rule_based.py` -> `solutions/advanced/legacy_agents.py`

It all still imports and runs — code that cannot run is not example code — but
the participant-facing package now exports exactly `Flight`,
`CollisionCourseSimulator` and `Flight4DEnv`, and nothing else.

Deleted outright, because they only ever drove the retired environment:
`scripts/{train,evaluate,evaluate_baseline,render}.py`, the root
`um_flight_env.py` duplicate (which imported a package that is not in this repo
and had been dead for some time), `{train,eval}_umflightenv_sb3.py`, all three
`notebooks/*.ipynb`, `conf/default.yaml`, `plotting.py`, `utils/viz.py`,
`tests/{test_baseline,test_macro_action,test_policy}.py`, and
`slides/scripts/{extract_advanced,make_figures_advanced}.py` with the six CSVs
and figures they produced.

**Authoring `Flight4DEnv` notebooks is deliberately deferred**, not forgotten.

### Changed — the trainer gains the settings the exercise is *about*

Session #3's exercise is choosing settings, not writing them, so
`scripts/train_4d.py` now offers the two that were worth porting:

- `--curriculum` / `--curriculum-frac` / `--curriculum-n-start` — narrows the
  per-episode draw of `n` and widens it to the scored density. `n + m` is fixed
  either way, so the observation and action spaces never move. Needs the new
  `Flight4DEnv.set_n_range(lo, hi)`.
- `--auto-entropy` / `--auto-entropy-target` — tunes `ent_coef` so the policy's
  entropy tracks a fraction of `ln(n_actions)`. A *fixed* coefficient has failed
  in both directions here; the failures are about the entropy, not the knob.

Action-set choice, observation frame stacking and LR-schedule choice were
considered and **not** ported: the first two contradict the environment's own
design (auto-reverting clearances replace RESUME; the state is Markov on one
frame), and the third only ever offered a known-worse option.

`--checkpoint-every` (default 500k) closes a real gap: `train_4d.py` saved only
at the very end, so an interrupted 80-minute run lost everything.

New `scripts/callbacks_4d.py` houses both callbacks.

### Changed — the competition harness scores the environment the session teaches

`competition/` was still building the retired environment; the deck said so out
loud. It now runs `Flight4DEnv` end to end. The action no longer factorises
across flights, so the submission format changes with it:

- **One row per `(seed, step)`** carrying the flat `Discrete` index, replacing
  the per-`(seed, step, flight)` grid built for `MultiDiscrete`.
- `SCORE_VERSION` **1.0.0 -> 2.0.0**, per that file's own convention.
- The scored 4D quantity is `exit_miss` throughout, not `timeliness`.
- The action-set/action-mode dispatch is **deleted** rather than ported — there
  is one action space now, so there is nothing to dispatch on.

`kaggle/metric.py` re-derives the new physics standalone in numpy — the metric
sandbox has no gymnasium by design — including the `hold_steps` auto-revert, the
two-level macro and the 4D gate. `test_kaggle_metric.py`'s replay-parity
property is the oracle that keeps that honest, and it passes.

`tests/test_env_contract.py` was 26 tests of retired-environment internals;
it is now a genuinely generic contract, which is what its own docstring always
claimed it was and what a student's subclass has to pass.

### Added — the reference agent, and the checkpoints that were only ever promised

Retrained on `Flight4DEnv`: 4M steps, curriculum + auto-entropy, ~80 min.
On 100 held-out seeds, against the rule-based bar:

| agent | failed | cong./100 steps | conflict-free | exit_miss | clearances |
|---|---:|---:|---:|---:|---:|
| do-nothing | 96 | 44.04 | 4% | 0.000 | 0 |
| random-legal | 34 | 15.05 | 9% | 0.754 | 2924 |
| rule-based | 14 | 2.01 | 63% | 0.554 | 667 |
| **PPO 4M `[det]`** | **1** | 3.46 | 49% | **0.087** | 2278 |
| PPO 4M `[sto]` | 2 | 4.58 | 48% | 0.143 | 3338 |

The learner wins the two ranked-first objectives and loses efficiency. That is
the first time in this bootcamp a learned policy has cleared the heuristic.

**One hundred seeds does not pin a failure rate.** The same checkpoint fails
roughly 3.5% of a further 600 held-out seeds. The board row reproduces; the
*rate* needs more seeds than a board has, and the deck now says so on the slide.

`*.zip` stays gitignored, with three narrow exceptions, because participant
material promised checkpoints that no fresh clone could produce:
`crippled-ant/checkpoints/ppo_healthy_3M/`, `ppo_randomised_3M/` (both laid out
as run directories, so `transfer_benchmark.py --run ... --model final` takes
them directly) and `airtraffic/checkpoints/ppo_4M_curriculum.zip`.

### Changed — decks

Session #2 rebalances away from healthy-to-cripple transfer and toward domain
randomisation: `fig_leg_asymmetry` and `fig_specialist` move to `Extra slides`,
and a measured three-way comparison arrives in the body — zero-shot 879-1747
across the four one-leg injuries (2.0x), randomised 858-1161 (1.35x), specialist
660 mean. Randomising flattened the spread **by giving up its best case**, and
`n_random_legs=1` never shows the policy a healthy Ant, which is a design bug
worth showing rather than a price worth paying. The `VecNormalize` frame moves
to `Extra slides`. Body 38 -> **37** pages.

Session #3 gains a frame documenting the settings menu, since "choose the right
settings" is the exercise. Body 44 -> **45** pages, both decks under the 50-page
cap.

### Fixed

- `fig_randomised` rendered two series under a title promising three: the
  randomised run was never registered in `extract_data.py`'s `ANT_RUNS`, and
  `fig_randomised`'s own `len(present) < 2` guard cannot catch a *missing third*.
- `fig_4d_board` pinned the trained row by a hardcoded label, so it vanished
  silently whenever a new agent was scored. It now finds the row by scoring mode.
- The deck offered pre-trained checkpoints at a path that did not exist, with a
  healthy return that matched no measurement.
- `docs/{environments/airtraffic,sessions/advanced}.md` carried "not migrated
  yet" banners. They are migrated.

## [0.29.0] — 2026-08-27

### `env.n_random_legs=1` never ran — Exercise 5 was a dead slide

`CrippledAnt.__init__` gated the mutual-exclusion check on
`disabled_joints is not None or disabled_legs is not None`, and `conf/config.yaml`
defaults both to `[]`. An empty list is not `None`, so **every** invocation of
the documented Exercise 5 command:

```
python scripts/train.py env.n_random_legs=1
ValueError: n_random_legs is mutually exclusive with disabled_joints/disabled_legs
```

`make_ant()` twenty lines below used truthiness (`if disabled_joints or
disabled_legs or n_random_legs is not None`), so the two disagreed about what
"no injury specified" means. Fixed to `bool(disabled_joints) or
bool(disabled_legs)`; `tests/test_crippled_ant.py::test_random_and_static_conflict`
still passes because it passes a *non-empty* `disabled_legs=[0]`. 31 tests green.

This is why Session #2's Module 4 had no ending: Exercise 5 shipped as an
instruction with no result, no data and no figure, because the instruction could
not be run.

### Session #3's deck moves onto `Flight4DEnv`

The two decks described **different environments**. Session #2's Module 6
previewed `Flight4DEnv` (`Discrete(1 + 7n)`, mask carried in the observation,
`score_4d.py`, brick fails 96/100). Session #3's deck described `LegacyFlightEnv`
end to end (`MultiDiscrete([5] * n)`, `competition.run`, brick fails 83/100).
Same audience, consecutive days.

`Flight4DEnv` is the baseline as of 0.27.0 and got its rule-based controller in
0.28.0, so the advanced deck follows it:

| Module | was | now |
|---|---|---|
| 1 — case study | "`LegacyFlightEnv` as it stands" | "the design you are **replacing**" |
| 3 — the bar | `competition.run compare`, UMFlight numbers | `score_4d.py`, the 4D board |
| 4 — eleven mistakes | 13 frames of war stories | 8 grouped frames, each ending in the line it put in `flight_4d.py` |
| 5 — sprint | `train.py --noop-bias 3.5` | `train_4d.py --steps 200000` |
| 6 — leaderboard | UMFlight table at `VERSION 0.17.0` | the measured 4D board |

Module 4's arc is the substantive change: the eleven mistakes are no longer a
war-story reel, they are **the derivation of `Flight4DEnv`**. Critique the old
design (Module 1) → see the eleven mistakes in it (Module 4) → here is what
replaced them → beat it (Modules 5–6).

**Known gaps, both flagged in place rather than papered over:**

- `competition/rollout.py` still builds `LegacyFlightEnv`, so the submission and
  replay harness has not moved. The cheat-sheet frame now says so explicitly
  rather than implying `competition.run compare` scores what you are graded on.
- `docs/environments/airtraffic.md` and `docs/sessions/advanced.md` still
  document `LegacyFlightEnv`. Both carry a warning admonition pointing at
  `flight_4d.py`, `train_4d.py` and `score_4d.py`. Migrating the handbook is the
  next piece of work after the harness.

### The rule-based controller is on the board, and it is not the loser

`scripts/extract_4d_board.py` scored only `noop`, `random` and a checkpoint.
Added `Priority4DController`, so `data/air_4d_board.csv` now carries the row the
whole "do you need RL?" argument turns on:

| agent | failed | ep_len | congestion | /100 steps | conflict-free | exit_miss | clearances |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO masked 1.5M `[det]` | **1** | 49.1 | 194 | 3.95 | 53% | **0.072** | 2421 |
| PPO masked 1.5M `[sto]` | 2 | 49.0 | 193 | 3.94 | 48% | 0.085 | 3410 |
| rule-based | 14 | 44.8 | **90** | **2.01** | **63%** | 0.554 | **667** |
| random-legal | 34 | 35.1 | 528 | 15.05 | 9% | 0.754 | 2924 |
| noop | 96 | 12.5 | 551 | 44.04 | 4% | **0.000** | **0** |

100 held-out seeds (900000–900099). The learner wins the rank key on `failed`
alone. But the controller is **twice as safe per step flown** (2.01 against
3.95) on comparable episode lengths — 44.8 to 49.1, so this is *not* the
dying-early confound the deck warns about elsewhere — and it does it with a
quarter of the clearances.

Two slides were stale against this. Session #2's Module 6 said "nobody has
hand-coded a controller for this environment yet, so *RL beats a heuristic here*
is untested"; it now quotes the comparison. Session #2 also mis-stated the
observation as `Box(13 * n_flights + 2)`; the mask rides **inside** the
observation, so it is `Box(-1, 1, (103,))` at *n*=5 — `13n` features + 2 globals
+ 36 legality bits.

### The decks are hand-written Beamer again

`scripts/rebuild_intermediate.py` and `scripts/rebuild_advanced.py` are deleted,
with `intermediate.tex.bak` and `advanced.tex.bak`. Introduced in 0.26.0 to hold
the pre-cut wording while the bodies were trimmed to 50 pages, the generator
doubled the edit surface for every change and silently destroyed hand edits to
the `.tex` — its own README said so. `intermediate.tex` and `advanced.tex` are
now the source. `git log -p` has the pre-cut text.

### Both decks are shorter, and the hyperparameter module is a mention

| Deck | body pages | PDF | was |
|---|---:|---:|---|
| `intermediate` | 38 | 54 | 48 / 58 |
| `advanced` | 44 | 59 | 50 / 87 |

Session #2's Module 5 went from three frames to one. The shipped grid is 6 cells
× 300k steps, and at 300k **every cell is still in the statue regime** — it ranks
how fast policies stand up, not how well they walk, on one seed per cell against
a seed-to-seed swing of 500. The frame that survives is the one about
`.hydra/config.yaml` being what stops evaluation disagreeing with training; the
grid-search walkthrough moved to the appendix and the timetable gave the ten
minutes to Module 4.

Appendix frames that duplicated a body frame, or that described `LegacyFlightEnv`
mechanics the sessions no longer use (`--noop-bias`, the `n`=5 `m`=5 frozen
scenario, the UMFlight renders), were deleted rather than kept. Both decks now
build with **zero overfull `\vbox` over 18 pt** — including four frames that
were already clipped before this change.

## [0.28.0] — 2026-08-27

### Session #2 slides — Module 6 rebuilt on `Flight4DEnv`, and the statue claim corrected

**The statue number was right; the slide that quoted it was wrong.** A perfect
statue banks **1000.6 ± 4.0** (zero torque, 20 episodes, 3 cm travelled) — the
healthy bonus alone. The 1M run's `best_model.zip` banks **987.2 ± 2.9** over the
full 1000 steps and covers **0.43 m**. So "≈ 990" was a fair round number.

What was wrong is where the deck said to look for it. Slide 13 claimed the 1M
curve was "flat at exactly the statue line". It is not:

| tag, 1M run | at 1M |
|---|---:|
| `rollout/ep_rew_mean` | **16.4** |
| `rollout/ep_len_mean` | **65.3** |
| `eval/mean_reward`, best | **987.8, at 20k steps** |

The statue lives in the **eval** curve, and only for the first 100k steps —
after which that run degrades monotonically to `eval` 121 by 1M. `EvalCallback`
froze the 20k checkpoint as `best_model.zip`, and `evaluate.py --model best` is
the default, so **the shipped policy comes from the first 2% of the run** while
the headline curve never goes near 1000. `fig_ppo_curves` now plots rollout and
eval together and marks that checkpoint. The conclusion is unchanged and better
supported: the 3M run's eval does not clear the statue line until **1.64M
steps**.

Measured checkpoints, 20 episodes, deterministic:

| checkpoint | return | ep_len | distance |
|---|---:|---:|---:|
| zero torque (perfect statue) | 1000.6 ± 4.0 | 1000 | 0.03 m |
| uniform random torques | −35.2 ± 73.2 | 132 | 0.42 m |
| PPO 1M, `best_model` | **987.2 ± 2.9** | 1000 | **0.43 m** |
| PPO 1M, `final_model` | 54.3 ± 42.1 | 36 | 2.51 m |
| PPO 3M, `best_model` | 1879.1 ± 732.1 | 863 | 77.5 m |
| PPO 3M, `final_model` | **2316.2 ± 589.2** | 930 | **101.1 m** |

### Module 6 now previews the environment Session #3 actually uses

The litmus-test module was built on the pre-redesign `LegacyFlightEnv` — a
`MultiDiscrete` action space, a clearance budget, an unconditional alive bonus,
and a 60-line controller that beat PPO outright. All of it is superseded. The
module is rebuilt on `Flight4DEnv`: three ranked objectives, hard action
masking, the 4D exit gate, and the board re-measured on the finished 1.5M run.

`slides/scripts/extract_4d_board.py` (new) regenerates
`slides/data/air_4d_board.csv`; `fig_4d_board` plots it. 100 held-out seeds
(900000–900099), `hold_steps=5`, `runs/v4d_s123/final_model.zip`:

| agent | failed | ep_len | congestion | per 100 steps flown | exit_miss | clearances |
|---|---:|---:|---:|---:|---:|---:|
| **PPO masked, 1.5M [det]** | **1** | 49.1 | **194** | **3.95** | 0.072 | 2421 |
| PPO masked, 1.5M [sto] | 2 | 49.0 | 193 | 3.94 | 0.085 | 3410 |
| random-legal | 34 | 35.1 | 528 | 15.05 | 0.754 | 2924 |
| do-nothing | 96 | 12.5 | 551 | 44.04 | 0.000 | 0 |

These supersede the 0.27.0 table above, which was taken from a mid-run
checkpoint while the same run was still training. The direction is unchanged.

**Raw congestion cannot rank these rows.** The brick's 551 against the agent's
194 is a factor of 2.8; per step actually flown it is **11×**, because the brick
stops accumulating conflict-steps the moment it hits something. Every congestion
figure in the deck is now quoted both ways, and `extract_4d_board.py` carries
the steps-flown column `score_4d.py` does not.

**What the module no longer claims.** No hand-written controller exists for
`Flight4DEnv`, so "RL beats a heuristic" is untested here and the slide says so.
The agent beats the two baselines that need no training, and that is the claim.

Also in the deck: the Module 1 discussion frame drops its list of expected
answers; Exercise 7 and 8 point at `score_4d.py` and `train_4d.py` (25 min for
1.5M, down from 80); the wall-clock and deliverables appendices match; the title
page reads `VERSION` from the repo instead of a hard-coded string. 48 body
pages, within the 50-page budget.

### Fixed
- `solutions/README.md` still warned that the directory is gitignored and "not
  pushed". Untrue since 0.27.0 tracked the reference-solution code — and it is
  the doc someone reads *before* deciding whether a path there is safe to quote.
  It now says what actually keeps the directory away from participants (the
  filtered export, the docs-site exclusion, and the rule that nothing under
  `sessions/` may import from it), and that artefacts remain ignored.
- Both deck generators now emit a `GENERATED FILE — DO NOT EDIT BY HAND` banner
  naming the script and the `.bak`. `slides/intermediate.tex` and
  `slides/advanced.tex` are build products; a hand edit compiled fine and was
  destroyed by the next generator run with no error and no diff.

### Pending — the deck leads the move
`Flight4DEnv` becomes the participant-facing baseline solution and moves from
`solutions/advanced/reference_solution/` into `sessions/03-advanced/airtraffic/`
(`envs/flight_4d.py`, `scripts/score_4d.py`, `scripts/train_4d.py`,
`scripts/calibrate_4d.py`, `policies/autoregressive.py`,
`tests/test_flight_4d.py`). Exercises 7 and 8 and the cheat sheet already quote
the **destination** paths, so they are wrong until the move lands and right
afterwards — the only ordering that leaves no window where a participant-facing
deck points into the instructor bundle.

`slides/scripts/extract_4d_board.py` needs no edit on the day: it tries
`envs.flight_4d` under the session tree first and falls back to `env_4d` under
`solutions/`, and looks for the checkpoint in both `runs/` directories. Verified
against the old layout — the regenerated CSV is byte-identical.

`reward_probe.py` is **not** in the move (it probes `LegacyFlightEnv`, not
`Flight4DEnv`), so it stays in the instructor bundle and has been dropped from
the participant cheat sheet.

## [0.27.0] — 2026-08-27

### Result — the first agent on `Flight4DEnv` that is good on its own terms
100 held-out seeds (900000+, away from the training callback's block):

| # | agent | failed | congestion | conflict-free | exit_miss | clearances |
|---|---|---:|---:|---:|---:|---:|
| **1** | **trained [det]** | 2 | **184** | **55%** | **0.065** | 2395 |
| 2 | trained [sto] | 6 | 171 | 56% | 0.157 | 2933 |
| 3 | random-legal | 34 | 528 | 9% | 0.754 | 2924 |
| 4 | do-nothing | 96 | 551 | 4% | 0.000 | 0 |

Against the previous best on this environment: congestion 611 → **184**,
conflict-free 4% → **55%**, `exit_miss` 0.418 → **0.065**.

Normalised for survival — the comparison raw congestion cannot support, since a
policy that dies early accumulates fewer conflict-steps by existing for less
time:

| agent | steps flown | congestion / 100 steps |
|---|---:|---:|
| do-nothing | 1251 | 44.04 |
| random | 3509 | 15.05 |
| **trained** | **4902** | **3.75** |

Four times better than random per step flown, while flying 40% more steps.

The remaining weakness is objective 3: **48.9 clearances per 100 steps** against
random's 83.3. Cheaper than random, nowhere near frugal, and the only objective
that has not moved.

### Changed — the mid-air charge is proportional to the steps not flown
```python
charge(t) = collision_rate * w_safe * (max_steps - t)
```
450 at step 0, 225 at step 25, 9 at step 49, 0 at the horizon. One dimensionless
multiplier, expressed in `w_safe`, replacing the flat `w_collision`.

**The purpose is to regularise the return, not to punish.** An episode ending at
step 10 banks ten steps of cost and one running to 50 banks fifty, so returns are
not comparable across episodes and the critic fits that length-driven spread on
top of everything else. Charging the steps not flown completes every episode to
the same horizon. Deterring a crash is a side effect — a welcome one, since the
rate sits comfortably above the ~3.3 per step a real episode costs.

Note this reverses a decision made deliberately in 0.19.0 for `LegacyFlightEnv`,
where the charge was made time-*independent* on the grounds that "sizing it on
the steps remaining would make a mid-air at step 49 nearly free, which invites
reckless flying near the horizon". The reasoning differs because every term in
`Flight4DEnv` is a cost: charging the forfeited steps is the correct accounting
of what ending early avoids, and at step 49 there is genuinely only one step of
cost to forfeit.

### Added — two tests holding the charge
- `test_crash_charge_covers_the_steps_not_flown` — **per state**, since the
  charge now varies with the step: at every `k`, crashing must cost at least what
  flying on would have. It strips the terminal charge back out before comparing,
  or the test is circular and raising the charge raises its own bar.
- `test_crash_charge_falls_to_zero_at_the_horizon` — linear in steps forfeited,
  exactly zero at `max_steps`.

123 tests pass.

### Fixed
- `train_4d.py`'s run header referenced the removed `w_collision` and crashed on
  launch. It now prints `collision_rate`, and — as with the weights and
  `hold_steps` before it — reads from the environment rather than the CLI.

## [0.26.0] — 2026-08-27

### Changed — **breaking**: `Flight4DEnv`'s reward, rebuilt from first principles
Three terms and nothing else, replacing the potential-based pair:

```
r = −( w_safe · Σ_conflicts severity          predicted, summed over infringements present
     + (w_safe/(n+m)) · Σ_i dev_i             timeliness, always present, dev ∈ [0,1]
     + w_clearance · 1[clearance issued] )    per non-NOOP action
```

The scaling rule is now **asserted by a test**, not stated in a comment: the whole
fleet at maximum deviation costs exactly one severity-1 infringement, so each
aircraft carries `w_safe/(n+m)`. Measured split over an episode under a random
legal policy: **safety 91%, timeliness 5.9%, action 3.1%.**

**Why predicted severity is the right quantity.** `predict_conflicts` *omits* any
pair at or beyond `ALT_MIN_SEP` outright, and weights the rest by
`exp(−d_alt/ALT_MIN_SEP)`. So one flight level removes **39%** of the
infringement and the second removes the remaining **61%** by deleting the
conflict from the list. That weighting falls out of the simulator's own conflict
definition rather than an exponent chosen by hand — and it cannot drift away from
the scored rule, because it *is* the scored rule.

Potential-based shaping is gone. Predicted severity is already dense — a conflict
twenty steps out is in today's sum — so it does not need shaping to be learnable.
The trade is deliberate and recorded: direct costs *can* move the optimum, where
PBRS provably cannot.

Absolute scale reduced 10× (`w_safe` 30 → 3). Only ratios matter to the policy;
magnitude is what the critic must fit from a zero initialisation, and an episode
return near −80 is kinder than −800.

### Changed — **breaking**: the action mask is part of the observation
`Flight4DEnv.observation_space` is now `13·5 + 2 + 36 = 103`, with the legality
mask as the trailing `1 + 7·n_flights` entries and `mask_offset` naming where it
starts. `MaskInObsWrapper` is a deprecated no-op that asserts rather than
appending a second copy.

SB3's rollout buffer stores observations and nothing else, so a mask supplied any
other way is present when the action is *chosen* and absent when its
log-probability is *recomputed* — PPO then compares two different distributions
and optimises a ratio nobody intended. It fails as a wrong answer, not an error.

### Removed
- `w_4d`, `w_invalid`, and the invalid-action penalty. An invalid action is not
  something to learn to avoid; it is something to remove from the action space.
  The counter remains so a broken policy is loud rather than silent.

### Fixed — the same duplication bit twice
`train_4d.py` carried its own copies of the environment defaults and silently
overrode them — once for `hold_steps`, once for the reward weights, each time
launching a run that was not the run intended. Every env knob in the trainer now
defaults to `None` and falls through to `Flight4DEnv`, and the run header prints
what the **environment** ended up with rather than what the CLI asked for.

### Measured — why the previous run plateaued
Diagnosed on the 1.5M-step checkpoint, 60 seeds:

- The temporal component of `exit_miss` is **exactly 0.0000 on 100% of
  aircraft** — the symmetric speed pulse works. The entire residual 0.42 is
  *vertical*: permanent level changes never undone, on 105 of 109 touched
  aircraft.
- The policy used **2 of 35 actions** — 96.9% permanent `FL_DEC`/`FL_INC`, with
  the reverting macro and `RESUME` at **zero uses** — and issued its last
  clearance at step **1.4** (median 1), idling for the remaining 48.
- Nothing was masked out: `RESUME` was legal on 105 of 109 off-plan aircraft, and
  injecting it never hurt (Δcongestion −0.42, −0.18, −0.15, 0.00, 0.00 at steps
  3/8/15/25/40, Δ`exit_miss` negative throughout). It was free improvement left
  on the table.
- **Root cause was the reward, not exploration.** The old `Φ_safe` was linear in
  `d_alt`, so a one-level move — which leaves the pair squarely in conflict —
  collected 50% of the credit. The agent found that local optimum at step 1, took
  it on every aircraft, and stopped. It was optimising the shaping exactly as
  written.

### Slides
Module 4 is now **ten** mistakes. New frame *"We paid 50% for a manoeuvre that
resolves nothing"* with `fig_vertical_credit`: the hand-made linear factor
against the simulator's own `exp(−d_alt/24)`, beside the trained policy's
clearance histogram. Body is 50 pages, at the cap.

## [0.25.0] — 2026-08-27

### Changed — the advanced deck is now problem-first
The body was weighted towards *solutions*. It is now weighted towards
**defining the problem** and towards the **mistakes made building it**, which is
what a participant can actually use in a four-hour session.

| Module | was | now |
|---|---:|---:|
| 1 — The problem | 3 | **6** |
| 2 — What you get to design | 2 | 3 |
| 3 — What we score | 5 | 4 |
| 4 — **Nine mistakes we made** | — | **12** |
| 5 — Your design sprint | 4 | 3 |
| 6 — The competition | 7 | 6 |

Body stays at **49 pages**. The solution-heavy material — potential-based
shaping in full, both defect derivations, the network and masking slides, all six
prior-project field notes, the Kaggle metric — moved to `Extra slides`, now 35 frames.

### Added — Module 1 defines the problem before offering any answer
- **The sector, and what you are asked to do in it** — the conflict predicate,
  the no-turns restriction, and the three counted quantities.
- **The 4D exit gate** — `t* = s/v(0)`, `Δt = t̂ − t*`, why an untouched aircraft
  is free by construction, and the "slow everything down" hole this closes.
- **Four constants decide what is possible** — `DALT` 12 against `ALT_MIN_SEP`
  24; 10 levels at 2-level separation giving exactly 5 slots for exactly 5
  converging aircraft; `DVEL` 35 needing ~14 steps against a 1000 m radius; 50
  steps against a 100-clearance budget. **Three of the nine mistakes are one of
  these relations, missed.**
- **The tension at the centre of the problem** — `δs = Δv·H·Δt_step`,
  `δt = δs/v₀`, why returning to nominal *speed* does not return the *schedule*,
  and why a matched `+Δv … −Δv` pair is a `p²` event for an independent sampler.

### Added — Module 4, the mistakes, each with its arithmetic
Grouped so the classes transfer even when the specifics do not: **1–3** the
reward said something you did not mean; **4–5** the reward was fine and could not
be reached; **6–7** the action space was not what you thought; **8–9** the
measurement was wrong.

New measured figures, all from `slides/data/air_mistakes.csv`:

- **`fig_reward_scale`** — the two asymmetries that make a weight lie about its
  own strength. One conflict is worth 30 and is collected immediately; one
  clearance cost 0.35, giving **86:1 in favour of acting**, and the whole
  episode's clearance bill was 6.64 against that same 30. Second panel: a
  terminal charge is multiplied by `γ^T ≈ 0.75` before the agent feels it.
- **`fig_worse_than_random`** — `exit_miss` 0.590 random, **0.844 trained**,
  0.531 after recalibration. A trained agent worse than random on an objective is
  telling you that objective is not priced.
- **`fig_mask_legality`** — speed clearances legal on **13–16%** of idle
  (aircraft, step) pairs under a horizon-based dead-end test, **81–84%** under a
  schedule-based one. A mask that refuses everything is a missing action.
- **`fig_eval_size`** — the same two checkpoints, ranked one way on a 20-episode
  callback and the other way on 50 frozen seeds.
- **`fig_det_sto`** — an idle-biased policy scored by argmax issues 0 clearances
  and is byte-identical to the brick. Three runs were written off on that
  evidence and all three were still learning.

Plus `scripts/make_figures_mistakes.py`, which reads the CSV so every number on
these slides is traceable like every other number in the deck.

## [0.24.0] — 2026-08-27

### Result — the macro action set did NOT survive training
Two 1.5M-step runs, identical but for `action_set` (`macro` vs `resume`),
`action_mode=single`, hierarchical head, endpoint timeliness, n=5 m=5, seed 123.
Frozen board, 50 public seeds:

| # | agent | failed | congestion | timeliness | clearances |
|---|---|---:|---:|---:|---:|
| 1 | rule-based-resume | 2 | **133** | 0.034 | 874 |
| 2 | rule-based | 2 | 134 | 0.104 | 1311 |
| 3 | **control** @1.31M | **2** | 660 | 0.698 | 405 |
| 4 | control final | 5 | 582 | 0.639 | 415 |
| 5 | **macro** @1.31M | 6 | 508 | 0.709 | 2252 |
| 6 | macro final | 10 | 453 | 0.693 | 2097 |
| 7 | noop | 42 | 467 | 0.000 | 0 |

**The control ranks above the macro run.** The macro agent buys lower congestion
with more failed episodes — the pattern `failed_episodes`-first exists to catch.
It also issues **45 clearances/episode against the control's 8**: making the
effective intervention cheap in *decisions* made the policy prolific, not
selective.

**The 20-episode training callback said the opposite** (macro 3 failures / 213
congestion vs control 6 / 249) and inverted on 50 frozen seeds. Same lesson as
`EvalCallback` picking a best model from 5 episodes: **never conclude a
head-to-head from the training callback.**

This does not overturn the 0.22.0 injection probe. Its claim was *reachability*
and that still holds; *convergence* was never claimed, and this disconfirms it.

### Added — a fresh environment built around the 4D constraint
- **`solutions/advanced/reference_solution/env_4d.py`** — `Flight4DEnv`, a
  standalone `gym.Env` (not a subclass of the `LegacyFlightEnv` chain; only the
  kinematics are reused). `n + m = 5` with `n ~ U{2,3,4}` per reset.

  Every disturbing clearance **auto-reverts**, so the 4D recovery is inside the
  instruction rather than left as a second decision: `FL_INC2_T` climbs two
  levels and comes back, `SPD_UP_T` is a **symmetric pulse** (`+dv` for
  `hold_steps`, `-dv` for `hold_steps`, then nominal). One clearance each.

  Reward is `Φ_safe` (squared horizontal × vertical intrusion, zero outside
  either threshold) plus `Φ_4D` (quadratic in temporal and vertical deviation),
  both **potential-based**; the clearance count is a **transition cost**, charged
  once at issue, never shaped.

  The observation carries the temporal deviation `dt_exit`, the vertical
  deviation, a per-aircraft clearance counter and the manoeuvre countdown —
  which is what makes it Markov on a *single frame*, since headings are never
  commanded and every track is a straight line.

- **`policy_ar.py`** — a two-stage head (`aircraft → clearance`, stage two
  reading the selected aircraft's embedding) with a custom SB3 `Distribution`.
  Masks arrive in the observation via `MaskInObsWrapper`, because SB3's rollout
  buffer stores only observations: a mask supplied any other way is present when
  the action is chosen and absent when its log-probability is recomputed, and
  PPO then optimises a ratio between two different distributions.

- **`test_env_4d.py`** — 85 tests. The load-bearing ones assert that the
  autoregressive `log_prob` and `entropy` match a flat Categorical over
  `log p(i) + log p(c|i)`; that the shaping telescopes exactly to
  `γ^T Φ_T − Φ_0`; and that a speed pulse leaves `dt_exit` at zero.

### Fixed — two bugs the tests and probes caught, both consequential
- **The kinematic dead-end mask used the episode horizon, not the aircraft's own
  exit schedule.** Most aircraft leave well before step 50, so the remaining time
  was massively overstated and nearly every speed clearance looked like it would
  arrive early. Speed clearances were legal on **13–16%** of idle (aircraft,
  step) pairs — the velocity axis, the one the 4D tension is *about*, was
  effectively deleted. Now **81–84%**.
- **A symmetric pulse that was not symmetric.** `_advance_holds` ticks inside the
  step that issues, so a counter of `2·hold` spent `hold−1` steps fast and `hold`
  slow, leaving exactly one `dv·DT` of along-track debt — 70 m, or **0.298 s** at
  235 m/s, which nothing ever paid back. The counter is now `2·hold + 1`.
- `hold_steps` default 8 → **5**, and `train_4d.py`'s own default aligned to it.
  At 8 the pulse ran 34 s against a ~57 s crossing and the mask correctly refused
  it most of the time.

### Measured, not changed — the prior project uses GRU, not LSTM
`multi_agent/network/rlm.py`: GRU, 192 hidden, BPTT window 8, RLlib stateful
handling. No recurrence has been added here. SB3 ships only LSTM recurrence
(sb3-contrib `RecurrentPPO`), so matching it is not available off the shelf —
and `Flight4DEnv` is Markov on one frame, where that full simulator is not.

## [0.23.1] — 2026-08-27

### Documentation — brought in line with 0.22.0 and 0.23.0
- **[Reward design](reference/reward-design.md)** gains *What actually moves the
  gradient — measured*: the four-variant ladder, the false-negative table, and
  why the macro-action is the row that closes it. The three results a designer
  should internalise before tuning anything — endpoint-charging is necessary but
  not sufficient (3.0% → 3.0% help rate); **scale cannot create gradient where a
  term is exactly zero**; and decomposing a scalar reward per aircraft is
  algebraically a no-op.
- **[Environment API](reference/environment-api.md)** gains an *Action sets*
  section covering `"macro"`, the one-clearance cost, the rail refusal, and the
  `DALT` 12 vs `ALT_MIN_SEP` 24 arithmetic that motivates it.
- The **eleventh per-aircraft feature** is documented as *a standing instruction
  is in progress* rather than *resume-in-progress*, with the note that the slot
  was reused deliberately so the observation shape — and old checkpoints —
  survive the new action set.
- **[AirTraffic environment](advanced/airtraffic.md)**: the MDP-in-one-screen
  table said the observation was `11·k·F + 2` with a survival bonus and no
  terminal state. It is `11·k·F + 4·a + 3` (**593** at the scored `n=5 m=5`),
  there is no survival bonus since 0.19.0, and a **mid-air terminates** while
  every allowance merely truncates.
- Fixed: the API reference still gave `collision=5.0` as the default weight. It
  has been **10.0** since 0.18.0.

### Slides
- **New frame in the advanced deck**, *"The reward pays for the good move. The
  action space could not."* — the 0.9% / 17.3% / macro table, the 0.0%
  false-negative result, and the one-line reason a threshold is an action-space
  bug rather than a reward bug. Body is now **50 pages**, at the cap.
- Design-sprint **Track B** cites the measured macro result instead of listing
  macro-actions as a hunch; **field note #5** notes that the macro set is a third
  answer to the interface question, reached by measuring rather than arguing.

## [0.23.0] — 2026-08-26

### Added — `action_set="macro"`
- **`FL_INC2` / `FL_DEC2`** on `LegacyFlightEnv` — climb or descend **two flight
  levels booked by one instruction**, flown one level per step. **Opt-in**; the
  default `"basic"` and the existing `"resume"` set are untouched, so the frozen
  competition is unaffected.

  The physics does not change. `FL_INC2` at step *k* is **byte-identical** to
  `FL_INC` at *k* plus `FL_INC` at *k+1* — asserted over 8 seeds × 3 aircraft ×
  both directions in the new `tests/test_macro_action.py`. What changes is that
  the minimum effective intervention becomes **one decision instead of two**, and
  costs **one clearance instead of two**.

  Motivation, measured in 0.22.0: `DALT` is 12 and `ALT_MIN_SEP` is 24, so a
  single altitude clearance leaves a co-level pair in conflict. One `FL_INC`
  improves the sector in **0.9%** of cases; two consecutive ones in **17.3%**.
  That is a threshold, and a policy sampling independently each step reaches it
  with probability p².

- **`tests/test_macro_action.py`** — the equivalence, the one-clearance cost, the
  rail refusal, opt-in isolation, and supersession by a later clearance.
  154 tests pass in `sessions/03-advanced/airtraffic`.

- **`info["climbing"]`** — aircraft with levels still owed.

### Acceptance test — passed
Re-ran `reward_probe.py` with the macro in the command set, 46,080 injections,
24 public seeds:

| case | improves | no change | worse | mean Δcongestion |
|---|---:|---:|---:|---:|
| 1 altitude clearance | 0.9% | 96.9% | 2.2% | **+0.053** |
| 2 altitude clearances | 17.3% | 81.2% | 1.6% | −0.407 |
| **1 MACRO clearance** | **17.3%** | **81.2%** | **1.6%** | **−0.407** |

**Exact reproduction on every ground-truth column, at half the clearance cost.**
The macro's own commitment profile is a *smooth slope* (marginal −0.407 then
+0.052) — the threshold now sits inside one decision.

It lifts every reward, not only the shaped one. Mean Δ return at depth 1:

| variant | without macro | with macro |
|---|---:|---:|
| stock | −0.54 | −0.26 |
| endpoint | −0.14 | **+0.18** |
| dominant | −0.17 | **+0.48** |
| pairwise | +0.56 | **+1.63** |

`endpoint` and `dominant` were negative-mean and are now positive; the help rate
for the unshaped rewards goes 3.0% → **7.8%**. With the dense potential it stays
at ~21%. The two fixes are complementary and neither substitutes for the other:
**the potential supplies signal, the macro supplies reachability.**

### Changed
- **The eleventh per-aircraft feature is now "a standing instruction is in
  progress"** rather than "resuming" specifically — it covers a RESUME still
  flying home *and* a macro climb with levels still owed. Deliberately reuses the
  existing slot: `N_FEATS` stays **11**, the observation shape is unchanged, and
  old checkpoints still load. Under `basic`/`resume` the value is identical to
  before.
- `OrderInvariantPolicy`'s clearance mask gates `FL_INC2` / `FL_DEC2` on **two**
  levels of rail headroom rather than one.

### Deliberate asymmetry, documented
At the rail the macro is **invalid rather than half-executed**. Two singles
against the ceiling give one level plus an invalid — the exact half-manoeuvre the
command exists to remove; the macro books nothing and charges the invalid. This
is the only case where the equivalence breaks, and it breaks in the intended
direction.

### Still not established
That PPO converges with it. The probe measures **reachability from a do-nothing
policy**, not optimisation. This is the point at which the training run is worth
its 80 minutes.

## [0.22.0] — 2026-08-26

### Added
- **`solutions/advanced/reference_solution/reward_probe.py`** — action-injection
  probes over the reward-design space, parallel over seeds, with a six-panel
  figure. Writes `slides/data/air_probe.csv` (tidy, 184,320 rows) and
  `slides/figures/fig_probe.pdf`. `--quick` for a smoke test, `--plot-only` to
  re-draw without re-running.

  It answers a question training cannot: **does the reward pay for the first
  useful thing an agent could try?** From an all-NOOP rollout it injects one
  command on one aircraft, held for `depth` consecutive steps, and scores every
  candidate reward on the *same* trajectory — so cross-variant differences carry
  no sampling noise at all.

### Measured — the commitment threshold, confirmed
30,720 injections, 24 public seeds, n=5 m=5. Ground truth (`d_congestion`,
reward-independent), fraction of injections that improve the sector:

| | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---:|---:|---:|---:|
| **altitude** | **0.9%** | **17.3%** | 17.6% | 17.8% |
| altitude, mean Δcongestion | **+0.053** | −0.407 | −0.407 | −0.419 |
| **speed** | 5.2% | 8.2% | 9.9% | 10.2% |
| speed, mean Δcongestion | −0.020 | −0.124 | −0.185 | −0.195 |

**One altitude clearance is not a weak move, it is the wrong move** — it improves
the sector in 0.9% of cases and makes it worse in 2.2%, mean `+0.053`, the wrong
sign. Two improve it in 17.3%, a **20x** jump; the third and fourth buy nothing
(marginal `+0.001`, `−0.013`). That is `DALT` 12 against `ALT_MIN_SEP` 24, seen
as a rate. 95% CIs exclude zero everywhere.

The prediction it was tested against was sharp and it held: the effect appears on
**altitude and not on speed**, because `DVEL` opens along-track separation
continuously. Speed shows a decelerating slope, no cliff.

### Measured — the reward was never the bottleneck for *recognising* a good move
Splitting the same injections on whether congestion actually moved:

| case | improves | no change | worsens | **FN** | FP | signal on ties |
|---|---:|---:|---:|---:|---:|---:|
| altitude d=1 | 0.9% | 96.9% | 2.2% | **0.0%** | 0.0% | 61.8% |
| altitude d=2 | 17.3% | 81.2% | 1.6% | **0.0%** | 33.3% | 54.4% |
| speed d=1 | 5.2% | 90.9% | 3.9% | **0.0%** | 15.3% | 59.3% |

`FN = P(reward ≤ 0 | congestion improved)` is **0.0% in every cell** — the shaped
reward pays for every clearance that actually helped. It is not missing good
moves; there are barely any single good moves to miss.

`signal on ties` is what the dense potential buys: on the ~90% of injections
where the KPI does not move at all, the reward still says something 54–62% of the
time. That is the entire difference between a 3% and a 21% help rate.

### Measured — the reward ladder
Help rate at depth 1, six variants scored on identical trajectories:

| variant | %help | mean Δ | what it adds |
|---|---:|---:|---|
| stock 0.20.0 | 3.0% | −0.54 | — |
| endpoint | 3.0% | −0.14 | Defect 2 fixed: timeliness as a potential |
| dominant | 3.0% | −0.17 | + congestion weight 10 → 20 |
| **pairwise** | **21.0%** | **+0.56** | + predicted-conflict potential over pairs |
| peraircraft | 19.6% | 0.00 | …over aircraft (`max`) instead of pairs |

**Only the dense potential moves the help rate.** Endpoint-charging removes 74%
of the mean penalty and changes the help rate by *nothing*. Raising the
congestion weight so one event dominates the worst timeliness bill — the
lexicographic condition the leaderboard already applies — also changes it by
nothing, and doubles the variance. **Scale cannot create gradient where a term is
exactly zero.**

Two corollaries worth keeping:
- **Per-aircraft decomposition of a scalar reward is a no-op.**
  `Σᵢ ½ Σ_{pairs∋i} c = Σ_pairs c`. It only buys anything if you stop re-adding
  (per-head advantages in a custom loss) or change the functional form — and the
  `max`-per-aircraft form measured *worse* than the pairwise sum, because `max`
  discards every improvement except to an aircraft's single worst conflict.
- **Endpoint-charging is what makes the dominance condition affordable.** With a
  running charge the worst timeliness bill is `3.947 × 50 = 197.4` and one
  conflict dominating it needs `collision ≈ 790`; charged at the endpoint the
  bill is bounded at `3.947` and `collision = 16` suffices — 50× smaller.

### Consequence — do not train against a new reward yet
The reward already pays for the good move; the agent cannot **express** it.
Committing to two consecutive steps on one aircraft is a `p²` event for a policy
that samples independently each step, which is exactly the observed v15
pathology: **31.5% of the agent's interventions were a single clearance**, i.e.
structurally incapable of resolving anything.

The next lever is the **action space** — a macro-action (`CLIMB_2`, atomic over
two steps) or action repeat. `reward_probe.py` is its acceptance test: add the
macro-action to the command set and `depth=1` should reproduce the `depth=2` row.

## [0.21.0] — 2026-08-26

### Changed — slides
- **Both decks cut to 50 pages in the body.** Beamer's page count is
  `1 title + one page per section + one page per frame`, so the section count is
  part of the budget:

  | Deck | Body pages | was | Body frames | Sections |
  |---|---:|---:|---:|---:|
  | `intermediate` | **48** | 64 | 39 (was 55) | 8 (was 8) |
  | `advanced` | **49** | 72 | 39 (was 60) | 9 (was 12) |

  The advanced deck folds **Module 4.5 into Module 4** and **the post-mortem
  into Module 7**, which buys back two pages before a single frame is cut.

- **Nothing is deleted.** Every frame cut from a body is re-emitted under
  `\appendix / Extra slides` in the same deck — 10 frames in the intermediate,
  28 in the advanced. The long-form derivations (DQN, policy gradients, the full
  PPO and SAC treatments), the individual field notes, the `MaskablePPO`
  discussion and both command cheat sheets are still there to jump to.

- **The surviving frames were thinned, not just merged.** Overfull `\vbox`
  warnings — content running off the bottom of a slide — went from 24 to 18 in
  the intermediate deck and the worst case from 122 pt to 19 pt; the advanced
  deck body has none over 18 pt.

### Added
- **`scripts/rebuild_intermediate.py` and `scripts/rebuild_advanced.py`.** The
  decks are now generated from the pre-cut sources kept as `*.tex.bak`. Each
  script slices surviving frames out of its backup **verbatim by line range** and
  writes merged or rewritten frames out in full, so a frame is edited in exactly
  one place. A `FIXUPS` list applies repo-wide facts that drifted, so a re-run
  cannot reintroduce them.

- **`\figh[width]{max height}{name}`** in `preamble/rlbootcamp.sty` — as `\fig`
  but capped vertically as well, so a figure sharing a frame with two callout
  boxes cannot push them off the bottom. `\fig` scales on width alone, which is
  what produced most of the overfull frames.

### Fixed — slides that disagreed with the code
- **The advanced deck no longer describes the `+100` survival bonus as a live
  reward term.** It was removed in 0.19.0. The Module 4 frame now lists the four
  live terms plus the mid-air terminal charge, and presents the bonus as the
  *history* it is — including what removing it did to the baselines
  (brick `+75.04 → −524.00`, controller `+110.64 → −13.89`).
- **The `LegacyFlightEnv` case-study frame is current**: observation **593** at
  n=5 m=5 (was 552), congestion coefficient **10** (was 5), the three
  observation blocks, and mid-air as the only true terminal state.
- **The reference bar is re-measured at 0.20.0**, 100 public seeds:

  | agent | failed | congestion | timeliness | clearances |
  |---|---:|---:|---:|---:|
  | rule-based-resume | 7 | **258** | **0.0411** | 1667 |
  | rule-based | 7 | 262 | 0.0958 | 2508 |
  | noop | **83** | 904 | 0.0000 | 0 |

  The brick now **fails 83 of 100 episodes** on mid-air termination, which makes
  the `failed_episodes`-first rank key legible on the slide instead of abstract.
- `envs/airtraffic/` → `sessions/03-advanced/airtraffic/` in every command on
  both decks.
- The advanced deck's title stamp no longer claims every number is 0.17.0; it
  says which are 0.20.0 and which are not.

### Known stale — not fixed here
The learned-agent tables (`air_leaderboard.csv`, `airtraffic_litmus.csv`, and
the intermediate deck's whole of Module 6) are still **0.17.0** measurements,
taken before `DVEL` 10 → 35 halved the congestion, before the survival bonus was
removed and before mid-air termination. The slides that quote them now say so.
Re-measuring needs a PPO retrain against the current observation space — old
checkpoints will not load, the shape changed.

## [0.20.0] — 2026-08-21

### Changed — **breaking**
- **The exit gate matches position, level and time — not speed.** `N_FEATS`
  12 → 11; the signed velocity-deviation feature is gone. Observation is **318**
  at n=2 m=3, **593** at n=5 m=5. The scored KPI never included speed, so this
  aligns the observation with what was already being measured. `RESUME` still
  restores speed as well as altitude — you must return to nominal or you keep
  accumulating schedule error.

- **`RESUME` is gated on the traffic picture.** It is legal only when the
  aircraft is off its level, is **not already resuming**, and has **no predicted
  conflict**.

  The gate exists because the action's value flips sign sharply and the asymmetry
  is brutal. Injecting `RESUME` into a trained policy's rollout:

  | from step | Δ shaped return | Δ congestion |
  |---:|---:|---:|
  | 2 | **−249.75** | +54 |
  | 6 | −92.37 | +60 |
  | 14 | −28.69 | +23 |
  | 18 | −1.71 | +9 |
  | 22 | +1.62 | +1 |
  | 26 | **+2.04** | 0 |
  | 42 | +2.11 | 0 |

  **The downside is a hundred times the upside.** A policy sampling it uniformly
  correctly learns "never" — right on average, wrong in exactly the states where
  it pays. Masking removes the bad half instead of asking the agent to learn its
  way around it.

- **`RESUME` is inert rather than invalid** when it would change nothing: already
  on plan, or already resuming. No clearance charged, no invalid, no
  action-history entry. Charging those as invalids was teaching the agent the
  command was dangerous during precisely the early phase when every aircraft is
  still on plan — before it had ever seen the case where the command pays.

### Added
- **`AutoEntropyCallback`** — SB3's PPO has **no `ent_coef="auto"`**; the
  parameter is typed `float` and `PPO.train` has no auto path. Only SAC has one.
  This ports SAC's dual-variable mechanism:

  ```
  log_coef += lr * (target_entropy − measured_entropy)
  target = 0.35 × ln(n_actions)
  ```

  The project has now failed in both directions with a fixed coefficient —
  diffusing toward uniform when too high, collapsing onto NOOP when too low.
  Neither failure was about the coefficient; both were about the entropy, and the
  mapping between them shifts as the reward scale and advantage normalisation
  change underneath. `--auto-entropy`, `--auto-entropy-target`.

- **`--target-kl`**, default **0.03**. SB3's default is `None`, i.e. unbounded,
  so this is a new constraint rather than a loosened one; the epoch loop stops at
  `1.5 × target_kl`.
- **`--timeliness-scale`** exposes the endpoint-potential weight (default 20).

### Changed
- **`--warmup-frac`** replaces `--warmup-steps`, default **0.01** (15k of 1.5M,
  was 100k / 0.067).

## [0.19.0] — 2026-08-20

### Removed — **breaking**
- **The survival bonus.** `+MAX_CLEARANCES` was paid to every agent that reached
  the horizon, do-nothing included, which made it simultaneously the largest term
  in the return and the only one no policy could influence. Decomposed per episode
  at n=2 m=3 it was **+100.00 of the brick's +75.04 total** — the agent was being
  paid, overwhelmingly, for existing, and every real signal was a rounding error
  against it. `ShapedFlightEnv`'s per-step `alive_bonus` (the same money paid out
  differently) went with it.

  Effect on the stock return, 20 public seeds:

  | scenario | agent | with bonus | without |
  |---|---|---:|---:|
  | n=2 m=3 | do-nothing | +75.04 | **−524.00** |
  | n=2 m=3 | rule-based | +110.64 | **−13.89** |
  | n=5 m=5 | do-nothing | — | **−138.50** |
  | n=5 m=5 | rule-based | — | **−35.33** |

### Added
- **Mid-air termination.** Two aircraft at the same flight level within
  `COLLISION_RADIUS` (100 m) end the episode with `terminated=True` and a hard
  charge. This is the **only** genuine terminal state in the environment — the
  thing it exists to prevent has happened, so the state's value is zero.

  The radius was chosen by measurement, not assertion. Over 40 seeds, do-nothing
  brings co-level traffic within **2–10 m**; the rule-based controller's closest
  co-level approach at n=2 m=3 is **435 m**. At 100 m the brick collides in 97.5%
  of episodes and the controller in none, so the threshold separates the two
  behaviours rather than labelling both. Note this is a different question from
  `find_congestion`, which fires at 1000 m and two levels — that is a procedural
  violation you fly out of.

  The penalty is `max_steps × collision / divisor` — one conflicting pair held for
  the whole horizon, charged in the units of the congestion term it replaces.
  **Deliberately independent of when the collision happens**: sizing it on the
  steps *remaining* would make a mid-air at step 49 nearly free, which invites
  reckless flying near the horizon.

- `info["collided"]`, `_is_collision()`, `collision_penalty()`.

### Changed — **breaking**
- **Running out of an allowance now truncates rather than terminates.** Both the
  clearance budget and the invalid-action allowance set `truncated=True`, and the
  old `−MAX_CLEARANCES` hard penalty for invalid overflow is gone. Only a mid-air
  terminates. The distinction is not cosmetic: a truncated state still has a
  future worth bootstrapping from, a terminal one does not.
- `competition/kaggle/metric.py` mirrors mid-air termination, verified by the
  existing harness-parity test.

### Consequence worth watching
- **A near-do-nothing policy now dies at around step 10.** That is the truncation
  cliff reappearing in a new form — but this time the episode ends *because the
  problem happened*, which is the right moment to end it, rather than because the
  agent ran out of allowance before the problem arrived.
- The do-nothing baseline now **fails 20 of 20 scored episodes** and ranks last on
  `failed_episodes`, which is the rank key working exactly as designed.

## [0.18.0] — 2026-08-20

### Added
- **`action_set="resume"`** on `LegacyFlightEnv` — a sixth command, `RESUME`
  ("resume own navigation"). One instruction, one clearance charged, and the
  aircraft flies itself back to its entry level and speed at one increment per
  step. **Opt-in; the default `"basic"` is unchanged.**

  Motivation, measured: a trained policy flew the outbound manoeuvre correctly —
  climbing exactly two flight levels, precisely `ALT_MIN_SEP / DALT` — and then
  never came home. 0% of touched aircraft ended on plan against 95.8% for the
  rule-based controller. A *timed* return costs zero extra congestion and
  improves the shaped return, so the reward already wanted the behaviour; the
  agent could not find it. Each clearance is drawn independently, so a
  half-finished manoeuvre is as likely as a finished one.

  `PriorityLevelController(action_set="resume")` uses it: same safety, same
  timeliness, **34–36% fewer clearances**.

- **Hierarchical policy head** — `OrderInvariantPolicy(head="hierarchical")`.
  Stage one picks an aircraft or idles; stage two picks that aircraft's clearance
  from its own embedding. Written as `log p(i) + log p(c | i)` over the unchanged
  flat action space, so it remains a single `Categorical` and SB3's `log_prob`,
  entropy and KL stay correct. Requires `action_mode="single"`.

- **Observable action history** — a ring of the last `a_frames` clearances
  issued, `(valid, aircraft, command, age)` each. A ring of *events* rather than
  one slot per step, which is why `age` is explicit. The policy **scatters** each
  event back onto the aircraft it names, so permutation equivariance survives a
  block that refers to aircraft by index.

- **`invalid_headroom()`** as a third global scalar, and a **resume-in-progress
  flag** as the twelfth per-aircraft feature. The action ring shows that a
  `RESUME` was issued, never whether it completed.

### Changed — **breaking**
- **`DVEL` 10 → 35.** At 10 a speed clearance opened 20 m of along-track offset
  per step against a 1000 m conflict radius — 50 steps of a 50-step episode — so
  neither the trained agent nor the controller ever issued one, and the
  along-track term of the timeliness KPI was always exactly zero.

  It helped, and it did not make speed competitive. Conflicts cleared by holding
  one clearance for two steps: speed 0% → **35%**, altitude 76–82%, do-nothing
  **29%**.

  **It also halved the problem.** Faster traffic transits the hotspot sooner:

  | scenario | agent | was | now |
  |---|---|---:|---:|
  | n=2 m=3 | do-nothing | 180 | **88** |
  | n=2 m=3 | rule-based | 41 | **11** |
  | n=5 m=5 | do-nothing | 762 | **380** |
  | n=5 m=5 | rule-based | 129 | **68** |

  Every reference figure measured before this is stale.

- **Default congestion coefficient 5 → 10**, forced by the above. With half the
  congestion the congestion term halved while the clearance cost did not, and the
  stock return ranked **do-nothing above the rule-based controller at three of
  four configurations**. 10 restores a clear margin everywhere. General rule:
  halve the congestion, double the weight.

- **The observation is entirely in `[-1, 1]`** and is now three blocks —
  flights, actions, globals. `N_FEATS` 11 → 12, `N_GLOBALS` 2 → 3. Shape is
  `12·k·F + 4·a + 3`: **343** at n=2 m=3, **643** at n=5 m=5. Old checkpoints
  will not load against the new space.

### Fixed
- **`agent_action_set` read only `agent.action_space`.** A hand-written
  controller has none, so `rule-based-resume` was scored in a `basic`
  environment where every `RESUME` was rejected as invalid and no aircraft was
  ever handed back its level — turning the best baseline into a worse one,
  silently. Agents may now declare their command set.

### Measured, not changed
- **Pinning `COLL_DIVISOR` to the scored value made things worse.** At n=2 the
  divisor is 1 against 4 at n=5, which looked like a confound worth removing.
  Pinning it **collapsed both runs to zero clearances** (congestion 165) while
  the unpinned run reached 62. The apparent artefact was the only thing making
  safety loud enough to learn from.
- With the divisor left alone, `RESUME` took KPI timeliness from ~0.28 to
  **0.095** — but congestion rose from 5 to 62 in the same run. Coming home
  re-exposes an aircraft to traffic, and congestion ranks first.

### Documentation
- New: **[The rule-based controller](reference/rule-based-controller.md)** — the
  full algorithm, both phases, the arithmetic behind `SEP_LEVELS = 2`, the
  `RESUME` variant, what it deliberately does not do, and four specific
  weaknesses to attack.

## [0.17.0] — 2026-08-17

### Added
- **`CollisionCourseSimulator.advance_into_airspace()`** — uniformly time-shifts a
  freshly built scenario until every flight is inside the airspace box.

  `reverse_to_limit` deliberately rewinds each flight until it is *outside* the
  sector, so that it flies across rather than starting mid-way through. Left
  there, the episode opens on an empty screen: positions are normalised by
  `airspace_limit`, so an aircraft that has not arrived yet reads outside
  `[-1, 1]` and the agent is asked to plan around traffic it cannot properly see.
  The shift is uniform — relative geometry, closing speeds and arrival order are
  untouched, only the origin of the clock moves. Measured over the 20 default
  seeds: 16–36 s of shift, every aircraft inside at t=0.

- **`action_mode`** on `LegacyFlightEnv` and `OrderInvariantPolicy`. The default
  `"simultaneous"` is unchanged (`MultiDiscrete([5]*n_flights)`); `"single"` gives
  `Discrete(1 + 4*n_flights)` and commands at most one aircraft per step, which is
  closer to how a controller works and makes the clearance budget bite. Permuting
  aircraft permutes the four-logit blocks and leaves NOOP alone, so equivariance
  survives. `decode_action()` maps both onto the same command vector, so the
  simulator, KPIs, renderer and submission CSV are unaffected.

- **`budget_remaining()`** as a second global observation scalar — see the
  corrected 0.16.0 entry. The observation is **552** values at the defaults.

### Changed — **breaking**
- **`advance_into_airspace` changes every scenario.** A seed no longer names the
  same initial state, so every reference number in the handbook was re-measured
  against current code. The orderings did not change; the magnitudes did.

  | table | was | now |
  |---|---|---|
  | competition, rule-based congestion | 168 | **129** |
  | competition, noop congestion | 778 | **762** |
  | baseline n=5 m=5, rule-based return | 71.55 | **69.68** |
  | baseline n=5 m=5, do-nothing return | 53.29 | **54.71** |

### Fixed — documentation that disagreed with the code
- The observation was documented as `551` values with **one** trailing scalar in
  three places. It is `552` with **two**.
- The `noop_bias` measurement was quoted from a pre-shift run throughout. Current
  numbers: a scenario contains **40.32** conflict events (was 41.57), the busiest
  stretch is steps **10–21** (was 20–35), **62.7%** occur at step 15 or later
  (was 89.4%), and an unbiased policy experiences **16%** of them (was 3.8%).
- **The claim that `noop_bias=3.5` lifts first-iteration `ep_rew_mean` from 22.0
  to 71.2 did not reproduce.** Measured on the reference setup it is **−23.9 →
  17.6** — still below the ~54.7 do-nothing baseline. The docs now say what the
  bias actually buys: presence in the second half of the episode, not competence.
- `--collision-coeff` was documented as defaulting to `1.0`; it is `5.0`. Eight
  flags were missing from the training table, and the `conf/default.yaml` excerpt
  did not match the file.
- The environment docstring, the package README and `train.py --noop-bias` help
  all still described the 6-feature observation.

### Reference numbers — the PPO retrain
Three runs, 1.5M steps each, n=5 m=5, seed 123, identical hyperparameters;
evaluated over 30 seeded deterministic episodes (`best_model.zip`):

| agent | return | ep_len | congestion / ep | clearances | touched |
|---|---:|---:|---:|---:|---:|
| **rule-based** | **69.68 ± 9.73** | 50.0 | **8.0** | 32.8 | 4.0 |
| do-nothing | 54.71 ± 13.50 | 50.0 | 36.2 | 0.0 | 0.0 |
| PPO multi-head, `noop_bias=3.5` | 51.90 ± 14.94 | 50.0 | 36.1 | 0.9 | 0.6 |
| random control | −29.36 ± 9.47 | 14.5 | 6.7 | 115.1 | 10.0 |
| PPO multi-head, unbiased | −26.64 ± 8.38 | 17.9 | 9.0 | 104.1 | 9.9 |
| PPO flat `MlpPolicy` | −44.26 ± 35.56 | 14.4 | 4.4 | 131.5 | 10.0 |

Two distinct failures, and the distinction is the teaching point:

- **The unbiased runs never reach the problem.** They truncate around step 14–18,
  and the multi-head one is statistically indistinguishable from random actions.
- **The biased run collapses onto the brick.** It reaches the horizon and then
  learns that the best thing it knows how to do is nothing: 0.9 clearances per
  episode, 0.6 aircraft touched, congestion within noise of do-nothing. Its
  `final_model.zip` is the do-nothing policy exactly — **0.0 clearances**, return
  54.57. The survival bonus is `+100` and unconditional, so the dominant term pays
  for existing.

Fixing the truncation got the agent to the end of the episode. It did not give it
a reason to do anything once it got there.

## [0.16.0] — 2026-08-14

### Added
- **`CollisionCourseSimulator.predict_conflicts(horizon)`** — the conflicts that
  *will* occur if nothing changes, in closed form. Absent a clearance every track
  is a straight line, so solving `|r₀ + Δv·t| = hotspot_limit` per pair gives the
  conflict window and its vertex gives closest approach; no rollout is needed.

  Each record carries `t_enter`, `t_cpa`, `duration`, `d_cpa`, `d_alt`, `center`,
  `radius` and `severity`. Severity decays exponentially in **both** separations,
  so a pair passing head-on at co-altitude scores near 1 and one clipping the zone
  boundary scores near `e⁻¹` — the ordering that lets a controller triage.

  Validated over 30 seeds: **342/345 predicted conflicts occurred, no real conflict
  was missed**, timing error median −1.21 s against `DT=2`. The 3 that did not
  occur averaged severity 0.240 against 0.929 for those that did.

- `LegacyFlightEnv.conflicts()` — the same list, cached per simulator advance, so the
  observation and any reward built on it share one evaluation.

### Changed — **breaking**
- **The observation is now 11 features per aircraft, not 6** (`552` values at
  `n=5, m=5, k_frames=5`, was `301`). Every stored checkpoint is unloadable.
  The five new features:

  | # | Feature | Why |
  |:--:|---|---|
  | 6 | worst **predicted** conflict severity | `find_congestion` only fires once a conflict has already happened — far too late to guide the clearance that would have prevented it |
  | 7 | squashed time until that conflict | *when* is as actionable as *whether* |
  | 8–10 | signed log-scaled error on the **5D exit gate** — schedule, altitude, velocity | exactly what timeliness is scored on, so the agent can see its own scored quantity rather than infer it |

  Deviations are **signed** (two levels high and two low need opposite
  corrections; a magnitude cannot say which) and **log-scaled** (`log1p` is
  steepest at zero, which is the target), clipped at the full physical range.

- **A second global scalar: `budget_remaining()`**, the fraction of the clearance
  budget still unspent. The episode *truncates* when that budget runs out, so
  without it the agent is rationing a resource it has no reading of — and the
  horizon it is spending against is where the survival bonus pays out. Both
  globals bypass the order-invariant policy's pooling and enter the context
  directly, since neither belongs to any one aircraft.

### Fixed
- `_signed_log` was applied per scalar, 31 times a step, and numpy scalar dispatch
  cost more than the conflict prediction it sits beside. Vectorised: the enriched
  environment now steps in 0.93 ms, marginally *faster* than the old one.

## [0.15.0] — 2026-08-14

### Fixed
- **`render()` did not show the trajectory that was actually flown.** It
  `deepcopy`-ed the *final* simulator state and called `reverse_clock()` back to
  the start — which only undoes **position**. Altitude and velocity are mutated in
  place by clearances and cannot be run backwards, and the recorded clearances
  were read only to pick a marker colour, never re-applied.

  The result: every aircraft was drawn at its **end-of-episode flight level from
  frame 0**. A climb that resolved a conflict was invisible, because the
  separation looked like it had always been there. Measured on the rule-based
  controller, seed 3: altitude error up to **96** (eight flight levels), and only
  **4 of the 9** flight levels actually flown ever appeared. If an agent used
  velocity clearances the positions were wrong too, since the rewind used the
  final velocity.

  `LegacyFlightEnv` now records a per-step `state_history` and `render()` replays it.
  Altitude error is 0.000 and all flight levels appear.

### Changed
- Conflict circles in the render are drawn **once per frame** rather than once per
  aircraft (they were being layered `n_flights` times), and at `alpha=0.55`
  instead of `0.1` so they are actually visible.
- The render title now reports **live** conflicts alongside the running maximum.

### Added
- `LegacyFlightEnv.state_history` — `(n_flights, 5)` of `(x, y, altitude, velocity,
  heading)` per step, in raw simulator units.
- Contract test asserting the recorded trajectory matches the flown one (42 total).

## [0.14.0] — 2026-08-13

### Added
- **[Reward design](reference/reward-design.md)** — a new reference page documenting
  the whole reward: what each of the four terms is for, the measured decomposition
  of where the episode return comes from, the two places the stock formulation
  works against the ranking it serves, and why potential-based shaping is the safe
  knob to turn.

### Fixed
- The API reference's reward section documented **three** terms; the environment
  has had **four** since timeliness landed. Corrected, with the default weights
  and a pointer to the design page.

## [0.13.0] — 2026-08-13

### Added
- **`OrderInvariantPolicy(noop_bias=...)`** — an initial logit bias towards NOOP,
  defaulting to `0.0` so no existing baseline changes.

  It exists because of a measurement. A freshly initialised policy is near-uniform
  over five clearances, so ten aircraft issue ~8 clearances per step and the
  environment truncates at `MAX_CLEARANCES` around **step 13**. But the busiest
  stretch is steps 10–21 and 62.7% of loss-of-separation events happen at **step
  15 or later**. An unbiased policy therefore experiences 6.45 of the 40.32
  conflicts in a scenario (**16%**) and never once reaches the horizon.

  It is not failing to solve the problem so much as not being shown most of it.
  No reward change can fix that, because the states in question are never
  visited.

  With `noop_bias=3.5` (`p(NOOP) ≈ 0.89`) every episode reaches the horizon, 79%
  of conflicts are experienced, and first-iteration `ep_rew_mean` goes −23.9 →
  17.6. That is still below the ~54.7 do-nothing baseline: the bias buys
  presence, not competence. Students hitting a flat learning curve on the Flight
  Challenge should reach for it first anyway.

### Changed
- Reference trainer (`solutions/`, not shipped) gained a **linear warm-up →
  half-cosine** learning-rate schedule ported from the the prior ATC project single-agent ATC
  trainer, and now defaults to 2M steps.

### Tests
- Two regressions on `noop_bias`: that the default stays off, and that a biased
  policy reaches the horizon where an unbiased one provably does not (41 total).

## [0.12.0] — 2026-08-13

### Added
- **Post-mortem harness** (`competition/postmortem.py`, `competition/perturb.py`,
  `python -m competition.run postmortem`). A leaderboard position is one
  measurement of one policy under one set of conditions; these passes ask what it
  cannot:

  | Pass | Question |
  |---|---|
  | do-nothing | Does it beat the brick? |
  | stochastic | How much of the score survives sampling instead of argmax? |
  | next-best | Swap every decision for the runner-up — how much rested on thin margins? |
  | ε-random | Corrupt actions with probability ε ∈ {0.05, 0.1, 0.25}. |
  | best-extraction | k stochastic rollouts per seed, keep the best. |

- **Trophy computation** (`award_trophies`) for Golden Holding Pattern,
  Zero-Conflict Wings, Minimal-Intervention, Iron Stomach and The Brick with
  Wings. **Black Box** and **Icarus** are deliberately not computed — they are
  jury calls, and pretending a metric decides them would be worse than admitting
  a human does.

- **15 post-mortem tests**, covering perturbation semantics, pass skipping and
  trophy eligibility.

### Design notes
- **ε perturbation is per aircraft, not per step.** The action factorises across
  flights, so corrupting the whole vector at once would be a far blunter
  instrument than the robustness curve is meant to be.
- **Heuristics skip the passes that need an action distribution** and say so,
  rather than reporting a fabricated one. Their stochastic gap is zero because
  they have nothing to sample from, which is why `Iron Stomach` excludes them —
  a robustness trophy must not go to something that was never perturbed.
- **`best_extraction` seeds torch explicitly.** Sampling draws on the global RNG,
  so without it "best of k" changes between runs and the number cannot be quoted.
- The do-nothing baseline is not scored against itself: it reports as *the bar*
  rather than as a competitor that failed to beat itself.

### Observed
Random perturbation makes the do-nothing baseline **safer** — congestion falls
from 253 to 146 at ε=0.25 over 8 seeds. Random clearances scatter aircraft across
flight levels, and scattered aircraft do not conflict. A participant reading "my
agent got safer when I added noise" as evidence of robustness has found this
effect, not a good policy, which is exactly why the pass is worth showing them.

## [0.11.0] — 2026-08-13

### Added
- **Kaggle public-leaderboard metric** (`competition/kaggle/`). `metric.py` is the
  custom evaluation metric; `export.py` writes the solution file.

  The metric cannot import this package — Kaggle's sandbox has no gymnasium, no
  matplotlib and no bootcamp code — so it is necessarily a **second
  implementation of the rules**, and drift is the standing risk. Two things
  contain it:

  - **The scenarios ship in the solution file.** The metric never regenerates a
    scenario from a seed, it re-flies one it was handed, so the whole scenario
    generator is out of scope for divergence. 50,000 rows / ~4.6 MB for 100 seeds.
  - **An equivalence test pins the rest.** `test_kaggle_metric.py` asserts the
    metric and the real environment agree on every KPI across four action
    distributions and the rule-based controller. Change the dynamics or the KPIs
    and that test is what reports the metric went stale.

- **Lexicographic ranking packed into one float.** Kaggle sorts a single number,
  so `pack()` folds `(failed, congestion, timeliness, clearances)` positionally:
  each component clamped to a documented bound, weighted by the product of all
  lower-priority ranges. The span stays under `2**53` so the value is exact in
  float64 — tested, because losing precision here would silently reorder the
  board rather than raise anything. Out-of-range values clamp rather than wrap, so
  an absurd submission saturates at the bottom instead of reappearing at the top.

  The fifth key component, distinct aircraft touched, does not fit that budget and
  breaks ties on the private leaderboard only.

- Participant-facing errors are raised as `ParticipantVisibleError` with the fix
  attached (rebuild with `competition.run submit`, actions must be 0–4, and so on).
- **15 metric tests**, including one asserting the metric never imports torch,
  gymnasium, matplotlib or SB3.

## [0.10.0] — 2026-08-13

### Added
- **Competition scoring harness** (`sessions/03-advanced/competition/`). We own
  the score; students own their environment and policy.

  Ranking is lexicographic and every component is lower-is-better:

  ```
  (failed_episodes, congestion_total, timeliness_bucket, clearances_total, involved_total)
   └ safety ──────────────────────┘  └ timeliness ────┘  └ efficiency ─────────────────┘
  ```

  Three decisions in there are load-bearing rather than cosmetic:

  - **`failed_episodes` ranks before safety.** An episode that ends early
    accumulates fewer congestion events purely by existing for less time, so
    ranking on safety alone would make crashing out a strategy. This is also why
    completion is measured from the step counter and *not* from `truncated` —
    LegacyFlightEnv sets that flag both at the horizon and when the clearance budget
    runs out, so the flag cannot tell success from an early exit.
  - **Timeliness is bucketed.** It is a float, so exact ties never occur, so
    efficiency could never break one and the third objective would be decoration.
  - **Replay is numpy-only.** The public leaderboard re-flies a submitted action
    trace inside Kaggle's metric sandbox, where torch is unavailable; `agents.py`
    imports SB3 lazily and a test asserts torch never enters `sys.modules` on the
    replay path.

- **Action-trace submissions.** One row per (seed, step, aircraft), so there is no
  aggregate number in the file to inflate — the score comes from re-flying the
  actions. `rollout_agent` and `replay_actions` are verified to agree to the last
  digit of `timeliness_mean`, which is what makes the public board trustworthy.
- **Seed lists.** 100 public seeds ship with the repo; the private list is derived
  from a salt, gitignored, and regenerable from the salt alone
  (`python -m competition.make_seeds`).
- **CLI** — `python -m competition.run {submit,replay,compare}`.
- **19 scoring contract tests** covering objective priority, the early-exit
  exploit, bucketing, submission validation and replay fidelity.

### Reference numbers
20 public seeds, deterministic, score v1.0.0:

| # | agent | failed | congestion | timeliness | clearances | touched |
|---|---|---:|---:|---:|---:|---:|
| 1 | rule-based | 0 | 129 | 0.0348 | 623 | 80 |
| 2 | noop | 0 | 762 | 0.0000 | 0 | 0 |

The brick is perfectly on time and perfectly efficient — it never touches an
aircraft — and loses anyway, because safety is compared first.

## [0.9.0] — 2026-08-13

### Added
- **`policies.OrderInvariantPolicy` — now the default policy.** Ported from the
  the prior ATC project project's `ATCEncoder`: a shared per-aircraft MLP with LayerNorm, masked
  mean/max pooling for traffic context, and one shared head per flight that sees
  `[own embedding | context]`.

  Traffic is a *set*, not a vector. SB3's flat `MlpPolicy` ends in a single
  `Linear(hidden, 5·n_flights)`, so the weights choosing flight 0's clearance are
  separate from those choosing flight 7's — one idea has to be learned once per
  slot, and swapping two aircraft produces an unrelated observation. The shared
  encoder makes permutation equivariance hold **by construction**; the contract
  tests assert it to 1e-5, and value invariance is exact.

  Clearances that would run past an altitude or speed rail are masked at the
  logits, which drives invalid clearances to **exactly zero** in rollout. `NOOP`
  is always legal, so no row is ever fully masked.

  `train.py --policy mlp` keeps the flat baseline for comparison, and `--no-mask`
  isolates masking from weight sharing.

  > the prior ATC project's own policy is autoregressive (aircraft head → clearance head) because
  > it issues one clearance at a time. LegacyFlightEnv commands every flight
  > simultaneously, so there is no "which aircraft" decision to factor out — the
  > encoder ports, the head does not. Adopting the autoregressive action space
  > would cap interventions at `max_steps`, and the rule-based baseline already
  > needs 63.6 clearances at `n=12, m=3`.

- **Ten new policy contract tests** covering equivariance, value invariance, mask
  legality, zero probability on masked actions, and a training smoke test.

### Changed
- **Observation is now a 6-tuple per frame**, adding **normalised time-to-exit**:
  how long an aircraft has left in the sector on its current track. Without it the
  agent can see where traffic is but not how long it stays a problem — and it is
  what makes "which aircraft should I delay?" answerable, since delaying one that
  is about to leave is nearly free.

  Computed by ray-vs-box slab intersection (`CollisionCourseSimulator.time_to_exit`),
  not "first wall crossing ahead": flights **spawn outside the sector and fly in**,
  so the first crossing ahead is the *entry*, and reporting it would make an
  aircraft that has not arrived yet look like one about to leave.

  Squashed as `t / (t + horizon)` rather than clipped to `[0, 1]`. Clipping pinned
  every flight at 1.0 for the first third of the episode — they all spawn outside,
  so they all exit later than the horizon — leaving the feature with no gradient
  exactly when the agent is choosing whom to touch.

- **`k_frames` default 10 → 5.** With six features per frame the observation is
  `6·5·7 + 1 = 211` at `n=5, m=2`.

## [0.8.0] — 2026-08-13

### Changed
- **Observation stacks `k_frames=10` history frames** instead of current+previous
  (`LegacyFlightEnv(k_frames=...)`, class default `K_FRAMES = 10`). Per flight the
  layout is `[now | now-1 | ... | now-(k-1)]`, newest first, so index 0 is still
  the current state and `obs[:5]` still reads as before.

  A single frame is not Markovian here: position alone cannot say whether an
  aircraft is accelerating, or how long two flights have been closing. One
  difference gives velocity but not acceleration. Ten frames put the recent
  trajectory of every aircraft in the observation, which widens the scope of the
  MDP the policy is solving. `k_frames=2` reproduces the old observation exactly.

  > [!WARNING]
  > **This invalidates every checkpoint trained before this release.** For
  > `n=5, m=5` the observation goes from 101 to 501 dimensions and SB3 refuses to
  > load: `Unexpected observation shape (501,) ... please use (101,)`. The
  > reference PPO runs under `solutions/` need retraining, and the trained-agent
  > rows in the baseline and syllabus tables are marked *pending retrain* until
  > they are.

- **Default congestion weight raised to `collision=5.0`** (was `1.0`). At the old
  weight a congestion event cost less than the clearances needed to prevent it,
  so doing nothing out-scored the rule-based controller — the reward ranked
  agents in the opposite order to the KPIs, and an agent trained on it learns to
  stop intervening. Safety is priority 1 in the competition ranking and the
  reward now says the same thing.

  The old behaviour is one flag away and is now an *exercise* rather than an
  accident: `evaluate_baseline.py --collision-coeff 1.0` inverts the ranking on
  demand (at `n=12, m=3`: noop 86.4 vs rule-based 64.7, while causing 2.7× more
  congestion).

- **The rule-based baseline flies aircraft home.** `PriorityLevelController` used
  to park each aircraft at its deconfliction level and freeze it there, which
  under the new timeliness KPI is a job left half done. It now has a second
  phase: once a flight is diverging *and* a hotspot radius clear of every other
  converging flight, it is returned to its entry level.

  The separation trigger matters more than it looks. All converging flights share
  an entry level, so a naive "recover once past the hotspot" rule re-creates the
  conflict it just resolved — congestion at `n=8, m=0` went *up* from 21.6 to
  41.9 before the trigger was keyed on separation from peer traffic instead of
  distance from the hotspot. Tuned (`RECOVERY_SEP_FACTOR = 1.0`) it now beats the
  old non-recovering controller on **both** axes: congestion 19.6 vs 21.6, and
  timeliness 0.08 vs 1.08.

### Added
- `evaluate_baseline.py --collision-coeff` and `--k-frames` for reproducing the
  reward-inversion and history-depth comparisons from the CLI.
- Six new contract tests covering stack depth, padding at reset, newest-frame-first
  ordering, and the KPI keys.

### Fixed
- `decode_observation()` no longer assumes a 2-frame observation. It takes
  `n_flights` or `k_frames` (exactly one) and derives the other; length alone
  cannot separate "many flights, few frames" from the reverse.

## [0.7.0] — 2026-08-13

### Changed
- **Directory names no longer contain spaces.** The two space-named session
  directories made every `cd`, path and shell snippet awkward. The repository is
  now laid out by session, with the shared AirTraffic package hoisted to the root:

  | Was | Now |
  |---|---|
  | `Tutorial Hands-On Session: Tabular RL & Discrete MDPs/` | `sessions/01-fundamentals/` |
  | `RL Intermediate .../Crippled Ant/` | `sessions/02-intermediate/crippled-ant/` |
  | `RL Intermediate .../Airtraffic/` | `envs/airtraffic/` |

  AirTraffic moved out of the intermediate directory because it is used by both
  Session #2 (the litmus test) and Session #3 (the competition) — filing it under
  one of them was misleading. `sessions/03-advanced/` is new and holds the
  competition material. Every path reference in the docs, notebooks and READMEs
  was updated; the moves are pure renames, so history is preserved.

- **Intermediate exercises renumbered `1`–`8`.** They were numbered by module
  (`3.1`, `3.2`, `4.1`, …), which meant the first thing a participant does was
  labelled "Exercise 3.1" — implying two earlier exercise sets that never existed,
  since Modules 1–2 are theory. Numbering now runs straight through the session.

### Added
- **Competition rules page** (`docs/sessions/competition.md`) — the student-facing
  statement of the Flight Challenge: the three ranked objectives, the submission
  zip format, the contract-test gate, and the public/private leaderboard split.
- **The scoring objective is now stated precisely**: safety, then timeliness, then
  efficiency, compared lexicographically.
- **Timeliness KPI and the 4D exit point** (`LegacyFlightEnv._exit_deviation`,
  `timeliness_penalty`). Every aircraft enters with a plan — a point, an altitude
  and a time it was headed for. A clearance is a detour, not a new destination:
  an aircraft slowed to open a gap must speed back up, one stepped down a level
  must climb back. The KPI is the log-scaled distance still remaining to that
  plan, and it closes the degenerate "slow every aircraft down" solution that
  would otherwise top a safety-only leaderboard. Untouched traffic scores exactly
  zero by construction. Also added as a reward term (`rew_coeffs.timeliness`).
- **`step()` now returns a populated `info` dict** — `congestion_events`,
  `timeliness`, `exit_deviation`, `clearances`, `aircraft_involved`, `invalids`.
  It previously returned `{}`, so every KPI had to be scraped out of environment
  internals. Four new contract tests assert the keys and their anchor behaviours,
  and a student environment that drops one cannot be scored.

## [0.6.0] — 2026-08-07

### Changed
- **Setup is now documented in exactly one place** (board task **A1**). This
  repository had its own `docs/setup/` — installation, Conda workflow and
  troubleshooting — that shadowed the public
  [participant primer](https://sarl-plus.github.io/rl-bootcamp-setup/), and the
  two had already drifted apart. The three pages are replaced by a single
  [Setup](setup.md) pointer carrying the quick-start commands, the notebook
  locations and the two problems that really are ours (`No module named 'envs'`,
  the docs toolchain). Everything else links to the primer, which is
  year-agnostic, public and permanent.
- **`environment.yml` is now byte-identical to the primer's**, so cloning either
  repository produces the same environment. The MkDocs toolchain moved out of it
  into `requirements-docs.txt` — participants never needed it.
- The **AirTraffic and Crippled Ant quick-starts** pointed at per-folder
  `requirements.txt` files. Both were subsets of `environment.yml`, and
  AirTraffic's `gymnasium>=0.29` would have installed a pre-1.0 API. Removed;
  both READMEs now say `conda activate rlbootcamp`.
- **The navigation is reorganised around three questions**, replacing a single
  *Tutorial Guides* tab holding a flat list of eight pages in no particular
  order. **Environments** — what each MDP is (maze, Crippled Ant, AirTraffic).
  **Running** — the commands to train, evaluate, render and sweep. **Sessions** —
  what happens in the room. Within each tab, pages follow session order.

  The two usage guides each described an environment *and* how to drive it, so
  each was split at its natural seam: `airtraffic-usage.md` → the environment
  page plus separate training and evaluation pages; `crippled-ant-usage.md` →
  environment plus running. The maze MDP was extracted out of the Fundamentals
  page so all three environments can be read side by side.

### Removed
- **`RL_bootcamp_2026_code_setup_information.pdf`** and its copy inside
  `Tutorial Hands-On Session: Tabular RL & Discrete MDPs/`. Identical text, two
  different checksums, and both described a `venv` + `requirements.txt` install
  that no longer exists — plus a Colab flow that generates an SSH key and clones
  a private repository, which cannot work for a participant. The Fundamentals
  README and guide now link to the primer.

### Added
- `validation: anchors: warn` in `mkdocs.yml`. Now that every setup link leaves
  this site, a renamed heading in the primer should break the build rather than
  quietly rot.

**Open decision — reward restructuring.** The `LegacyFlightEnv` survival bonus is
paid as a terminal lump at `max_steps`, but the episode *also* truncates once the
clearance budget is spent, which early policies burn in 13–17 steps. The
dominant reward term is therefore unreachable for most of training. Paying it
per step would fix this — but it changes the environment and invalidates every
reference number published in 0.5.0, so it needs deciding before the solutions
bundle is distributed.

## [0.5.0] — 2026-08-06

### Added
- **A reference solution for AirTraffic exists at last.** The Intermediate
  syllabus has always promised *"PPO, pre-trained checkpoint provided"* while no
  checkpoint existed and `**/runs/` is gitignored — there was no distribution
  mechanism at all. There is now a **solutions bundle** (checkpoints, a
  standalone evaluation script, a renderer, pre-rendered episodes and a written
  analysis) distributed from the event website rather than from this repository,
  since the run takes ~80 minutes.
- **Track F — Policy Architecture** in the Advanced syllabus, plus a summary on
  the [Advanced session](advanced/index.md) page: a shared per-aircraft encoder
  with symmetric pooling and factored *aircraft → clearance* heads, masking
  inside the policy. Worth **+50 return** over the flat `MlpPolicy` and it drives
  invalid clearances from 32.6 per episode to **zero**.
- **Module 5.5 — Field Notes: What Actually Breaks**, six lessons carried over
  from a production RL prototype into the Advanced syllabus.
- **Measured `n=5, m=5` results** on the
  [Rule-based baseline](advanced/baseline.md) page, covering both
  trained agents, both baselines and a **random control**, with the
  congestion-per-step normalisation and the reason the random row belongs in
  every table.

### Changed
- **Intermediate syllabus audience clarified**: Session #2 is for *RL users*,
  Session #3 for *RL practitioners who design the MDP*. Dead material removed
  (an inline `CrippledAnt` sketch superseded by the real `envs/crippled_ant.py`,
  a cut DDPG module, a resolved dependency question).
- Exercise 6.1/6.2 now set expectations honestly: with the shipped
  configuration, participants will produce a failing agent. That is defensible
  as a deliberate lesson — but it is now *stated*, and `ep_len_mean` is handed
  over as the diagnostic, so nobody concludes they broke something.

### Fixed
- `Ant-v5`'s observation was documented as **27-dimensional**; it is **105**.
- The [Advanced session](advanced/index.md) page still described the rule-based
  baseline controller as *"in progress"*. It landed several versions ago and is
  the best-performing agent in the table.
- The same page presented `MaskablePPO` without its central limitation: it
  consumes a **flat** mask and cannot express a mask conditioned on a
  sampled part of the action. Anyone following it into a factored action space
  hit a wall with no warning.

## [0.4.0] — 2026-08-04

### Added
- **Session #1 (Fundamentals) is now part of this handbook.** New
  [Fundamentals guide](fundamentals/index.md) covering the maze notebooks,
  their MDP formulation, how to run them locally rather than in Colab, and what
  carries forward into Session #2. The session was previously invisible here —
  `index.md` listed it as *"owned by the Fundamentals team"* with no link.
- **Notebook check** in the installation guide, and a notebook section in
  `conda-environment.md` listing which notebooks belong to which session.

### Fixed
- **`environment.yml` shipped no Jupyter at all**, while Session #1 is delivered
  *entirely* as notebooks and `conda-environment.md` claimed JupyterLab was
  included. Anyone building the environment from this file and following the
  docs hit `jupyter: command not found`; the documented workaround was an
  ad-hoc `pip install jupyterlab ipykernel`. Added `jupyterlab`, `notebook` and
  `ipykernel` as proper conda dependencies. `notebook` is included alongside
  `jupyterlab` so both `jupyter lab` and `jupyter notebook` work — under
  Notebook 7 they share one stack.
- The `pip`-only fallback install in `installation.md` omitted Jupyter *and*
  `gymnasium[classic_control]`, so it produced an environment that could not run
  Session #1 at all and could not render classic-control environments.
- `index.md` still advertised **docs version `0.0.1`** while `VERSION` read
  `0.3.2`.

### Known issues (not yet fixed — see the volunteer task sheet)
- The Session #1 notebooks open with five Colab-only cells
  (`drive.mount`, `ssh-keygen`, `git clone`, `!pip install`). Run locally,
  `from google.colab import drive` raises `ModuleNotFoundError` on the first
  cell. **Verified:** with those cells stripped, all five notebooks execute
  cleanly end-to-end in `rlbootcamp`. The guide documents the workaround; the
  notebooks themselves still need fixing.
- `Solutions/` contains only *Simple maze solutions.ipynb*; the folder's own
  README advertises four.

## [0.3.2] — 2026-07-23

### Fixed
- **`pygame` was missing from `environment.yml`.** Rendering any classic-control
  environment (CartPole, Pendulum, MountainCar) raised
  `DependencyNotInstalled: pygame is not installed` — including when *recording
  video*, not only when opening an on-screen window. Training is unaffected, so
  the failure surfaces late and looks unrelated to setup. Added via
  `gymnasium[classic_control]`.

  No Session #2 or #3 code path hits this (AirTraffic renders through
  Matplotlib, Crippled Ant through MuJoCo), so this is preventive: participants
  experimenting with the standard SB3/Gymnasium starter examples would have hit
  it. Found while building the public participant primer.
- `mkdocs.yml`'s `extra.version` default had drifted to `0.0.3` while `VERSION`
  read `0.3.1`, so the version chip in the docs header showed the wrong number
  on any build without `DOCS_VERSION` set. Both now read `0.3.2`.

## [0.3.1] — 2026-07-16

### Fixed
- **VecNormalize statistics were mispaired with mid-training checkpoints.**
  SB3's `EvalCallback` saves `best_model.zip` but not the normalisation
  statistics behind its score, so loading it with the end-of-training
  `vecnormalize.pkl` fed the policy differently-scaled observations than it
  was evaluated with — and `evaluate.py` / `transfer_benchmark.py` both
  default to `--model best`. Measured on the 3M-step Ant: **2570 ± 55** with
  matched statistics vs **1964 ± 738** (worst episode 286) when mispaired —
  no crash, just quietly worse and wildly inconsistent.
    - `train.py` now saves `best_model/vecnormalize.pkl` on every new best
      (`SaveVecNormalize` callback on `callback_on_new_best`).
    - `utils.loading.find_vecnormalize` resolves the statistics belonging to a
      given checkpoint (periodic snapshot → best-model snapshot → run-level
      fallback); all three scripts use it and warn when only a fallback exists
      (i.e. for runs trained before this fix).
    - Four pairing tests; the gotcha is written up in the
      [usage guide](intermediate/running.md).

## [0.3.0] — 2026-07-16

### Added
- **Rule-based AirTraffic baseline** (`Airtraffic/agents/`) — the litmus-test
  heuristic for Ex 6.1: priority-based altitude-slot assignment with
  just-in-time clearance issuance and outbound freezing. Works from the
  observation only, via `decode_observation`; exposes the SB3
  `predict()` interface. Plus a `NoopController` (the do-nothing floor) and
  `scripts/evaluate_baseline.py`, which compares any mix of heuristics and
  checkpoints on identical seeds with safety KPIs (congestion-steps,
  clearances, invalids) alongside return. Beats do-nothing on return *and*
  safety at n=5 and n=8 — see the new
  [Rule-based baseline guide](advanced/baseline.md) for how to
  approach the environment programmatically and the reference numbers.
- 10 baseline contract tests (decode round-trip, no invalid clearances,
  deconflicts and out-scores noop, stateless across resets).

## [0.2.0] — 2026-07-16

### Added
- **`scripts/render_agent.py`** (Crippled Ant) — point it at *any* checkpoint
  `.zip` (best/final/periodic); it walks up to the run's saved Hydra config,
  rebuilds the training environment with the matching VecNormalize statistics
  (periodic checkpoints use their own stats snapshot), and records `.mp4`
  episodes. Injury overrides (`--disabled-legs/-joints`, `--n-random-legs`)
  render a healthy-trained policy on an injured Ant — transferability and
  robustness by eye. `--stochastic`, `--episodes`, `--seed` supported.
- Video filenames encode checkpoint + injury + mode; the auto-reset stub
  video after the last episode is no longer written.

## [0.1.1] — 2026-07-16

### Fixed
- **Dead syllabus links.** The session guides and the syllabus pages' source
  banners linked to `github.com/.../blob/main/tutorial/...`, which 404s (the
  files live on the `tutorial-code` branch). The guides now link to the
  handbook's own Intermediate and Advanced syllabus pages — the full detailed
  module breakdowns, embedded verbatim from `tutorial/TUTORIAL_SYLLABUS_*.md`.
  (Both pages and both source files were removed in 0.32.0; the links here are
  de-linked rather than rewritten, because a changelog records what happened.)
  Same fix for the `environment.yml` link on the installation page.

## [0.1.0] — 2026-07-15

### Added
- **Crippled Ant package** (`Crippled Ant/`) — the Session #2 reality-gap
  toolkit: `CrippledAnt` wrapper (fixed joints, whole legs, or randomised legs
  per reset), Hydra-configured `scripts/train.py` (PPO/SAC via `algo=`),
  `scripts/evaluate.py` (injury overrides, stochastic mode, video), and
  `scripts/transfer_benchmark.py` (healthy vs. 1-leg vs. 2-leg table + chart).
  Every syllabus exercise 3.1–5.1 is a one-line command — see the
  [usage guide](intermediate/running.md).
- **Test suites** in both session packages (`pytest`): wrapper/config contract
  tests and end-to-end training smoke tests for Crippled Ant, and an
  **environment contract suite** for AirTraffic that student modifications
  (Advanced session, competition submissions) must keep green.
- `pytest` and `moviepy` added to `environment.yml`.

### Fixed
- **`LegacyFlightEnv.reset(seed=...)` is now actually reproducible** — the seed is
  threaded through to the simulator's RNG (previously the simulator drew from
  the global `random`/`np.random` state and ignored the seed). Foundation for
  the competition's seeded validation set.
- **`LegacyFlightEnv` observation space claimed `Box(-1, 1)` but produced values
  outside it** — flights spawn outside the airspace, so positions legitimately
  exceed ±1 and `gymnasium.utils.env_checker.check_env` failed. Bounds are now
  per-dimension and honest ([details](reference/environment-api.md)).

### Removed
- `Crippled Ant/post_training_analysis.py` — imported `src.*` from the 2025
  repo and could not run here; superseded by `scripts/evaluate.py`. The stale
  pip-freeze `requirements.txt` was replaced by a curated one.

## [0.0.3] — 2026-06-05

### Added
- Full **Intermediate** and **Advanced syllabi** as dedicated docs pages under
  Tutorial Guides. They embed `tutorial/TUTORIAL_SYLLABUS_*.md` verbatim via the
  include-markdown plugin, so those files remain the single source of truth.

## [0.0.2] — 2026-06-05

### Changed
- Header wordmark is now solid white instead of the blue→green gradient — the
  green half was low-contrast against the dark header.

## [0.0.1] — 2026-06-05

Initial documented release.

### Added
- **MkDocs Material documentation site** (`mkdocs.yml`, `docs/`) — wiki-style HTML
  handbook, dark/light brand theme matching the bootcamp website.
- Brand stylesheet `docs/stylesheets/extra.css` and `docs/assets/` slot for
  logo/favicon — single place to change colours, logo and icons.
- **Setup pages:** multi-OS installation (Linux/macOS/Windows), Conda-environment
  workflow, and an OS-by-OS troubleshooting guide.
- **AirTraffic / `LegacyFlightEnv` usage guide** — layout, import patterns, training,
  evaluation/rendering, configuration, simulator exploration, recipes.
- **Environment API reference** for `LegacyFlightEnv`, `CollisionCourseSimulator`,
  `Flight`.
- Intermediate and Advanced session orientation pages.
- Root `environment.yml` defining the `rlbootcamp` Conda environment (verified
  against the installed env: gymnasium 1.2, SB3 2.7, MuJoCo 3.3, torch 2.8).
- Root `VERSION` file (`0.0.1`).

### Verified
- AirTraffic package imports and steps under gymnasium 1.2.0 / SB3 2.7.0 in the
  `rlbootcamp` env (obs shape 71 for `n=5, m=2`; `gym.make("LegacyFlightEnv-v0")` OK).
