---
title: The Flight Challenge
---

# The Flight Challenge — competition rules

The Session #3 competition. You are an air traffic controller for a sector where
aircraft converge on a hotspot. Keep them apart, keep them on schedule, and touch
the traffic as little as you can.

!!! abstract "The goal, in one sentence"
    Your agent is ranked on three objectives **in strict priority order** —
    **safety first, then timeliness, then efficiency**. A submission that is
    safer always outranks one that is faster or cheaper, no matter how much
    faster or cheaper.

## What you are ranked on

| Rank | Objective | What we measure | Better is |
|:--:|---|---|---|
| **1** | **Safety** | Loss-of-separation events — any pair of aircraft simultaneously within the horizontal hotspot radius **and** within `ALT_MIN_SEP` vertically, summed over every step of every episode | Lower — **zero is the bar** |
| **2** | **Timeliness** | Distance from each aircraft's **original 4D exit point** — the place, altitude and time it was headed for before you touched it | Lower — **zero is achievable** |
| **3** | **Efficiency** | Clearances issued, then the number of distinct aircraft that received any clearance | Lower |

Ranking is **lexicographic**: we compare safety first. Only submissions tied on
safety are compared on timeliness, and only those tied on both are compared on
efficiency. Timeliness is bucketed to a fixed resolution so that ties genuinely
occur and efficiency is not dead weight.

### Timeliness: a clearance is a detour, not a new destination

Every aircraft enters the sector with a plan: it will cross a particular point,
at a particular flight level, at a particular time. Because headings are never
commanded here, that plan is a straight line and its **4D exit point** is fixed
the moment the aircraft appears.

You may deviate from it freely — that is what clearances are for. What you may
not do is *leave* the aircraft deviated. If you slow an aircraft to open a gap,
it has to speed back up afterwards and make the time up. If you step it down a
flight level, it has to climb back. Timeliness measures how far each aircraft
still is from the plan it was flying when you found it:

```text
exit_miss = mean over aircraft of ( |time late| / one_hold_period
                                   + |altitude off plan, in flight levels| )
```

`one_hold_period` is `hold_steps x DT` — the environment's own natural time
unit, the length one temporary clearance holds before it auto-reverts. An
aircraft you never touched is at zero by construction, so untouched traffic
costs nothing. This is a **direct sum**, not log-compressed: a large deviation
costs proportionally, so there is no free tail the way a log scale would give
you.

!!! warning "This closes the hole a pure delay metric leaves open"
    There is a trivially safe policy: slow every aircraft down. Decelerating
    de-synchronises arrivals at the hotspot and conflicts simply stop happening.
    It would top a safety-only leaderboard and it is operationally worthless.

    Note what the 4D rule does and does not forbid. Slowing traffic down is
    **fine** — encouraged, even, if it resolves a conflict. What costs you is
    slowing it down *and never recovering*. The interesting consequence: aircraft
    already at maximum velocity cannot make up lost time, so **which** aircraft
    you choose to delay is now part of the problem.

We score with **our** KPI function, never your reward. You are free to shape your
reward however you like — that is the whole point of the design sprint — but
shaping it will not move the leaderboard except through the behaviour it produces.

!!! tip "Beat the brick, then beat the heuristic"
    Two reference agents ship with the harness, and you can rank yourself against
    them at any time:

    ```bash
    python -m competition.run compare --agent noop --agent random --agent rule-based
    ```

    | # | submission | failed | congestion | exit miss | clearances | touched |
    |---|---|---:|---:|---:|---:|---:|
    | 1 | rule-based | 20 | 96 | 0.5700 | 727 | 231 |
    | 2 | random | 76 | 667 | 0.7012 | 2947 | 497 |
    | 3 | noop | 100 | 752 | 0.0000 | 0 | 0 |

    `random` is uniform over the legal actions — what your action space scores
    for free, and the row nothing trained should sit below. It is seeded from
    each episode's first observation, so its numbers are reproducible.

    (100 episodes, the harness's own public seeds — a separate measurement
    from `slides/data/air_4d_board.csv`, which uses a different held-out
    block. Both are honest; they are just not the same 100 seeds.)

    The brick fails *every* episode here — it never resolves a conflict, only
    avoids causing one by never touching an aircraft — and it is still
    perfectly on time and perfectly efficient by construction. Safety is
    compared first regardless, so it loses to a controller that occasionally
    fails less often. If your agent cannot beat the do-nothing baseline on
    safety, the honest number goes on the board next to it.

### One thing the ranking deliberately refuses to reward

An episode that ends early — because a mid-air happened, or the clearance
budget ran out — racks up fewer loss-of-separation events simply by existing
for less time. So **failing to fly the sector is ranked before safety, not
folded into it**: `failed_episodes` sits first in the sort key precisely so a
policy cannot buy a better congestion number by dying young. (Invalid actions
cannot end an episode here — the mask makes them structurally unreachable for
a policy that reads it; they are counted, not punished, so a broken submission
is loud rather than silently truncated.)

## What you submit

A single zip. At minimum it contains your trained model; in practice a successful
attempt will have changed code too, and we need every changed file to reproduce
your result.

```
submission.zip
├── submission.yaml      # required — metadata, see below
├── model/
│   └── policy.zip       # required, unless policy_entry_point is set — your SB3 checkpoint
├── envs/                # your environment package, if you changed it
└── policy/              # your policy class, if policy_entry_point points here
```

!!! info "Where it goes, and when"
    This zip is **not** uploaded to Kaggle — Kaggle only ever sees
    `submission.csv` (below). Drop the zip in the shared
    [Drive folder](https://drive.google.com/drive/folders/1Mrv5aZ70LDN5goMj0k67XFGK84LZtBOI?usp=sharing),
    named `<team-name>.zip`, once — this is a one-time delivery at the close of
    the design sprint, not a running submission like the public board. Re-upload
    to the same filename if you revise before the deadline; only the last one
    before the deadline is scored.

!!! abstract "Kaggle is how you submit. The zip is how we check."
    The public board is the submission channel, all session. **If the top of
    the board looks close, we will ask the teams involved for their code and
    zips** and score them ourselves on seeds nobody has seen — so keep the zip
    valid all afternoon (the dry run below takes a minute), not something you
    assemble in the last ten.

    We score every zip ourselves, offline, one at a time in its own process —
    not because Kaggle couldn't run a metric, but because your zip can carry
    your *own* environment code, and a shared Kaggle sandbox has no way to
    keep twenty different `envs/` packages from colliding. Kaggle is the live
    scoreboard; this is the actual grading run.

```yaml title="submission.yaml"
team: brick-with-wings
algo: PPO                            # recorded, not acted on
env_entry_point: envs.my_env:MyFlightEnv   # omit to use the stock Flight4DEnv
policy_entry_point: policy.my_policy:MyPolicy  # omit to load model/policy.zip via SB3
deterministic: true                  # false = sampled actions
notes: "Relative observations + potential-based shaping"
```

`team` is the only required key. `env_entry_point` and `policy_entry_point` are
how we find your classes — there is no `inference.py` and no loader for you to
write, because the manifest says everything the harness needs.

!!! tip "Not training an SB3 policy? `policy_entry_point` is for you"
    "Network architecture, training regime" are yours to choose — a hand-tuned
    controller is a legitimate entry, not just an SB3 checkpoint. Point
    `policy_entry_point` at a class with SB3's `predict(obs, deterministic)`
    interface and skip `model/policy.zip` entirely; it is instantiated with
    **no arguments**, so read the frozen scenario yourself from
    `competition.config.SCENARIO` if your class needs it. Two worked examples —
    one shipping the reference PPO checkpoint, one shipping a hand-written
    controller through `policy_entry_point` — are in the
    [Drive folder](https://drive.google.com/drive/folders/1Mrv5aZ70LDN5goMj0k67XFGK84LZtBOI?usp=sharing).

!!! warning "If you ship `envs/`, ship the **whole** package"
    Your `envs/` directory replaces ours on the import path. So a file in it that
    does `from envs.flight_4d import Flight4DEnv` will not find our copy — it
    will look inside *your* package and fail.

    Copy the whole directory (`cp -r airtraffic/envs .`), add your module beside
    the others, and import with a **relative** import:

    ```python title="envs/my_env.py"
    from .flight_4d import Flight4DEnv

    class MyFlightEnv(Flight4DEnv):
        ...
    ```

    This is the single most likely way for a submission to be rejected, and it
    is entirely avoidable. Test it before you hand it in — see below.

!!! note "`vecnormalize.pkl` is not applied"
    `Flight4DEnv` already emits observations scaled into `[-1, 1]`, so the
    harness does not wrap the scored environment and the reference checkpoint
    does not use `VecNormalize`. If you ship one, the harness prints a warning
    rather than silently ignoring it — tell us and we will score you by hand.

## How your submission is scored

1. We unzip it and read `submission.yaml`.
2. If you named an `env_entry_point`, it must pass the **contract test
   suite** — `sessions/03-advanced/airtraffic/tests/test_env_contract.py`.
   Red means unscored. No exceptions, no manual fixes on our side.
3. We run **100 episodes on a frozen, secret seed list**, using `reset(seed=...)`
   so every submission faces exactly the same traffic.
4. We compute safety, timeliness and efficiency from the resulting trajectories
   and rank you lexicographically.

!!! abstract "Two environments, and only one of them is yours"
    Your environment and ours run **in lockstep on the same seed**. Yours
    produces the observations your policy was trained to read. Ours is stepped
    with the same action, and **every KPI is read from ours** — the harness
    loads its own copy of `envs` by file path, so the `envs/` you ship cannot
    stand in for it whatever it is called.

    Nothing crosses that boundary except the action index. Your reward, your
    termination shaping and your info dict are never consulted.

    The useful consequence: you cannot lose points by getting your environment
    subtly wrong in a way we failed to notice, and you cannot gain any either. A
    wrong environment just feeds your policy misleading observations, and shows
    up as a worse score honestly earned. If your env gives up early, ours keeps
    flying the sector — ending early is how `failed_episodes` is earned, not a
    way to stop accumulating congestion.

### Build the zip

Everything the harness needs is a directory with `submission.yaml` at its top
level. From `sessions/03-advanced`:

```bash
mkdir -p my-team/model
cp airtraffic/runs/mine/final_model.zip my-team/model/policy.zip   # your checkpoint
cp -r airtraffic/envs my-team/envs                                  # ONLY if you changed it — the whole package
cat > my-team/submission.yaml <<'EOF'
team: my-team
algo: PPO
env_entry_point: envs.flight_4d:Flight4DEnv     # drop this line if envs/ is not in the zip
deterministic: true
notes: "what you changed, in one line"
EOF
(cd my-team && zip -r ../my-team.zip .)
```

If you edited `envs/flight_4d.py` in place — which is what the sprint expects —
`env_entry_point` is `envs.flight_4d:Flight4DEnv`, your modified copy. If you
subclassed into `envs/my_env.py`, name that class instead. A hand-written
controller goes in `policy/` with `policy_entry_point` (below) and needs no
`model/`.

!!! tip "Run the gate yourself before you hand it in"
    You will not see your score on the private seeds until the ceremony — but
    you do not have to fly blind on *validity*. Zip your submission and run the
    exact check the organiser runs, on your own public-seed numbers:

    ```bash
    cd sessions/03-advanced
    python -m competition.ingest --zip your-team.zip --contract
    ```

    A green run means it *will* be scored, and prints your public-seed KPIs so
    you know it behaves as you expect before you upload it once and move on.
    `--limit 10` makes it a ten-second smoke test; `--workdir DIR` keeps the
    unpacked files where you can look at them.

    **Changed the environment? This is also your route to Kaggle.**
    `competition.run submit` rolls a checkpoint out on the *stock*
    environment, so a policy trained on observations of your own would be
    fed the wrong vector. Add `--out`:

    ```bash
    python -m competition.ingest --zip my-team.zip --contract --out submission.csv
    ```

    That runs your zip exactly as the private board does — your environment
    observed, ours scored, in lockstep — on the public seeds, and writes the
    action trace Kaggle takes. Then `replay` and upload as usual.

    Just the contract suite, faster, if you only changed the environment:

    ```bash
    cd sessions/03-advanced/airtraffic
    RLB_ENV_ENTRY_POINT=envs.my_env:MyFlightEnv \
        python -m pytest tests/test_env_contract.py -q
    ```

    That is the *same* suite, invoked the *same* way the harness invokes it.
    Twenty-four tests, a fraction of a second. There is no reason to discover a
    red one at the prizegiving.

!!! danger "What you may not change"
    The **flight physics are frozen.** `CollisionCourseSimulator` — how aircraft
    move, how the scenario is generated from a seed, what counts as a loss of
    separation — is ours and is identical for everyone. A contract test enforces
    it: the same seed and the same clearances must reproduce the reference
    trajectories exactly.

    Everything else is yours: observation space, action space, reward function,
    termination shaping, wrappers, network architecture, action masking,
    training regime.

    This is not us being precious. If your sector has different weather than
    your neighbour's, the two of you are not solving the same problem and the
    ranking between you means nothing. You may absolutely add wind, sensor
    noise or heading drift **during training** — domain randomisation is a
    legitimate and often excellent robustness strategy. It just does not
    travel to the scored run.

## Three numbers, and only one decides the trophies

You will see three scores over the session. They are not the same measurement,
and it is worth knowing which is which before you start reading too much into
any of them.

| | What it scores | When you see it | Secret? |
|---|---|---|:--:|
| **Kaggle public** | 60 of the 100 **published** seeds | live, every submission | no |
| **Kaggle private** | the other 40 **published** seeds | when the competition closes | **no** |
| **Ours** | 100 seeds you have never seen | at the ceremony | **yes** |

=== "Kaggle — live all session"

    Build `submission.csv` with the provided harness:

    ```bash
    cd sessions/03-advanced
    python -m competition.run submit --agent path/to/policy.zip --out submission.csv
    ```

    It rolls your agent out on the **published** seed list and records the action
    it chose at every step. Kaggle replays those actions through the frozen
    simulator and scores the result. You can check the score yourself first:

    ```bash
    python -m competition.run replay --submission submission.csv
    ```

    This is honest scoring, not self-reporting: bad actions produce bad KPIs on
    replay, and there is no number in the file for you to inflate. Submit as
    often as you like.

    Kaggle holds 40% of those seeds back and reveals them at the close, so your
    Kaggle rank moves once at the end. That is a **chronological** holdout, not
    an informational one — see the note below.

=== "Ours — at the ceremony"

    Your zip, our machine, **100 secret seeds you have never trained on**. This
    is the ranking that decides the trophies, and you will not see it until it
    is revealed.

!!! info "Why Kaggle's hidden half is not actually secret"
    Every seed Kaggle scores is in `competition/seeds_public.txt`, committed in
    this repository. Nothing stops you scoring all 100 locally right now, and
    that is fine.

    It could not be otherwise. A submission here is an **action trace keyed by
    seed** — to produce the rows for a seed you have to roll out on it, and to
    roll out on it you have to know it. A genuinely secret seed is one you
    cannot submit for. So Kaggle's split hides its half from *view* until the
    close; it does not hide it from *you*.

    That is exactly why the ranking that matters is your zip on our machine. We
    run the policy ourselves, so the seeds never have to leave our side.

!!! tip "Expect all three to disagree"
    The published seeds are published, so you can tune against them — and you
    will, and that is fine. The gap between your Kaggle score and your score at
    the ceremony is the most useful number you will take home from this session.
    It is the difference between a policy that solved the sector and one that
    memorised a hundred scenarios.

## Beyond the top line

Every submitted policy also goes through a **post-mortem** harness, and several
trophies are decided there rather than on rank:

- **Do-nothing baseline** — all-NOOP on the same seeds. Any agent that scores
  below it gets the honest number printed next to it.
- **Next-best action** — replace each chosen action with the second-highest
  probability one. How much of your score rests on razor-thin argmax margins?
- **ε-random action** — perturb actions with probability ε ∈ {0.05, 0.1, 0.25}.
  An action-robustness curve.
- **Best-extraction** — k stochastic rollouts per seed, keeping the best. How
  good is your policy when we actively try to extract its best behaviour?

!!! warning "Noise can flatter you --- read failures alongside congestion"
    Perturbing the **do-nothing** baseline moves both numbers, and by less than
    intuition suggests: at ε=0.25 (100 episodes) congestion falls from 752 to
    670 (~11%) and failures from 100 to 84. Random clearances occasionally
    scatter a pair that would otherwise have collided — a real effect, just a
    small one, and 84 failures out of 100 is nowhere close to what an actual
    policy achieves.

    So "my agent got safer when I added noise" is not, by itself, evidence of
    robustness. Check what happened to **failures**, not only to congestion —
    a large drop in one without the other is the flattering half of the
    picture.

The point, delivered with the trophies: **RL is a constant experiment in
controlling a complex dynamic environment.** A single leaderboard number is the
beginning of the evaluation, not the end.

## Related

- [The design sprint](design-sprint.md) — the design tracks that feed this.
- [The Kaggle leaderboard](kaggle.md) — building and uploading a submission.
- [Organiser runbook](organisers.md) — how we actually run the scoring.
- [AirTraffic environment](airtraffic.md) — what you are controlling.
- [Training](training.md) · [Evaluation](evaluation.md)
- [Rule-based baseline](baseline.md) — beat this before you celebrate.
