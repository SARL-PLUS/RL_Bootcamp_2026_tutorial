---
title: Home
---

# RL Bootcamp 2026 — Tutorial Handbook

Welcome! This is the **student-facing handbook** for the hands-on tutorial track of
the Reinforcement Learning Bootcamp 2026.

!!! info "At a glance"
    - **When:** September 16–18, 2026
    - **Where:** Salzburg, Austria
    - **Docs version:** see the [changelog](changelog.md) *(pre-release — pinned once the code is tested)*
    - **You need:** a laptop, ~5 GB free disk, and about 30 minutes for setup

---

## Do this first

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Install everything**

    ---

    Conda, the `rlbootcamp` environment and the smoke test. **Before you
    arrive** — venue Wi-Fi will not carry everyone downloading PyTorch at once.

    [:octicons-arrow-right-24: Setup](setup.md)

-   :material-bug-check:{ .lg .middle } **Something broke?**

    ---

    The fixes for what people actually hit: MuJoCo rendering, `ffmpeg`,
    `sb3-contrib`, Apple Silicon, Windows long paths.

    [:octicons-arrow-right-24: Troubleshooting](https://sarl-plus.github.io/rl-bootcamp-setup/setup/troubleshooting/)

</div>

---

## The three sessions

Each session has **its own tab**, and each tab is self-contained: what the
environment is, how to run it, and what happens in the room. **The colour is the
difficulty** — it follows you into the tab, so you always know where you are.

<div class="grid cards rlb-map" markdown>

-   <span class="rlb-level rlb-f">Session #1 · beginner</span>

    ### :material-grid: Fundamentals

    ---

    **Tabular RL on a maze gridworld.** Q-learning and SARSA, states and
    actions you can count, and the point where a Q-table stops working.

    *Runs entirely in Jupyter notebooks.*

    [:octicons-arrow-right-24: Start Fundamentals](fundamentals/index.md)

-   <span class="rlb-level rlb-i">Session #2 · intermediate</span>

    ### :material-robot: Intermediate

    ---

    **Continuous control and the reality gap.** PPO and SAC on a MuJoCo Ant,
    then break a leg and measure what the policy actually learned.

    *Environment given. The question is how well you wield it.*

    [:octicons-arrow-right-24: Start Intermediate](intermediate/index.md)

-   <span class="rlb-level rlb-a">Session #3 · advanced</span>

    ### :material-airplane: Advanced

    ---

    **Design the MDP yourself.** An air-traffic sector, a rule-based bar to
    beat, and a competition. You change the observation, the actions and the
    reward.

    *Bring a policy. Leave with evidence it works.*

    [:octicons-arrow-right-24: Start Advanced](advanced/index.md)

</div>

!!! tip "Not sure where to start?"
    Start at your **lowest unfamiliar colour**. Session #2 assumes you can
    explain what Q-learning does; Session #3 assumes you have trained something
    with Stable Baselines3 and read its curves. Nothing stops you reading ahead
    — the tabs are independent.

---

## Understanding the codebase

Everything is in one repository, laid out to match the tabs above.

| Path | What lives there | Tab |
|---|---|---|
| `sessions/01-fundamentals/` | Maze notebooks — environments and worked solutions | Fundamentals |
| `sessions/02-intermediate/crippled-ant/` | The Ant package: `envs/`, `scripts/`, Hydra `conf/`, pre-trained `checkpoints/` | Intermediate |
| `sessions/03-advanced/airtraffic/` | `Flight4DEnv`: `envs/`, `agents/`, `policies/`, `scripts/`, `tests/` | Advanced |
| `sessions/03-advanced/competition/` | The scoring harness — KPIs, seeds, submission format | Advanced |
| `docs/` | This handbook | — |

**Three things worth knowing before you read any of it:**

- **The tests are the specification.** Each package ships a `tests/` directory that states what must stay true. `pytest` from the package directory is the fastest way to find out whether you broke something.
- **Session #2 and #3 are separate packages,** each imported by path rather than
  installed. Run commands from inside the package directory, not the repo root.

---

## Reference

Session-independent detail, kept out of the session tabs so they stay readable:
the exact observation and action spaces, the reward maths, and the rule-based
controller every submission is measured against.

[:octicons-arrow-right-24: Reference](reference/environment-api.md)

!!! tip "Found a problem in the docs?"
    The documentation is treated as a first-class deliverable — if something is
    wrong, unclear, or out of date, tell an organiser or open an issue. Every
    code change ships with a matching docs update and a version bump (see the
    [Changelog](changelog.md)).
