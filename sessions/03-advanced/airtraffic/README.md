# AirTraffic — `Flight4DEnv`

The Session #3 environment: `n` aircraft converging on a hotspot plus `m`
background flights, and a policy that issues speed and flight-level clearances
to keep them apart *and* on their 4D exit slots.

## Structure

```
airtraffic/
├── envs/
│   ├── flight.py          # Flight kinematics
│   ├── simulator.py       # CollisionCourseSimulator (physics engine)
│   └── flight_4d.py       # Flight4DEnv (Gymnasium interface)
├── agents/
│   └── rule_based_4d.py   # Priority4DController — the bar to clear
├── policies/
│   └── autoregressive.py  # the default policy: aircraft -> clearance, masked
├── scripts/
│   ├── train_4d.py        # PPO training (CLI)
│   ├── score_4d.py        # the board: 100 held-out seeds, ranked
│   ├── calibrate_4d.py    # reward-term calibration
│   └── callbacks_4d.py    # curriculum + auto-entropy callbacks
└── tests/                 # environment contract + env/controller tests
```

Scenarios are reproducible: `env.reset(seed=42)` always generates the same
traffic.

## Quick start

```bash
conda activate rlbootcamp
```

```python
import sys; sys.path.insert(0, '.')
from envs import Flight4DEnv
import numpy as np

env = Flight4DEnv()                 # n + m = 5, n ~ U(2, 4)
obs, _ = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(0)   # 0 idles
legal = np.flatnonzero(env.action_masks())
```

Or via `gym.make` (after importing `envs`):

```python
import envs                          # triggers gymnasium.register()
import gymnasium as gym
env = gym.make("Flight4DEnv-v0")
```

## Training

```bash
OMP_NUM_THREADS=1 python scripts/train_4d.py --steps 1500000 --out runs/mine
```

`OMP_NUM_THREADS=1` is not optional when running more than one job. Torch
otherwise opens ~46 threads per process; two such processes on 16 cores measured
**31–39 fps** against **428 fps** pinned — an 11× difference that looks exactly
like a slow environment.

The reference run adds a curriculum and entropy targeting:

```bash
OMP_NUM_THREADS=1 python scripts/train_4d.py \
    --steps 4000000 --out runs/ref --target-kl 0.05 \
    --curriculum --curriculum-frac 0.2 --auto-entropy
```

`python scripts/train_4d.py --help` documents every setting. They are all
flags — the exercise is choosing between them, not implementing them.

## Scoring

```bash
python scripts/score_4d.py --agent noop --agent random --agent rule-based \
    --agent runs/mine/final_model.zip
```

100 held-out seeds, held fixed across agents, ranked the way the competition
does: **failures first**. Both deterministic and stochastic modes are reported —
`idle_bias` makes idling the argmax at initialisation, so a deterministic
rollout of a weak policy is byte-identical to doing nothing, and three runs in
this project were written off as collapsed on exactly that evidence.

## Environment summary

| | |
|---|---|
| Observation | `Box(shape=(N_FEATS·n_flights + N_GLOBALS + n_actions,))` — 13 features per aircraft, two globals (clock, clearance budget), and the **legality mask carried inline**. The mask is in the observation rather than a side channel so that PPO recomputes log-probabilities against the same distribution it sampled from. |
| Action | `Discrete(1 + 7·n_flights)` — index 0 idles; `1 + 7·i + c` issues clearance `c` to aircraft `i`. At most one aircraft is commanded per step. |
| Clearances | `FL_DEC`, `FL_INC`, `FL_DEC2_T`, `FL_INC2_T`, `SPD_DN_T`, `SPD_UP_T`, `RESUME`. Every `_T` variant auto-reverts after `hold_steps`. |
| Reward | Potential-based shaping on separation and on 4D adherence (neither can move the optimum), plus a per-clearance transition cost, plus a crash charge covering the steps not flown. |
| Termination | horizon reached; clearance budget exhausted; mid-air collision. |

## Tests — the environment contract

```bash
pytest
```

`tests/test_env_contract.py` standardises what every submitted environment must
provide (spaces, seeding, determinism, termination, the KPI keys the scorer
reads). **Advanced session:** when you subclass or modify `Flight4DEnv`, point
`env_factory` in `tests/conftest.py` at your class — competition submissions are
only scored if the contract suite passes.

## Rendering

`Flight4DEnv` itself declares no render modes — hard masking and the flat
action space were the design problem, not visualisation. `scripts/render_4d.py`
drives the environment from the outside instead, replaying the recording
through `CollisionCourseSimulator`'s own geometry (`proximity_pairs`,
`find_congestion`, `get_alt_color`), which both this and the retired
environment share:

```bash
python scripts/render_4d.py --agent noop --agent random --agent rule-based \
    --agent checkpoints/ppo_4M_curriculum.zip --seed 900000

# the action vocabulary, one clearance per clip: two aircraft and nothing else
python scripts/render_4d.py --n-flights 2 --n-range 2 2 --seed 2 --ext gif \
    --agent noop --agent fl_inc --agent fl_inc2_t --agent spd_dn_t --agent resume
```

`--agent` also accepts the scripted demos in `DEMOS` — `split`, `macro`,
`speed`, and one entry per clearance — which replay a fixed
`(step, aircraft, clearance)` list so a clip shows exactly one decision. Their
outcomes on seed 2 are pinned by `tests/test_render_4d.py`.

Every proximity ring is dashed while the pair is still legally separated and
solid once it is not, with opacity and linewidth rising with severity, and a
fading trail so a resolved conflict does not vanish the instant the aircraft
move on — the congestion-visibility fix from the retired environment's own
renderer, carried forward rather than solved twice.
