# Published checkpoints — Session #2

Tracked deliberately (`*.zip` is gitignored repo-wide; these are excepted in
`.gitignore`), because the exercises offer them as the alternative to an hour of
training. They are **copies** of run outputs — the run directories themselves
stay untracked.

| Directory | Training | Use |
|---|---|---|
| `ppo_healthy_3M/` | PPO, healthy `Ant-v5`, 3M steps | Exercise 3's zero-shot transfer benchmark |
| `ppo_specialist_3M/` | PPO, `env.disabled_legs=[0]`, 3M steps | Exercise 4's specialist comparison — same budget as `ppo_healthy_3M`, deliberately: comparing specialist vs. generalist at *different* budgets would confound "less training" with "the wrong objective" |
| `ppo_randomised_range_3M/` | PPO, `env.n_random_legs_max=1`, 3M steps | Exercise 5's domain-randomisation comparison — the injury *count* is sampled uniformly on `{0, 1}` every reset (never more than one leg down), so a healthy Ant is part of the training distribution. A wider draw (`n_random_legs_max=4`) was tried first and collapsed: the policy learned that standing still is near-optimal at the worst severities and generalised that to every case, scoring worse than the do-nothing floor even healthy. `n_random_legs_max=1` keeps the task inside what 3M steps can actually solve |

Each is laid out like a run directory — `final_model.zip`, `vecnormalize.pkl`
and `.hydra/config.yaml` — so the benchmark script takes it directly:

```bash
python scripts/transfer_benchmark.py \
    --run zero-shot=checkpoints/ppo_healthy_3M --model final --episodes 20
```

**You need the stats as well as the weights.** `VecNormalize`
statistics are part of the model: loading a policy without its matching stats
silently produces garbage, because the observations it sees are scaled
differently from the ones it trained on.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

venv = VecNormalize.load("checkpoints/ppo_healthy_3M/vecnormalize.pkl",
                         DummyVecEnv([make_env]))
venv.training = False        # freeze the statistics
venv.norm_reward = False     # score the real reward, not the normalised one
model = PPO.load("checkpoints/ppo_healthy_3M/final_model.zip", device="cpu")
```

Official numbers — `scripts/transfer_benchmark.py`, 20 episodes/scenario,
identical seeds across policies, from `slides/data/ant_transfer.csv`:

| policy | healthy | 1 leg dead (mean of 4) | 2 legs dead (mean of 6) |
|---|---:|---:|---:|
| `ppo_healthy_3M` (zero-shot) | **2316** | 1193 | 919 |
| `ppo_specialist_3M` (leg 0 only) | 94 | 1247 | 722 |
| `ppo_randomised_range_3M` | 787 | 1220 | **973** |

No policy wins everywhere. The zero-shot generalist wins healthy by a wide
margin and nowhere else. The specialist wins its own injury outright (3978 vs
1119 on leg 0 specifically — see `slides/figures/fig_specialist`) and
collapses everywhere else, healthy worst of all. The randomised policy is
never the best at any one thing, but it **wins 2-legs-dead outright** and
lands within 2% of the specialist's own 1-leg aggregate — without ever being
told which leg — at the cost of a healthy score below the do-nothing/"statue"
floor (~1000). That trade is Exercise 5's actual finding; do not round it off
into "the randomised one is just worse."
