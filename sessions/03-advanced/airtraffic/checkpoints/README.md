# Published checkpoint — Session #3

`ppo_4M_curriculum.zip` is the reference agent the deck quotes. Tracked
deliberately (`*.zip` is gitignored repo-wide; excepted in `.gitignore`) so the
board is reproducible from a fresh clone.

Trained with:

```bash
OMP_NUM_THREADS=1 python scripts/train_4d.py \
    --steps 4000000 --out runs/v4d_curriculum_autoent \
    --seed 123 --n-envs 8 --target-kl 0.05 \
    --curriculum --curriculum-frac 0.2 \
    --auto-entropy --auto-entropy-target 0.35 \
    --checkpoint-every 500000
```

~80 minutes on 16 CPU cores at ~1000 fps. Score it:

```bash
python scripts/score_4d.py --agent noop --agent random --agent rule-based \
    --agent checkpoints/ppo_4M_curriculum.zip
```

On the board's 100 held-out seeds it fails **1**, against the rule-based
controller's 14, with `exit_miss` 0.087 against 0.554 — safer *and* more
punctual. It loses the efficiency columns: 3.46 conflict-steps per 100 steps
flown against 2.01, and 2278 clearances against 667.

**One hundred seeds does not pin a failure rate.** The same checkpoint fails
roughly 3.5% of a further 600 held-out seeds. The board row is reproducible; the
*rate* needs more seeds than a board has. Do not quote `failed 1` as though the
agent solves the problem.

No `VecNormalize` here — `Flight4DEnv` emits observations already scaled into
`[-1, 1]`, so the checkpoint stands alone.
