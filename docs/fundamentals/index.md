---
title: Fundamentals (Session #1)
---

# Session #1 — Tabular RL & Discrete MDPs

**Owner:** Lea Keller. The material lives in
`sessions/01-fundamentals/` at the repository root.

This is the entry point to the tutorial track: a gridworld maze, solved with
**Q-learning** and **SARSA**, with no neural network anywhere. Everything you
learn here — states, actions, rewards, episodes, the exploration/exploitation
trade-off — is the vocabulary Sessions #2 and #3 assume.

!!! info "This session is delivered entirely as notebooks"
    Unlike Sessions #2 and #3 (which are scripts with notebook walkthroughs),
    Session #1 *is* the notebooks. Make sure `jupyter lab` works **before** you
    arrive — see the [notebook check](https://sarl-plus.github.io/rl-bootcamp-setup/setup/installation/#notebooks).

---

## What's in the folder

```
sessions/01-fundamentals/
├── Environments/                       ← build and look at the MDP
│   ├── Simplest maze environment.ipynb        3×3 grid
│   ├── Simple maze environment.ipynb          5×5 grid, holes
│   ├── Complex maze environment.ipynb         + mountains, moving obstacle
│   └── Larger complex maze environment.ipynb  10×10 grid
└── Solutions/                          ← solve it
    └── Simple maze solutions.ipynb            Q-learning + SARSA
```

Each *Environments* notebook is self-contained: it defines the environment class,
registers it with Gymnasium, and renders a frame so you can see the maze you are
about to solve. The *Solutions* notebooks add the learning algorithms and the
convergence plots.

---

## Running the notebooks

=== "Locally (recommended)"

    The maze environments are pure NumPy + Gymnasium and run comfortably on any
    laptop — there is no reason to be online for this session.

    ```bash
    conda activate rlbootcamp
    cd "sessions/01-fundamentals"
    jupyter lab
    ```

    Then open a notebook and pick the **Python (rlbootcamp)** kernel.

    !!! warning "Skip the first few cells"
        The notebooks open with Colab-specific setup — `drive.mount(...)`,
        `ssh-keygen`, `git clone`, `!pip install`. **None of these apply
        locally**, and `from google.colab import drive` will fail with
        `ModuleNotFoundError` if you run it. Start from the *"Useful imports"*
        cell.

=== "Google Colab"

    Upload the notebook to [Colab](https://colab.research.google.com/) and run
    the setup cells at the top. Everything the notebooks need is either
    preinstalled or installed by those cells.

    Colab is a reasonable fallback if your laptop setup is not working — but it
    is only viable for **this** session. Sessions #2 and #3 need MuJoCo and
    longer training runs, so get the local environment working regardless.

---

## The environment, in MDP terms

States, actions, rewards and terminations are set out on the
[**Maze gridworld**](maze.md) page, alongside the other two
environments so you can compare them.

The point of the progression — simplest → simple → complex → larger — is to
watch the *same* algorithm need more episodes as the state space and the reward
structure grow. Count the steps-to-target per episode and compare across mazes;
that curve is the deliverable.

---

## What to take into Session #2

Tabular methods work here because the state space is small enough to enumerate.
The moment the state becomes continuous — a joint angle, an aircraft heading —
enumeration stops working, and the table has to be replaced by a **function
approximator**. That substitution is the whole of Session #2.

Concretely, carry these forward:

- **Exploration is a choice you make.** ε-greedy here; entropy bonuses and
  stochastic policies later. Same problem, different machinery.
- **The reward function is a design decision, not a given.** You will change one
  in Session #3 and watch the agent's behaviour change with it.
- **Learning curves are noisy.** Two seeds disagree. This does not improve with
  deeper networks.

---

## Next

- [Session #2 — Intermediate](../intermediate/index.md): continuous control and the
  reality gap.
- [Session #3 — Advanced](../advanced/index.md): designing your own MDP.
