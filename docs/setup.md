---
title: Setup
---

# Setup

!!! abstract "Installation is documented in the participant primer, not here"
    One guide, maintained in one place, valid for every edition of the bootcamp:

    **[:octicons-arrow-right-24: sarl-plus.github.io/rl-bootcamp-setup](https://sarl-plus.github.io/rl-bootcamp-setup/setup/installation/)**

    This page is a pointer. If anything here and the primer disagree, the primer
    is right.

## The short version

```bash
git clone https://github.com/SARL-PLUS/rl-bootcamp-setup.git
cd rl-bootcamp-setup
conda env create -f environment.yml
conda activate rlbootcamp
python scripts/smoke_test.py
```

The smoke test creates a Gymnasium environment, trains a short
Stable-Baselines3 run, renders a frame and writes an `.mp4`. When the last line
reads **`Everything works. You are ready for the bootcamp.`** you are ready for
all three sessions.

**Do this before you arrive.** PyTorch and MuJoCo are several GB and the venue
Wi-Fi cannot serve a full room at once.

## Where to go

| | |
|---|---|
| [Installation](https://sarl-plus.github.io/rl-bootcamp-setup/setup/installation/) | Conda, the environment, the smoke test |
| [Linux](https://sarl-plus.github.io/rl-bootcamp-setup/setup/platforms/linux/) · [macOS](https://sarl-plus.github.io/rl-bootcamp-setup/setup/platforms/macos/) · [Windows](https://sarl-plus.github.io/rl-bootcamp-setup/setup/platforms/windows/) | Platform-specific detail |
| [VS Code](https://sarl-plus.github.io/rl-bootcamp-setup/setup/editors/vscode/) · [PyCharm](https://sarl-plus.github.io/rl-bootcamp-setup/setup/editors/pycharm/) | Interpreter and notebook kernel |
| [Working in the Conda environment](https://sarl-plus.github.io/rl-bootcamp-setup/setup/conda-environment/) | Daily commands, Jupyter kernels |
| [Troubleshooting](https://sarl-plus.github.io/rl-bootcamp-setup/setup/troubleshooting/) | Every failure we have seen, by symptom |

## Getting the tutorial code

This repository is published at the start of the event — you do not need it to
get set up, and that is deliberate: you can be completely ready before the
exercises exist.

Once it is available:

```bash
git clone https://github.com/SARL-PLUS/RL_Bootcamp_2026_tutorial.git
cd RL_Bootcamp_2026_tutorial
conda activate rlbootcamp
```

The same `rlbootcamp` environment runs all three sessions. There is no second
environment to create.

!!! note "The `environment.yml` in this repository"
    It is a copy of the primer's, kept identical so that cloning either
    repository gives you the same environment. Create it from whichever one you
    have; do not create both.

### Where the notebooks are

| Session | Notebooks |
|---|---|
| **#1 Fundamentals** | `sessions/01-fundamentals/environments/` and `.../solutions/` |
| **#2 Intermediate** | `sessions/02-intermediate/crippled-ant/notebooks/` |

Session #1 is delivered **entirely** as notebooks, so confirm `jupyter lab`
works before you arrive — the
[notebook check](https://sarl-plus.github.io/rl-bootcamp-setup/setup/installation/#notebooks)
takes two minutes.

## Problems specific to this repository

Anything about Conda, Python, MuJoCo, `ffmpeg` or rendering belongs in
[the primer's troubleshooting page](https://sarl-plus.github.io/rl-bootcamp-setup/setup/troubleshooting/).
Only these are ours:

??? failure "`ModuleNotFoundError: No module named 'envs'`"
    The AirTraffic package is not on the path. Run the scripts from inside the
    `sessions/03-advanced/airtraffic/` directory — `scripts/train_4d.py` and
    `scripts/score_4d.py` already handle it from there — or add it explicitly:

    ```python
    import sys
    sys.path.insert(0, "sessions/03-advanced/airtraffic")
    from envs import Flight4DEnv
    ```

??? failure "`mkdocs serve` fails with `Unrecognised theme 'material'`"
    The docs toolchain is not part of the participant environment. Only
    organisers need it:

    ```bash
    pip install -r requirements-docs.txt
    ```
