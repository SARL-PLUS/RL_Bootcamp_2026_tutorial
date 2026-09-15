# RL Bootcamp 2026

Welcome to the official repository for the **RL Bootcamp 2026**.

## 💻 Setup

Installation is documented **once**, in the participant primer — it is
year-agnostic, public and permanent:

📖 **<https://sarl-plus.github.io/rl-bootcamp-setup/setup/installation/>**

```bash
git clone https://github.com/SARL-PLUS/rl-bootcamp-setup.git
cd rl-bootcamp-setup
conda env create -f environment.yml
conda activate rlbootcamp
python scripts/smoke_test.py          # trains, renders, writes an .mp4
```

The `environment.yml` in *this* repository is a copy of that one. Create the
environment from either — not both. Do not add setup instructions to this
repository; link to the primer instead.

## 📅 Event Details

- **Date:** September 16 - 18, 2026
- **Location:** Salzburg, Austria (Venue TBD: NAWI/Itzling)
- **Status:** Phase 1: Planning & Foundation (Feb 2026)
- **Website:** [Local Preview](./website/index.html)

## 📁 Repository Structure

- **`website/`**: Source code for the landing page (HTML/CSS).
- **`tutorial/`**: Code for the hands-on sessions (e.g., "Litmus Test").
- **`docs/planning/`**: Strategy documents.
  - [Running Notes (Google Doc)](https://docs.google.com/document/d/1-ZOAOV1fCHykHFCYFKa3GLEFL_DuG3ewRYauKvb5DzA/edit?usp=sharing) / [Local Copy](./docs/planning/running_notes.md)
  - [Project Overview](./docs/planning/bootcamp_overview.md)
  - [Timeline](./docs/planning/timeline.md)
  - [Keynote Invitations](./docs/planning/keynote_invitations.md)
  - [Budget Estimate](./docs/planning/budget_estimate.md)
  - [Funding Strategy](./docs/planning/funding_strategy.md)
  - [Sponsorship One-Pager](./docs/planning/sponsorship_one_pager.md)

## ⏳ Master Timeline

*See full detail in [docs/planning/timeline.md](./docs/planning/timeline.md)*

| Phase | Months | Key Goals |
| :--- | :--- | :--- |
| **1. Foundation** | Feb - Mar | Website, Venue Booking, Keynote Invites, Tutorial Concepts |
| **2. Content** | Apr - Jun | Sponsorships, Registration Setup, Draft Tutorial Code |
| **3. Promotion** | Jul - Aug | Marketing, Repository Testing, Catering, Logistics |
| **4. Final** | Sep 1-15 | Final Tech Check, Badges, Printing |

## 🚀 High-Priority Todos

*From `docs/planning/task.md`*

### Immediate Actions

- [ ] **Venue:** Book Nonntal/NAWI or Itzling (Olga).
- [ ] **Sponsorship:** Draft/Send Email to NVIDIA (Thomas) & Skalar Systems (Jakob).
- [ ] **Funding:** Submit application for "FFG Impact Innovation 2026".
- [ ] **Keynotes:** Send confirmation emails to Shortlist (Sutton, Barto, etc.).

### Technical / Tutorials (Leander and Lea)

- [ ] **Environment:** Create prototype for "Baseline vs. RL" ([tutorial/01_baseline_vs_rl](./tutorial/01_baseline_vs_rl/README.md)).
- [ ] **Helpers:** Recruit PhDs/Students for on-site support.

### Logistics

- [ ] **Registration:** Set up Indico page.
- [ ] **Catering:** Confirm quote from JR.
