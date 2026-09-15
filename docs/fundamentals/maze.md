---
title: Maze gridworld
---

# Maze gridworld — the environment

A textbook finite MDP: an agent on a grid, holes to avoid, a target to reach.
It is the first environment of the track precisely because you can enumerate
every state and write the value function down by hand.

Unlike the other two environments, this one is not a package — each notebook in
`sessions/01-fundamentals/environments/` defines
its own environment class and registers it with Gymnasium. To work through it,
see [Session #1 — Fundamentals](index.md).

---

## Four mazes, increasing difficulty

| Notebook | Grid | Adds |
|---|---|---|
| `Simplest maze environment.ipynb` | 3×3 | — |
| `Simple maze environment.ipynb` | 5×5 | holes |
| `Complex maze environment.ipynb` | 5×5 | mountains, a moving obstacle |
| `Larger complex maze environment.ipynb` | 10×10 | scale |

The point of the progression is to watch the *same* algorithm need more
episodes as the state space and the reward structure grow.

---

## The MDP

| | |
|---|---|
| **State** | The agent's `(row, col)` on the grid, via `get_coord(obs)`. |
| **Actions** | `RIGHT`, `UP`, `LEFT`, `DOWN` — `self._action_to_direction`. |
| **Reward** | `STEP = 0`, `TARGET = +1`, `HOLE = -1`. The complex mazes add mountains (a movement cost) and a moving obstacle (a penalty). |
| **Termination** | The agent reaches the target: `np.array_equal(self._agent_location, self._target_location)`. |
| **Transitions** | Deterministic in the simple mazes; the moving obstacle makes the complex ones effectively stochastic. |

Tabular methods work here because the state space is small enough to enumerate:
a 10×10 grid is 100 states, so a Q-table is a 100×4 array.

---

## Why it stops working

The moment the state becomes continuous — a joint angle, an aircraft heading —
enumeration breaks down and the table has to be replaced by a **function
approximator**. That substitution is the whole of Session #2, and the reason the
next two environments look nothing like this one:

- [**Crippled Ant**](../intermediate/crippled-ant.md) — continuous observations and torques.
- [**AirTraffic**](../advanced/airtraffic.md) — a continuous airspace with a discrete
  clearance per aircraft.

---

## Next

- [**Session #1 — Fundamentals**](index.md) — run the
  notebooks, solve the mazes with Q-learning and SARSA.
