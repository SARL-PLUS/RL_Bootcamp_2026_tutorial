# Tutorial on Tabular RL & Discrete MDPs

## Structure

```
sessions/01-fundamentals/
├── Environments    # Different environments
│   ├── Simplest maze environment.ipynb
│   ├── Simple maze environment.ipynb
│   ├── Complex maze environment.ipynb
│   ├── Larger complex maze environment.ipynb
└── Solutions       # Different Tabular RL methods used to solve the the optimal path to follow
    ├── Simplest maze solutions.ipynb
    ├── Simple maze solutions.ipynb 
    ├── Complex maze solutions.ipynb
    ├── Larger complex maze solutions.ipynb
```

> **Setup:** install once, from the participant primer —
> <https://sarl-plus.github.io/rl-bootcamp-setup/setup/installation/>.
> Activate `rlbootcamp`, run `jupyter lab`, and pick the **Python (rlbootcamp)**
> kernel.

## Summary

In tabular Reinforcement Learning, instead of requiring the environment's probability distributions, agents learn optimal policies through trial and error.

The states are defined by the agent steps at each episode (one episode  is defined until the agent reach the target). 

This basic tutorial introduces two tabular RL algorithms, namely
- Q-learning, implemented with the Bellman optimal equation
- State-Action-Reward-State-Action, implemented with the Bellman expectation equation

 <!---
| | |
|---|---|
| State | `get_coord(obs)`  |
| Action | init with `self._action_to_direction` - RIGHT/UP/LEFT/DOWN and enumerated in the environment, `choose_action(state, q_table_algorithm, epsilon, env)` in the algorithm |
| Reward (simple maze) | success > move to next step, failure > fall in a hole |
| Reward (complex maze) | success > move to next step, failure > fall in a hole, hit_moving_obstacle > continue with penalty, on_mountain > continue with cost |
| Termination | `np.array_equal(self._agent_location, self._target_location)` the agent reaches the target |

- Learning goals
- Introduction: definition of states, actions, and rewards, MDP formulation

States = {(i, j) : where i or j is not a hole}
Actions = {(up, down, left, right, stay)} (defined both as enum and inside the algorithms)
Rewards = {(step, target, hole)} or Rewards = {(step, target, hole, mountain)} defined both as enum and inside the algorithms
Transition probabilities = ...
--->

**The tutorial workflow**
1. <a href="https://sarl-plus.github.io/rl-bootcamp-setup/setup/installation/"> Installation & setup </a>
2. <a href="environments/"> Run the 3 different environment (simple, complex and larger grid) </a>
3. <a href="solutions/"> Run Q-learning and SARSA algorithms with the different environments </a>

## Environments
The environments are created from the one of <a href="https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/">Gymnasium</a>. 
### Simplest maze
This simplest environment is used to test the Reinforcement learning algorithms on a 3x3 grid with a target at the bottom right hand corner and initial state at the top left hand corner. The goal is to reach the target from the initial state with the least steps as possible using tabular Reinforcement Learning algorithms introduced in the solutions part of this tutorial. 
### Simple maze
The simple environment is slightly larger than the simplest one with a size of 5x5.
### Complex maze
In the complex maze of size 5x5, the rewards are affected differently depending on the maze terrain and obstacles. 
### Larger complex maze
This maze is similar to the complex maze in terms of obstacles involved, but its size is 10x10.

## Solutions
The solutions of the maze with different sizes and complexities give visualisation plots of the solutions by intoducing their q-tables. The number of steps taken by the agent depend on the complexity of the maze. The possibilities of falling into a hole or climbing a mountain are driven by the different amount of rewards. There is no stopping in the algorithm before the target is reached, meaning that "...Running Q-learning..." or "...Running SARSA..." is `terminated` or `truncated` depending on the definition of the observation and rewards implemented in the `step` function.

## How to use the notebook?
There is no need to import the environments into the solutions code but the cells of the notebooks have to be compiled in order.

**A few points that are worth observing**

After runing the code, 
- try to change a few parameters in the maze solutions files such that the `seed` value that affect the reset of the environment, the initial agent location (careful as this sometimes needs to be changed from the implemented functions),
- understand the way that the `step` function is implemented and how it differs in the different maze,
- understand how the truncated and terminated states are affected, use `env_check`modules or utils in the different environments,
- understand how a change in the rewards will affect the Q-values and the path taken to reach the target,
- change the number of episodes depending on the maze complexity and understand how to prevent infinite loop with the algorithms implemented
