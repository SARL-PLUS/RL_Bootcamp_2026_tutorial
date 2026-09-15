"""Crippled Ant environments for Session #2 — Continuous Control & the Reality Gap."""
import gymnasium as gym

from envs.crippled_ant import LEG_JOINTS, CrippledAnt, make_ant

# Convenience registration: gym.make("CrippledAnt-v5", disabled_legs=[0]).
# No max_episode_steps here — the inner gym.make("Ant-v5") already applies
# the standard 1000-step TimeLimit.
if "CrippledAnt-v5" not in gym.registry:
    gym.register(
        id="CrippledAnt-v5",
        entry_point="envs.crippled_ant:make_ant",
    )

__all__ = ["CrippledAnt", "make_ant", "LEG_JOINTS"]
