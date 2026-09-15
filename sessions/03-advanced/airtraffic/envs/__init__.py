from .flight import Flight
from .simulator import CollisionCourseSimulator
from .flight_4d import Flight4DEnv

try:
    import gymnasium
    gymnasium.register(
        id="Flight4DEnv-v0",
        entry_point="envs.flight_4d:Flight4DEnv",
    )
except Exception:
    pass

__all__ = ["Flight", "CollisionCourseSimulator", "Flight4DEnv"]
