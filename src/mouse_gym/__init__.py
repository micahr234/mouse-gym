"""Mouse Gym — reset-free rollout formatting for Gymnasium environments."""

from importlib.metadata import version

from mouse_gym.build import make_env, make_group_env
from mouse_gym.config import EnvConfig
from mouse_gym.format import (
    GroupEnv,
    GroupMetrics,
    Metrics,
    SingleEnv,
)

__version__ = version("mouse-gym")

__all__ = [
    "__version__",
    "EnvConfig",
    "GroupEnv",
    "GroupMetrics",
    "make_env",
    "make_group_env",
    "Metrics",
    "SingleEnv",
]
