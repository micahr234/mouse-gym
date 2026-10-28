"""Mouse Gym — reset-free rollout formatting for Gymnasium environments."""

from importlib.metadata import version

from mouse_gym.build import make_env, make_group_env
from mouse_gym.config import EnvConfig
from mouse_gym.format import (
    GroupEnv,
    SingleEnv,
)

from mouse_gym.tracker import EpisodeSummary, RepeatTaskTracker, RolloutTracker, TaskSummary

__version__ = version("mouse-gym")

__all__ = [
    "__version__",
    "EnvConfig",
    "GroupEnv",
    "make_env",
    "make_group_env",
    "SingleEnv",
    "RolloutTracker",
    "RepeatTaskTracker",
    "EpisodeSummary",
    "TaskSummary",
]
