"""Mouse Gym — reset-free rollout formatting for continual RL."""

from importlib.metadata import version

from mouse_gym.build import make_env, make_group_env
from mouse_gym.config import EnvConfig
from mouse_gym.format import (
    FieldSpec,
    GroupEnv,
    GroupMetrics,
    InputSpec,
    Metrics,
    OutputSpec,
    SingleEnv,
    StepOutput,
)

__version__ = version("mouse-gym")

__all__ = [
    "__version__",
    "EnvConfig",
    "FieldSpec",
    "GroupEnv",
    "GroupMetrics",
    "InputSpec",
    "make_env",
    "make_group_env",
    "Metrics",
    "OutputSpec",
    "SingleEnv",
    "StepOutput",
]
