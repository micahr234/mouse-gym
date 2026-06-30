"""Environment configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class EnvConfig:
    """Configuration for building an environment via :func:`mouse_gym.make_env`.

    Attributes:
        id: Gymnasium env ID (e.g. ``"CartPole-v1"``). Used as the base name when
            ``name`` is not set.
        reset_seed: Seed for mouse-gym's internal Gymnasium reset stream.
        episodes_per_task: Number of episodes before the task terminates. Defaults to
            ``0`` (unlimited) — the task boundary (done codes 3/4) never fires
            automatically.
        name: Optional display name; overrides ``id`` for env instance naming.
        kwargs: Extra keyword arguments forwarded to ``gymnasium.make``.
        episode_reset_options: Extra options forwarded to every internal
            ``env.reset(options=...)``.
        task_reset_options: Extra options overlaid on top of ``episode_reset_options``
            when an internal reset starts a new task.
        render: Enable render mode (``"human"``).
        env_fn: Zero-arg factory that returns a freshly built Gymnasium env. When
            set, ``id`` is used only for naming.
        reset_reward: Reward value injected into the reset frame (default ``0.0``).
    """

    id: str
    reset_seed: int
    episodes_per_task: int = 0
    name: str | None = None
    kwargs: dict | None = None
    episode_reset_options: dict | None = None
    task_reset_options: dict | None = None
    render: bool = False
    env_fn: Callable[[], Any] | None = None
    reset_reward: float = 0.0
