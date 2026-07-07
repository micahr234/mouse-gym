"""Environment configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class EnvConfig:
    """Configuration for building an environment via :func:`mouse_gym.make_env`.

    Attributes:
        reset_seed: Seed stream for underlying ``env.reset(seed=...)`` calls inside
            ``step()``. Not a public ``reset()`` on mouse-gym envs.
        id: Gymnasium env ID (e.g. ``"CartPole-v1"``). Mutually exclusive with
            ``env_fn`` — provide exactly one of ``id`` or ``env_fn``.
        env_fn: Zero-arg factory that returns a freshly built Gymnasium env.
            Mutually exclusive with ``id``.
        name: Display name for the env instance. Defaults to ``id`` when ``id`` is
            set, otherwise to the factory callable's ``__name__`` (named functions
            and classes only — anonymous ``lambda`` factories require an explicit
            ``name``).
        episodes_per_task: Number of episodes before the task terminates. Defaults to
            ``0`` (unlimited) — the task boundary (done codes 3/4) never fires
            automatically.
        kwargs: Extra keyword arguments forwarded to ``gymnasium.make`` (``id`` configs only).
        episode_reset_options: Options forwarded to underlying ``env.reset(options=...)``
            inside ``step()`` (every reset frame).
        task_reset_options: Options overlaid on ``episode_reset_options`` when a
            reset frame starts a new task (after ``done`` 3/4).
        render: Enable render mode (``"human"``) for ``id`` configs when not already in
            ``kwargs``.
        reset_reward: Reward value on reset frames (``time=0``, ``done=0`` outputs
            from ``step()``; default ``0.0``).
    """

    reset_seed: int
    id: str | None = None
    episodes_per_task: int = 0
    name: str | None = None
    kwargs: dict | None = None
    episode_reset_options: dict | None = None
    task_reset_options: dict | None = None
    render: bool = False
    env_fn: Callable[[], Any] | None = None
    reset_reward: float = 0.0

    def __post_init__(self) -> None:
        has_id = self.id is not None
        has_fn = self.env_fn is not None
        if has_id and has_fn:
            raise ValueError("EnvConfig: set id or env_fn, not both.")
        if not has_id and not has_fn:
            raise ValueError("EnvConfig: set id or env_fn (at least one required).")
        if self.id is not None and not self.id.strip():
            raise ValueError("EnvConfig id must be a non-empty string.")
