"""Environment configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class EnvConfig:
    """Configuration for building an environment via :func:`mouse_gym.make_env`.

    Attributes:
        seed: Seeds the reset stream. One value is drawn per task and passed to
            the underlying ``env.reset(seed=...)`` at the task-start reset
            only; episode resets within a task pass no seed, following
            Gymnasium's seed-once-per-session convention. Each task is thus a
            fresh, seeded Gymnasium session: the whole task is reproducible
            from its seed, while episode-level randomness (e.g. the start
            position) still varies across episodes as the env's RNG continues.
            Envs that generate their problem instance (e.g. a procedural map)
            when ``reset`` receives an explicit seed present the same instance
            to every episode in the task. With ``episodes_per_task=0`` the env
            is seeded once, on the first reset. Not a public ``reset()`` on
            mouse-gym envs.
        id: Gymnasium env ID (e.g. ``"CartPole-v1"``). Mutually exclusive with
            ``env_fn`` — provide exactly one of ``id`` or ``env_fn``.
        env_fn: Zero-arg factory that returns a freshly built Gymnasium env.
            Mutually exclusive with ``id``.
        name: Display name for the env instance. Defaults to ``id`` when ``id`` is
            set, otherwise to the factory callable's ``__name__`` (named functions
            and classes only — anonymous ``lambda`` factories require an explicit
            ``name``).
        episodes_per_task: Number of episodes before the task is truncated
            (``task_done=2``). Defaults to ``0`` (unlimited) — the task boundary
            never fires automatically. ``task_done=1`` (task terminated) is
            reserved and unused.
        kwargs: Extra keyword arguments forwarded to ``gymnasium.make`` (``id`` configs only).
        episode_reset_options: Options forwarded to underlying ``env.reset(options=...)``
            inside ``step()`` (every reset frame).
        task_reset_options: Options overlaid on ``episode_reset_options`` when a
            reset frame starts a new task (after ``task_done`` 2).
        render: Enable render mode (``"human"``) for ``id`` configs when not already in
            ``kwargs``.
        reset_reward: Reward value on reset frames (``step_index=0``,
            ``episode_done=0``, ``task_done=0`` outputs from ``step()``; default
            ``0.0``). Not multiplied by ``reward_scale``.
        reward_scale: Multiplier applied to the underlying Gymnasium step reward
            in the ``step()`` output (default ``1.0``). Reset-frame rewards use
            ``reset_reward`` as given. ``env.metrics`` records raw (unscaled)
            episode and task returns.
    """

    seed: int
    id: str | None = None
    episodes_per_task: int = 0
    name: str | None = None
    kwargs: dict | None = None
    episode_reset_options: dict | None = None
    task_reset_options: dict | None = None
    render: bool = False
    env_fn: Callable[[], Any] | None = None
    reset_reward: float = 0.0
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        has_id = self.id is not None
        has_fn = self.env_fn is not None
        if has_id and has_fn:
            raise ValueError("EnvConfig: set id or env_fn, not both.")
        if not has_id and not has_fn:
            raise ValueError("EnvConfig: set id or env_fn (at least one required).")
        if self.id is not None and not self.id.strip():
            raise ValueError("EnvConfig id must be a non-empty string.")
        if has_fn and self.kwargs is not None:
            raise ValueError(
                "EnvConfig kwargs only apply to id configs (forwarded to gymnasium.make); "
                "pass construction arguments inside env_fn instead."
            )
        if has_fn and self.render:
            raise ValueError(
                "EnvConfig render only applies to id configs; "
                "set render_mode inside env_fn instead."
            )
        if self.episodes_per_task < 0:
            raise ValueError(
                f"EnvConfig episodes_per_task must be >= 0 (0 = unlimited); "
                f"got {self.episodes_per_task}."
            )
