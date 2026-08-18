"""Environment configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class EnvConfig:
    """Configuration for building an environment via :func:`mouse_gym.make_env`.

    Attributes:
        episode_seed: Seeds the per-episode stream. A fresh seed is drawn on
            every episode reset inside ``step()`` and passed to the underlying
            ``env.reset(seed=...)``. Governs episode-level randomness — e.g.
            the start position — which may vary from episode to episode; the
            underlying problem instance should not depend on it. Not a public
            ``reset()`` on mouse-gym envs.
        task_seed: Seeds the per-task stream (optional). A new seed is drawn at
            each task start, stays constant for every episode reset within that
            task, and is forwarded to the underlying env on every reset as
            ``options["task_seed"]``. Governs the problem instance itself —
            e.g. the FrozenLake map or other procedurally generated variables —
            so all episodes in a task face the same problem. Envs that don't
            read ``options["task_seed"]`` are unaffected. ``None`` (default)
            disables the stream and leaves reset options untouched.
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
            ``0.0``).
    """

    episode_seed: int
    id: str | None = None
    task_seed: int | None = None
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
        for options_field in ("episode_reset_options", "task_reset_options"):
            options = getattr(self, options_field)
            if options and "task_seed" in options:
                raise ValueError(
                    f"EnvConfig {options_field} must not contain 'task_seed'; "
                    "that key is managed by mouse-gym — set the task_seed field instead."
                )
