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
            to every episode in the task. With no task boundary
            (``max_task_episodes=0`` and no ``terminate_task``) the env is
            seeded once, on the first reset. Not a public ``reset()`` on
            mouse-gym envs.
        id: Gymnasium env ID (e.g. ``"CartPole-v1"``). Mutually exclusive with
            ``env_fn`` — provide exactly one of ``id`` or ``env_fn``.
        env_fn: Zero-arg factory that returns a freshly built Gymnasium env.
            Mutually exclusive with ``id``.
        name: Display name for the env instance. Defaults to ``id`` when ``id`` is
            set, otherwise to the factory callable's ``__name__`` (named functions
            and classes only — anonymous ``lambda`` factories require an explicit
            ``name``).
        max_task_episodes: Number of episodes before the task is truncated
            (``task_done=2``). Defaults to ``0`` (unlimited) — this timeout never
            fires on its own. A task can still end earlier via
            ``terminate_task``.
        terminate_task: Optional callable that decides whether the task
            terminates (``task_done=1``). Invoked on episode-end steps with
            the same keyword arguments as ``reward_transform``
            (``step_index``, ``episode_index``, ``state``, ``action``,
            ``reward``, ``done``, ``next_state``). ``reward`` is the
            post-``reward_transform`` value. A truthy return emits
            ``task_done=1`` and wins over ``max_task_episodes``. Defaults
            to ``None`` (no terminate condition).
        kwargs: Extra keyword arguments forwarded to ``gymnasium.make`` (``id`` configs only).
        episode_reset_options: Options forwarded to underlying ``env.reset(options=...)``
            inside ``step()`` (every reset frame).
        task_reset_options: Options overlaid on ``episode_reset_options`` when a
            reset frame starts a new task (after ``task_done`` 1 or 2).
        render: Enable render mode (``"human"``) for ``id`` configs when not already in
            ``kwargs``.
        reward_transform: Optional callable applied to each step reward
            (including reset frames) before it is written to the step output
            and accumulated into episode/task metrics. Called with
            ``step_index``, ``episode_index``, ``state``, ``action``,
            ``reward``, ``done``, and ``next_state``. ``done`` is the
            episode-done code (``0``/``1``/``2``). ``state`` is the
            observation before the step (``None`` on the first reset);
            ``next_state`` is the observation after. ``action`` is the
            step-input action array, or ``None`` on reset frames. A reset
            frame is ``step_index == 0`` (``reward`` is ``0.0``, no
            Gymnasium reward); a task-start reset is also
            ``episode_index == 0``. Accept unused fields with
            ``**kwargs``. With no transform, reset-frame reward is
            ``0.0``. Defaults to ``None`` (identity on env steps).
    """

    seed: int
    id: str | None = None
    max_task_episodes: int = 0
    terminate_task: Callable[..., bool] | None = None
    name: str | None = None
    kwargs: dict | None = None
    episode_reset_options: dict | None = None
    task_reset_options: dict | None = None
    render: bool = False
    env_fn: Callable[[], Any] | None = None
    reward_transform: Callable[..., float] | None = None

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
        if self.max_task_episodes < 0:
            raise ValueError(
                f"EnvConfig max_task_episodes must be >= 0 (0 = unlimited); "
                f"got {self.max_task_episodes}."
            )
        if self.reward_transform is not None and not callable(self.reward_transform):
            raise ValueError(
                "EnvConfig reward_transform must be a callable "
                "that accepts the transition kwargs, or None."
            )
        if self.terminate_task is not None and not callable(self.terminate_task):
            raise ValueError(
                "EnvConfig terminate_task must be a callable "
                "that accepts the transition kwargs, or None."
            )
