"""Environment configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class EnvConfig:
    """Configuration for building an environment via :func:`mouse_gym.make_env`.

    Attributes:
        seed: Integer used by ``seed_mode``. Required for ``"constant"`` and
            ``"stream"``. Must be ``None`` for ``"none"``.
        seed_mode: How mouse-gym chooses the ``seed`` argument of every
            internal ``env.reset`` (the first reset and each autoreset):
            ``"constant"`` passes ``seed`` every time; ``"stream"`` uses
            ``seed`` to start an RNG and draws a new reset seed from it each
            time; ``"none"`` passes ``None``. Defaults to ``"none"``.
        id: Gymnasium env ID (e.g. ``"CartPole-v1"``). Mutually exclusive with
            ``env_fn`` — provide exactly one of ``id`` or ``env_fn``.
        env_fn: Zero-arg factory that returns a freshly built Gymnasium env.
            Mutually exclusive with ``id``. Any Gymnasium env is valid,
            including one the caller has already wrapped.
        name: Display name for the env instance. Defaults to ``id`` when ``id`` is
            set, otherwise to the factory callable's ``__name__`` (named functions
            and classes only — anonymous ``lambda`` factories require an explicit
            ``name``).
        kwargs: Extra keyword arguments forwarded to ``gymnasium.make`` (``id`` configs only).
            Pass ``render_mode`` here when the Gymnasium env should render.
        reset_options: Options forwarded to every underlying ``env.reset(options=...)``
            that mouse-gym issues inside ``step()`` (the first reset and each
            autoreset after ``terminated`` or ``truncated``). Each reset receives
            its own deep copy, taken from a snapshot captured when the env is built.
        reward_transform: Optional callable applied to each step reward
            (including reset frames) before it is written to the step output
            and accumulated into episode metrics. Called with ``step_index``,
            ``state``, ``action``, ``reward``, ``terminated``, ``truncated``,
            and ``next_state``. ``state`` is the observation before the step.
            On the first reset frame it is the reset observation, because no
            earlier observation exists. On a later reset frame it is the
            observation from the previous step. ``next_state`` is the
            observation after the step. ``action`` is the value passed to
            ``step()``, or ``None`` on reset frames, where that action is not
            applied. A reset frame is ``step_index == 0``. Its ``reward`` is
            ``0.0`` and ``terminated`` and ``truncated`` are false. Accept
            unused fields with ``**kwargs``. With no transform,
            reset-frame reward is ``0.0``. Defaults to ``None`` (identity on
            env steps).
    """

    seed: int | None = None
    seed_mode: Literal["constant", "stream", "none"] = "none"
    id: str | None = None
    name: str | None = None
    kwargs: dict | None = None
    reset_options: dict | None = None
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
        if self.reward_transform is not None and not callable(self.reward_transform):
            raise ValueError(
                "EnvConfig reward_transform must be a callable "
                "that accepts the transition kwargs, or None."
            )
        if self.seed_mode not in ("constant", "stream", "none"):
            raise ValueError(
                "EnvConfig seed_mode must be 'constant', 'stream', or 'none'; "
                f"got {self.seed_mode!r}."
            )
        if self.seed_mode == "none":
            if self.seed is not None:
                raise ValueError("EnvConfig seed must be None when seed_mode is 'none'.")
            return
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(
                "EnvConfig seed must be an int when seed_mode is "
                f"{self.seed_mode!r}; got {self.seed!r}."
            )
        if self.seed < 0:
            raise ValueError(f"EnvConfig seed must be >= 0; got {self.seed}.")
