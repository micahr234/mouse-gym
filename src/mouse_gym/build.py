"""Build environments from :class:`EnvConfig`."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from mouse_gym.config import EnvConfig
from mouse_gym.format import GroupEnv, SingleEnv, _EnvInstance


def _resolve_display_name(config: EnvConfig) -> str:
    if config.name is not None:
        return config.name
    if config.id is not None:
        return config.id
    assert config.env_fn is not None
    fn_name = getattr(config.env_fn, "__name__", "")
    if fn_name and fn_name != "<lambda>":
        return fn_name
    raise ValueError(
        "EnvConfig requires name when env_fn is an anonymous callable (e.g. lambda); "
        "use a named function or class factory, or set name explicitly."
    )


def make_env(config: EnvConfig) -> SingleEnv:
    """Create a standalone :class:`SingleEnv` from one :class:`EnvConfig`.

    For multiple environments use :func:`make_group_env`.

    Usage::

        env = make_env(EnvConfig(id="CartPole-v1"))
        for _ in range(1000):
            env.step(env.action_space.sample())
    """
    return SingleEnv(_make_env_instance(config))


def make_group_env(configs: list[EnvConfig], *, num_workers: int = 0) -> GroupEnv:
    """Build a group inline (0) or in persistent spawned worker processes.

    Each worker constructs and keeps its environments. Configs can contain
    notebook-local factories and reward transforms. Action/result payloads use
    reusable shared memory; output order always matches configuration order.
    In scripts, create process groups under ``if __name__ == "__main__":``.
    Close the group when finished to release workers and shared memory.

    Example::

        env = make_group_env([
            EnvConfig(id="CartPole-v1", seed=i, seed_mode="stream")
            for i in range(16)
        ], num_workers=4)
        try:
            outputs = env.step(env.action_space.sample())
        finally:
            env.close()
    """
    return GroupEnv(configs, num_workers=num_workers)


def _make_env_instance(config: EnvConfig) -> _EnvInstance:
    """Build one env instance from one :class:`EnvConfig`."""
    name = _resolve_display_name(config)
    return _EnvInstance(
        env=_make_plain_single_env(config),
        name=name,
        seed=config.seed,
        seed_mode=config.seed_mode,
        reward_transform=config.reward_transform,
        reset_options=config.reset_options,
    )


def _make_plain_single_env(config: EnvConfig) -> gym.Env:
    if config.env_fn is not None:
        return config.env_fn()
    assert config.id is not None
    env_kwargs: dict[str, Any] = dict(config.kwargs or {})
    return gym.make(config.id, **env_kwargs)
