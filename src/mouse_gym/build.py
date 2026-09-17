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

        env = make_env(EnvConfig(id="CartPole-v1", seed=0, max_task_episodes=5))
        for _ in range(1000):
            output = env.step(env.sample_random_input())
    """
    return SingleEnv(_make_env_instance(config))


def make_group_env(configs: list[EnvConfig], *, max_threads: int = 0) -> GroupEnv:
    """Create a :class:`GroupEnv` from a list of :class:`EnvConfig` objects.

    Each config creates one independent :class:`SingleEnv`. You can also construct
    :class:`GroupEnv` directly from existing :class:`SingleEnv` instances::

        env_a = make_env(EnvConfig(id="CartPole-v1", seed=0))
        env_b = make_env(EnvConfig(id="CartPole-v1", seed=1))
        big = GroupEnv([env_a, env_b])
        sub = GroupEnv([env_a])   # overlapping groups are fine

    ``max_threads`` controls how ``GroupEnv.step`` runs constituent envs:

    - ``0`` (default) — step every env on the calling thread.
    - ``> 0`` — distribute envs across up to ``max_threads`` worker threads
      (capped at the number of envs). Output order matches input order.
      On free-threaded CPython (``3.14t``) workers can run env steps in
      parallel. On a GIL-enabled interpreter, cheap Python envs
      (e.g. CartPole) are usually faster with ``max_threads=0``.

    Usage::

        env = make_group_env([
            EnvConfig(id="CartPole-v1", seed=0, name="cp-0", max_task_episodes=5),
            EnvConfig(id="CartPole-v1", seed=1, name="cp-1", max_task_episodes=5),
            EnvConfig(id="MountainCar-v0", seed=2, name="mc-0", max_task_episodes=5),
        ], max_threads=3)
        for _ in range(1000):
            outputs = env.step(env.sample_random_input())
            cartpole_outs = outputs[:2]
            mountaincar_outs = outputs[2:3]
    """
    return GroupEnv(
        [SingleEnv(_make_env_instance(cfg)) for cfg in configs],
        max_threads=max_threads,
    )


def _make_env_instance(config: EnvConfig) -> _EnvInstance:
    """Build one env instance from one :class:`EnvConfig`."""
    name = _resolve_display_name(config)
    return _EnvInstance(
        env=_make_plain_single_env(config),
        name=name,
        seed=config.seed,
        reward_transform=config.reward_transform,
        episode_reset_options=config.episode_reset_options,
        task_reset_options=config.task_reset_options,
        max_task_episodes=config.max_task_episodes,
        terminate_task=config.terminate_task,
    )


def _make_plain_single_env(config: EnvConfig) -> gym.Env:
    if config.env_fn is not None:
        return config.env_fn()
    assert config.id is not None
    env_kwargs: dict[str, Any] = dict(config.kwargs or {})
    if config.render and "render_mode" not in env_kwargs:
        env_kwargs["render_mode"] = "human"
    return gym.make(config.id, **env_kwargs)
