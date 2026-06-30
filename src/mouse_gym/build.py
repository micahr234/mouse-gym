"""Build environments from :class:`EnvConfig`."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from mouse_gym.config import EnvConfig
from mouse_gym.format import GroupEnv, SingleEnv, _EnvInstance
from mouse_gym.wrappers import SeedStreamWrapper


def _require_env_id(env_id: str) -> None:
    if not env_id:
        raise ValueError(
            "id is required on EnvConfig but was not set. "
            "Provide a non-empty id (e.g. 'CartPole-v1')."
        )


def make_env(config: EnvConfig) -> SingleEnv:
    """Create a standalone :class:`SingleEnv` from one :class:`EnvConfig`.

    For multiple environments use :func:`make_group_env`.

    Usage::

        env = make_env(EnvConfig(id="CartPole-v1", reset_seed=0, episodes_per_task=5))
        for _ in range(1000):
            output = env.step(env.sample_random_input())
    """
    return SingleEnv(_make_env_instance(config))


def make_group_env(configs: list[EnvConfig]) -> GroupEnv:
    """Create a :class:`GroupEnv` from a list of :class:`EnvConfig` objects.

    Each config creates one independent :class:`SingleEnv`. You can also construct
    :class:`GroupEnv` directly from existing :class:`SingleEnv` instances::

        env_a = make_env(EnvConfig(id="CartPole-v1", reset_seed=0))
        env_b = make_env(EnvConfig(id="CartPole-v1", reset_seed=1))
        big = GroupEnv([env_a, env_b])
        sub = GroupEnv([env_a])   # overlapping groups are fine

    Usage::

        env = make_group_env([
            EnvConfig(id="CartPole-v1", reset_seed=0, name="cp-0", episodes_per_task=5),
            EnvConfig(id="CartPole-v1", reset_seed=1, name="cp-1", episodes_per_task=5),
            EnvConfig(id="MountainCar-v0", reset_seed=2, name="mc-0", episodes_per_task=5),
        ])
        for _ in range(1000):
            outputs = env.step(env.sample_random_input())
            cartpole_outs = outputs[:2]
            mountaincar_outs = outputs[2:3]
    """
    return GroupEnv([SingleEnv(_make_env_instance(cfg)) for cfg in configs])


def _make_env_instance(config: EnvConfig) -> _EnvInstance:
    """Build one env instance from one :class:`EnvConfig`."""
    _require_env_id(config.id)
    name = config.id if config.name is None else config.name
    env_kwargs = {} if config.env_fn is not None else _prepare_plain_env_kwargs(config)
    env = _make_plain_single_env(config, env_kwargs=env_kwargs)
    return _EnvInstance(
        env=env,
        name=name,
        reset_reward=config.reset_reward,
        episode_reset_options=config.episode_reset_options,
        task_reset_options=config.task_reset_options,
        episodes_per_task=config.episodes_per_task,
    )


def _prepare_plain_env_kwargs(config: EnvConfig) -> dict[str, Any]:
    env_kwargs = dict(config.kwargs or {})
    if config.render and "render_mode" not in env_kwargs:
        env_kwargs["render_mode"] = "human"
    return env_kwargs


def _make_plain_single_env(
    config: EnvConfig,
    *,
    env_kwargs: dict[str, Any],
) -> gym.Env:
    def env_fn() -> gym.Env:
        if config.env_fn is not None:
            return config.env_fn()
        kw = dict(env_kwargs)
        return gym.make(config.id, **kw)

    return SeedStreamWrapper(
        env_fn,
        reset_seed=config.reset_seed,
    )
