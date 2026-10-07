"""Public step API and rollout types for mouse-gym."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import gymnasium as gym
import numpy as np

_RESET_SEED_BOUND = 2**32


class Metrics:
    """Accumulates episode statistics for a single :class:`SingleEnv`.

    :class:`SingleEnv` records a completed episode whenever the wrapped env
    reports ``terminated`` or ``truncated``. Those flags mean whatever the
    wrapped env uses them for. Call :meth:`clear` to wipe accumulated data
    (for example between evaluation runs).

    Attributes
    ----------
    episode_cum_rewards:
        List of cumulative rewards for every episode completed since the last
        :meth:`clear` call. Values use the post-``reward_transform`` rewards,
        including the reset-frame reward (``step_index == 0``).
    episode_lengths:
        List of episode step counts for every completed episode since the last
        :meth:`clear` call.
    """

    def __init__(self) -> None:
        self._episode_cum_rewards: list[float] = []
        self._episode_lengths: list[float] = []

    def _record(self, cum_reward: float, length: float) -> None:
        self._episode_cum_rewards.append(cum_reward)
        self._episode_lengths.append(length)

    def clear(self) -> None:
        """Wipe all accumulated episode data."""
        self._episode_cum_rewards = []
        self._episode_lengths = []

    @property
    def episode_cum_rewards(self) -> list[float]:
        """Cumulative rewards for completed episodes."""
        return self._episode_cum_rewards

    @property
    def episode_lengths(self) -> list[float]:
        """Episode lengths (step counts) for completed episodes."""
        return self._episode_lengths


class _EnvInstance:
    """Internal: wraps a single ``gym.Env`` with the mouse-gym step protocol.

    Each env instance tracks its step index and cumulative reward, and implements
    the two-frame boundary sequence: a step where the wrapped env reports
    ``terminated`` or ``truncated``, followed by a reset frame
    (``terminated=False``, ``truncated=False``, ``step_index=0``) on the next
    ``step()`` call. The user's action on the reset-frame call is ignored.

    Seeding follows ``seed_mode``: ``"constant"`` repeats ``seed``,
    ``"stream"`` draws a new reset seed from an RNG started with ``seed``,
    and ``"none"`` passes ``None``.

    ``step`` takes a Gymnasium action and returns
    ``(observation, reward, terminated, truncated, info)``.
    """

    def __init__(
        self,
        env: gym.Env,
        name: str,
        *,
        seed: int | None,
        seed_mode: Literal["constant", "stream", "none"],
        reward_transform: Callable[..., float] | None = None,
        reset_options: dict | None = None,
    ):
        self._env = env
        self._name = name
        self._reward_transform = reward_transform
        self._reset_options = dict(reset_options or {})
        self._seed = seed
        self._seed_mode = seed_mode
        self._seed_rng = np.random.default_rng(seed) if seed_mode == "stream" else None

        # Episode state
        self._needs_initial_reset = True
        self._autoreset_pending = False
        self._step_index = 0
        self._state: Any = None
        self._episode_cum_reward = 0.0

        # Cache spaces so step() does not hop Gymnasium wrapper properties.
        self._action_space = env.action_space
        self._observation_space = env.observation_space

    @property
    def name(self) -> str:
        return self._name

    @property
    def step_index(self) -> int:
        """Steps since the last mouse-gym reset. ``0`` on a reset frame."""
        return self._step_index

    def _apply_reward_transform(
        self,
        *,
        step_index: int,
        state: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_state: Any,
    ) -> float:
        if self._reward_transform is None:
            return reward
        return float(
            self._reward_transform(
                step_index=step_index,
                state=state,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                next_state=next_state,
            )
        )

    def _next_reset_seed(self) -> int | None:
        """Seed argument for the next internal ``env.reset``."""
        if self._seed_mode == "none":
            return None
        if self._seed_mode == "constant":
            assert self._seed is not None
            return self._seed
        assert self._seed_rng is not None
        return int(self._seed_rng.integers(0, _RESET_SEED_BOUND))

    def _do_reset(self) -> tuple[tuple[Any, float, bool, bool, dict[str, Any]], None]:
        """Call env.reset() and return the reset-frame tuple; no metric result."""
        obs, info = self._env.reset(
            seed=self._next_reset_seed(),
            options=dict(self._reset_options) or None,
        )
        self._step_index = 0
        reward_f = self._apply_reward_transform(
            step_index=0,
            state=self._state,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            next_state=obs,
        )
        self._episode_cum_reward = reward_f
        self._state = obs
        return (obs, reward_f, False, False, info), None

    def step(
        self, action: Any
    ) -> tuple[tuple[Any, float, bool, bool, dict[str, Any]], tuple[float, float] | None]:
        """Step this env instance; return ``(output, episode_result)``.

        ``output`` is ``(observation, reward, terminated, truncated, info)``.
        ``episode_result`` is ``(cum_reward, length)`` when the wrapped env
        reports ``terminated`` or ``truncated`` on this step, or ``None``
        otherwise (including reset frames).
        """
        if self._needs_initial_reset:
            self._needs_initial_reset = False
            return self._do_reset()

        if self._autoreset_pending:
            self._autoreset_pending = False
            return self._do_reset()

        obs, raw_reward, terminated, truncated, info = self._env.step(action)
        terminated = bool(terminated)
        truncated = bool(truncated)

        self._step_index += 1

        state = self._state
        reward_f = self._apply_reward_transform(
            step_index=self._step_index,
            state=state,
            action=action,
            reward=float(raw_reward),
            terminated=terminated,
            truncated=truncated,
            next_state=obs,
        )
        self._episode_cum_reward += reward_f
        self._state = obs

        episode_result: tuple[float, float] | None
        if terminated or truncated:
            episode_result = (self._episode_cum_reward, float(self._step_index))
            self._autoreset_pending = True
        else:
            episode_result = None

        return (obs, reward_f, terminated, truncated, info), episode_result

    def render(self) -> list:
        """Return rendered frames from this env instance."""
        frames = self._env.render()
        if frames is None:
            return []
        if isinstance(frames, (list, tuple)):
            return list(frames)
        return [frames]

    def close(self) -> None:
        self._env.close()


class SingleEnv:
    """A standalone environment wrapping one gym env with the mouse-gym rollout protocol.

    Construct via :func:`mouse_gym.make_env` with a single :class:`EnvConfig`.

    ``step`` implements the reset-free mouse-gym protocol: the first call returns a
    reset frame (``terminated=False``, ``truncated=False``, ``step_index=0``,
    action ignored). After the wrapped env reports ``terminated`` or ``truncated``,
    the next ``step`` is also a reset frame. There is no public ``reset()``, so
    training loops stay a single ``step()`` stream.

    ``step(action)`` returns Gymnasium's ``(observation, reward, terminated,
    truncated, info)``. ``reward`` is a Python ``float`` after
    ``reward_transform``. ``terminated`` and ``truncated`` are Python ``bool``s.
    The observation and ``info`` dict are forwarded unchanged. Read
    :attr:`step_index` for the count since the last mouse-gym reset (``0`` on a
    reset frame).

    Episode statistics are accumulated automatically in :attr:`metrics`
    (:class:`Metrics`) whenever those flags are true. Call
    ``env.metrics.clear()`` to wipe accumulated data between evaluation runs.
    """

    def __init__(self, env_instance: _EnvInstance) -> None:
        self._env_instance = env_instance
        self._metrics = Metrics()

    @property
    def metrics(self) -> Metrics:
        """Episode statistics; accumulates results from every completed episode.

        Call ``env.metrics.clear()`` to wipe accumulated data between evaluation runs.
        """
        return self._metrics

    @property
    def name(self) -> str:
        """Name of this env instance."""
        return self._env_instance.name

    @property
    def step_index(self) -> int:
        """Steps since the last mouse-gym reset. ``0`` on a reset frame."""
        return self._env_instance.step_index

    @property
    def action_space(self) -> gym.Space:
        """The underlying Gymnasium action space."""
        return self._env_instance._action_space

    @property
    def observation_space(self) -> gym.Space:
        """The underlying Gymnasium observation space."""
        return self._env_instance._observation_space

    def reset(self, **_kwargs: Any) -> None:
        """Not supported — episode transitions happen inside ``step()``."""
        raise NotImplementedError(
            "SingleEnv does not support public reset(); call step() only. "
            "When the wrapped env reports terminated or truncated, the next "
            "step() returns a reset frame."
        )

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the env and return ``(observation, reward, terminated, truncated, info)``.

        On the first call and on any call immediately after ``terminated`` or
        ``truncated``, ``action`` is ignored and a reset frame is returned instead.

        Completed-episode statistics are recorded automatically into :attr:`metrics`.
        Call ``env.metrics.clear()`` to reset between runs.
        """
        output, episode_result = self._env_instance.step(action)
        if episode_result is not None:
            cum_reward, length = episode_result
            self._metrics._record(cum_reward, length)
        return output

    def render(self) -> list:
        """Return rendered frames from this env instance."""
        return self._env_instance.render()

    def close(self) -> None:
        """Close the underlying env."""
        self._env_instance.close()


class GroupMetrics:
    """Live read-through view over :class:`Metrics` instances of a :class:`GroupEnv`.

    Stores no episode data of its own — all reads delegate to each constituent
    :class:`SingleEnv`'s metrics. Multiple :class:`GroupEnv` instances may point
    to overlapping sets of :class:`SingleEnv` objects without any data conflicts.

    Attributes
    ----------
    episode_cum_rewards:
        Per-env list of cumulative rewards. ``episode_cum_rewards[i]`` is the
        list from ``envs[i].metrics.episode_cum_rewards``.
    episode_lengths:
        Per-env list of episode step counts. ``episode_lengths[i]`` is the list
        from ``envs[i].metrics.episode_lengths``.
    """

    def __init__(self, envs: list[SingleEnv]) -> None:
        self._envs = envs

    @property
    def episode_cum_rewards(self) -> list[list[float]]:
        """Per-env cumulative rewards, read live from each env's metrics."""
        return [e.metrics.episode_cum_rewards for e in self._envs]

    @property
    def episode_lengths(self) -> list[list[float]]:
        """Per-env episode lengths, read live from each env's metrics."""
        return [e.metrics.episode_lengths for e in self._envs]

    def clear(self) -> None:
        """Clear metrics on every constituent env."""
        for e in self._envs:
            e.metrics.clear()


class GroupEnv:
    """A pure reference container that delegates to a list of :class:`SingleEnv` instances.

    Construct via :func:`mouse_gym.make_group_env` or directly:
    ``GroupEnv([env_a, env_b, env_c], max_threads=4)``.

    ``GroupEnv`` stores no episode data. Multiple ``GroupEnv`` instances — including
    overlapping ones — can point to the same :class:`SingleEnv` objects without conflict
    when stepped sequentially::

        big = GroupEnv([env_a, env_b, env_c])
        sub = GroupEnv([env_a, env_b])   # shares env_a and env_b with big — fine

    Do not step the same :class:`SingleEnv` concurrently (for example via overlapping
    groups with ``max_threads > 0``).

    Each :class:`SingleEnv` owns its own :class:`Metrics`; ``GroupEnv.metrics``
    is a live read-through view with no independent storage.

    ``step`` takes a list or tuple of actions, one per env, and returns a flat
    list of ``(observation, reward, terminated, truncated, info)`` tuples.
    ``action_space.sample()`` returns a tuple in that order.

    With ``max_threads=0`` (default), ``step`` runs every env on the calling thread.
    With ``max_threads > 0``, ``step`` distributes envs across up to ``max_threads``
    worker threads (capped at ``num_envs``) and preserves output order. On
    free-threaded CPython (``3.14t``) those workers can run env steps in
    parallel. On a GIL-enabled interpreter, cheap Python envs are usually
    faster with ``max_threads=0``.
    """

    def __init__(self, envs: list[SingleEnv], *, max_threads: int = 0) -> None:
        if not envs:
            raise ValueError("GroupEnv requires at least one SingleEnv.")
        if max_threads < 0:
            raise ValueError(f"max_threads must be >= 0; got {max_threads}.")
        self._envs = list(envs)
        self._metrics = GroupMetrics(self._envs)
        self._max_threads = max_threads
        self._executor: ThreadPoolExecutor | None = None
        if max_threads > 0:
            self._executor = ThreadPoolExecutor(
                max_workers=min(max_threads, len(self._envs)),
            )

    @property
    def envs(self) -> list[SingleEnv]:
        """The constituent :class:`SingleEnv` instances."""
        return self._envs

    @property
    def metrics(self) -> GroupMetrics:
        """Live read-through view over each env's :class:`Metrics`; stores no data."""
        return self._metrics

    @property
    def num_envs(self) -> int:
        """Number of constituent env instances."""
        return len(self._envs)

    @property
    def max_threads(self) -> int:
        """Worker-thread cap for ``step``; ``0`` means run on the calling thread."""
        return self._max_threads

    @property
    def names(self) -> tuple[str, ...]:
        """Names of all constituent env instances."""
        return tuple(e.name for e in self._envs)

    @property
    def action_space(self) -> gym.spaces.Tuple:
        """Gymnasium tuple action space, one subspace per env instance."""
        return gym.spaces.Tuple(tuple(e.action_space for e in self._envs))

    @property
    def observation_space(self) -> gym.spaces.Tuple:
        """Gymnasium tuple observation space, one subspace per env instance."""
        return gym.spaces.Tuple(tuple(e.observation_space for e in self._envs))

    def step(
        self, actions: Sequence[Any]
    ) -> list[tuple[Any, float, bool, bool, dict[str, Any]]]:
        """Step all env instances and return one Gymnasium tuple per env.

        ``actions[i]`` is the action for env instance ``i``. A list or tuple is
        accepted; ``action_space.sample()`` returns a tuple. Returns a flat
        list of ``(observation, reward, terminated, truncated, info)``. On the
        first call and on any call immediately after ``terminated`` or
        ``truncated``, the corresponding action is ignored and a reset frame
        is returned instead.

        With ``max_threads=0``, steps run sequentially on the calling thread. With
        ``max_threads > 0``, steps are distributed across worker threads; results stay
        index-aligned with ``actions``.
        """
        if isinstance(actions, (str, bytes)) or not isinstance(actions, (list, tuple)):
            raise ValueError(
                "actions must be a list or tuple with one action per env instance; "
                f"got {type(actions).__name__}."
            )
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions must contain exactly {self.num_envs} entries, got {len(actions)}."
            )
        if self._executor is None:
            return [e.step(action) for e, action in zip(self._envs, actions)]
        return list(self._executor.map(_step_env, self._envs, actions))

    def render(self) -> list:
        """Return rendered frames from all env instances, flattened into one list."""
        frames: list = []
        for e in self._envs:
            frames.extend(e.render())
        return frames

    def close(self) -> None:
        """Close all env instances and shut down any worker threads."""
        try:
            for e in self._envs:
                e.close()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None


def _step_env(
    env: SingleEnv, action: Any
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Step one env; module-level helper for :class:`ThreadPoolExecutor.map`."""
    return env.step(action)
