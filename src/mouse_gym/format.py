"""Public step API and rollout types for mouse-gym."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from mouse_gym.config import EnvConfig
from mouse_gym.process import ProcessEnvs

_RESET_SEED_BOUND = 2**32


def _copy_step_value(value: Any) -> Any:
    """Return an independent copy of an observation or info dict.

    Some Gymnasium envs reuse one array or dict and overwrite it on the next
    call. The copy is what ``step`` returns and what the next reward transform
    sees as ``state``.
    """
    return copy.deepcopy(value)


class _EnvInstance:
    """Internal: wraps a single ``gym.Env`` with the mouse-gym step protocol.

    Each env instance maintains the internal reset and transform context for
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
        # Snapshot at build time. Nested values are copied too, so later edits
        # to the caller's dict do not change resets that have already been configured.
        self._reset_options = copy.deepcopy(reset_options) if reset_options else {}
        self._seed = seed
        self._seed_mode = seed_mode
        self._seed_rng = np.random.default_rng(seed) if seed_mode == "stream" else None

        # Execution context for autoresets and reward transforms.
        self._needs_initial_reset = True
        self._autoreset_pending = False
        self._step_index = 0
        self._observation: Any = None

        # Cache spaces so step() does not hop Gymnasium wrapper properties.
        self._action_space = env.action_space
        self._observation_space = env.observation_space

    @property
    def name(self) -> str:
        return self._name

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

    def _do_reset(self) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Call env.reset() and return the reset-frame tuple."""
        # Fresh copy each reset so the env can mutate this dict without
        # changing the snapshot used on the next reset.
        options = copy.deepcopy(self._reset_options) or None
        obs, info = self._env.reset(
            seed=self._next_reset_seed(),
            options=options,
        )
        obs = _copy_step_value(obs)
        info = _copy_step_value(info)
        info["executed_action"] = None
        self._step_index = 0
        # The first reset has no earlier observation, so its state is the
        # reset observation. Later resets pass the observation we were holding.
        state = obs if self._observation is None else self._observation
        reward_f = self._apply_reward_transform(
            step_index=0,
            state=state,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            next_state=obs,
        )
        self._observation = obs
        return obs, reward_f, False, False, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step this env instance and return a Gymnasium output tuple."""
        if self._needs_initial_reset:
            self._needs_initial_reset = False
            return self._do_reset()

        if self._autoreset_pending:
            self._autoreset_pending = False
            return self._do_reset()

        obs, raw_reward, terminated, truncated, info = self._env.step(action)
        obs = _copy_step_value(obs)
        info = _copy_step_value(info)
        # Wrappers such as RepeatTaskEnv can emit inner reset frames without
        # ending this outer episode. Preserve their no-action indication.
        info.setdefault("executed_action", _copy_step_value(action))
        terminated = bool(terminated)
        truncated = bool(truncated)

        self._step_index += 1

        state = self._observation
        reward_f = self._apply_reward_transform(
            step_index=self._step_index,
            state=state,
            action=_copy_step_value(info["executed_action"]),
            reward=float(raw_reward),
            terminated=terminated,
            truncated=truncated,
            next_state=obs,
        )
        self._observation = obs

        if terminated or truncated:
            self._autoreset_pending = True
        return obs, reward_f, terminated, truncated, info

    def render(self) -> Any:
        """Return whatever the wrapped env's ``render()`` returns."""
        return self._env.render()

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
    The observation and ``info`` dict are deep-copied before they are returned.
    ``info["executed_action"]`` is the action that produced the observation,
    or ``None`` when the call reset the environment. A wrapped env may also
    set this reserved key to ``None`` for an inner reset; reward transforms
    receive ``action=None`` on those frames too.
    Rollout progress and statistics are collected separately: create a
    :class:`mouse_gym.RolloutTracker` and call ``tracker.update(output=output)``
    after every step. Use :class:`mouse_gym.RepeatTaskTracker` with repeated
    tasks. Environments do not own or update trackers.
    """

    def __init__(self, env_instance: _EnvInstance) -> None:
        self._env_instance = env_instance
        self._group: GroupEnv | None = None

    @property
    def name(self) -> str:
        """Name of this env instance."""
        return self._env_instance.name

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

        Pass the returned output to a separate rollout tracker when progress
        tracking or completed summaries are needed.
        """
        return self._env_instance.step(action)

    def render(self) -> Any:
        """Return whatever the wrapped env's ``render()`` returns."""
        return self._env_instance.render()

    def close(self) -> None:
        """Close the underlying env."""
        self._env_instance.close()


class _RemoteInstance(_EnvInstance):
    """A SingleEnv's public operations routed to its owning process."""

    def __init__(self, owner: ProcessEnvs, index: int, metadata: tuple) -> None:
        self._owner, self._index = owner, index
        self._name, self._action_space, self._observation_space = metadata
        self._closed = False

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self._owner.call("step", {self._index: action})[self._index]

    def render(self) -> Any:
        return self._owner.call("render", {self._index: None})[self._index]

    def close(self) -> None:
        if not self._closed and not self._owner._closed:
            self._owner.call("close_env", {self._index: None})
        self._closed = True


class GroupEnv:
    """Reset-free environments, stepped inline or in persistent processes.

    Pass EnvConfigs to build environments inside their workers. Existing
    SingleEnv objects can be grouped inline (num_workers=0); live simulator
    instances are never serialized to child processes. Use num_workers > 0
    for spawned processes, capped at the number of environments.

    Actions and outputs pass through reusable shared memory. Returned arrays
    remain valid after subsequent calls and after close(). Output order matches
    input order, including heterogeneous spaces. Keep separate rollout trackers
    in the caller. Workers do not own trackers or model inference.
    """

    def __init__(self, envs: Sequence[SingleEnv] | Sequence[EnvConfig], *, num_workers: int = 0) -> None:
        from mouse_gym.build import make_env

        if isinstance(envs, (str, bytes)) or not isinstance(envs, Sequence):
            raise ValueError("envs must be a sequence of EnvConfig or SingleEnv instances")
        if not envs:
            raise ValueError("GroupEnv requires at least one environment")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        configs = all(isinstance(env, EnvConfig) for env in envs)
        if not configs and not all(isinstance(env, SingleEnv) for env in envs):
            raise ValueError("Pass only EnvConfig objects or only SingleEnv objects")
        if num_workers and not configs:
            raise ValueError("Process workers require EnvConfig objects; live SingleEnv instances cannot be moved")
        if not configs:
            if len({id(env) for env in envs}) != len(envs):
                raise ValueError("GroupEnv envs must be unique; pass each SingleEnv once")
            for env in envs:
                assert isinstance(env, SingleEnv)
                if env._group is not None:
                    raise ValueError(f"{env.name!r} is already in a GroupEnv")
        self._num_workers = num_workers
        self._workers: ProcessEnvs | None = None
        self._envs: list[SingleEnv] = []
        self._closed = False
        try:
            if num_workers:
                self._workers = ProcessEnvs(envs, num_workers)
                self._envs = [SingleEnv(_RemoteInstance(self._workers, i, self._workers.metadata[i]))
                              for i in range(len(envs))]
            else:
                for env in envs:
                    self._envs.append(make_env(env) if isinstance(env, EnvConfig) else env)
            for env in self._envs:
                env._group = self
            self._action_space = gym.spaces.Tuple(tuple(env.action_space for env in self._envs))
            self._observation_space = gym.spaces.Tuple(tuple(env.observation_space for env in self._envs))
        except BaseException:
            self.close()
            raise

    @property
    def envs(self) -> tuple[SingleEnv, ...]:
        """Constituent env handles in order; process handles route calls remotely."""
        return tuple(self._envs)

    @property
    def num_envs(self) -> int:
        return len(self._envs)

    @property
    def num_workers(self) -> int:
        """Requested process cap; zero runs inline."""
        return self._num_workers

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(env.name for env in self._envs)

    @property
    def action_space(self) -> gym.spaces.Tuple:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Tuple:
        return self._observation_space

    def step(self, actions: Sequence[Any]) -> tuple[tuple[Any, float, bool, bool, dict[str, Any]], ...]:
        """Step all envs, preserving reset frames and construction order."""
        if self._closed:
            raise RuntimeError("GroupEnv is closed")
        if isinstance(actions, (str, bytes)) or not isinstance(actions, (list, tuple)):
            raise ValueError("actions must be a list or tuple with one action per env instance")
        if len(actions) != self.num_envs:
            raise ValueError(f"actions must contain exactly {self.num_envs} entries, got {len(actions)}")
        if self._workers is None:
            return tuple(env.step(action) for env, action in zip(self._envs, actions))
        outputs = self._workers.call("step", dict(enumerate(actions)))
        return tuple(outputs[i] for i in range(self.num_envs))

    def render(self) -> tuple[Any, ...]:
        if self._closed:
            raise RuntimeError("GroupEnv is closed")
        if self._workers is None:
            return tuple(env.render() for env in self._envs)
        outputs = self._workers.call("render", dict.fromkeys(range(self.num_envs)))
        return tuple(outputs[i] for i in range(self.num_envs))

    def close(self) -> None:
        """Close environments and release processes and shared-memory slots."""
        if self._closed:
            return
        self._closed = True
        if self._workers is not None:
            self._workers.close()
        else:
            for env in self._envs:
                env.close()

    def __del__(self):
        if hasattr(self, "_closed"):
            self.close()
