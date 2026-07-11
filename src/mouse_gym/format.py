"""Public step API and rollout contract types for mouse-gym ↔ mouse-core."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Required, TypedDict, cast

import gymnasium as gym
import numpy as np

ACTION_KEY = "action"
OBS_KEY = "observation"

STEP_INDEX_KEY = "step_index"

DONE_RUNNING             = 0
DONE_EPISODE_TERMINATED  = 1
DONE_EPISODE_TRUNCATED   = 2
DONE_TASK_TERMINATED     = 3
DONE_TASK_TRUNCATED      = 4


def _np_dtype_for_space(space: gym.Space) -> np.dtype:
    """Map a Gymnasium space dtype to the numpy dtype used to store its samples."""
    raw = getattr(space, "dtype", None)
    if raw is None:
        return np.dtype(np.float32)
    return np.dtype(raw)


@dataclass
class FieldSpec:
    """Describes one field in an output or input dict.

    ``dtype`` is the numpy dtype or Python type of the value (e.g. ``np.float32``,
    ``np.int64``, ``int``). ``shape`` is the array shape as a tuple; ``()`` for
    scalars and plain Python primitives.
    """

    dtype: np.dtype | type
    shape: tuple[int, ...]


@dataclass
class OutputSpec:
    """Mirrors the output dict: one attribute per key in ``outputs[i]``.

    ``observation`` is a single :class:`FieldSpec` for standard observation spaces, or
    a ``dict[str, FieldSpec]`` for ``gym.spaces.Dict`` observation spaces.

    The underlying Gymnasium ``info`` dict is forwarded verbatim in the step output
    under the ``info`` key. No env-specific filtering is applied.
    """

    step_index: FieldSpec
    observation: FieldSpec | dict[str, FieldSpec]
    reward: FieldSpec
    done: FieldSpec
    episode_index: FieldSpec
    task_index: FieldSpec


@dataclass
class InputSpec:
    """Mirrors the input dict: one attribute per key in ``inputs[i]``.

    ``action`` describes the single ``"action"`` array. Its ``dtype`` and shape
    mirror the underlying Gymnasium action space where possible.
    """

    action: FieldSpec


class StepOutput(TypedDict, total=False):
    """All per-env fields for one step (single-env view, ``outputs[i]``).

    Array fields are ``np.ndarray`` scalars or vectors; ``episode_index`` and
    ``task_index`` are plain ``int``. The ``observation`` field is an array for
    ordinary observation spaces, or a ``dict[str, np.ndarray]`` for
    ``gym.spaces.Dict`` observation spaces.

    The underlying Gymnasium ``info`` dict is forwarded verbatim under ``info``.
    For example, ``outputs[i]["info"]["q_star"]``, ``outputs[i]["info"]["map"]``,
    and ``outputs[i]["info"]["ns_params"]`` when the env emits those keys.
    """

    step_index: Required[np.ndarray]
    observation: np.ndarray | dict[str, np.ndarray]
    reward: Required[np.ndarray]
    done: Required[np.ndarray]
    episode_index: Required[int]
    task_index: Required[int]
    info: dict[str, Any]


class Metrics:
    """Accumulates episode and task statistics for a single :class:`SingleEnv`.

    :class:`SingleEnv` feeds completed-episode and completed-task results
    automatically. Call :meth:`clear` to wipe all accumulated data (e.g.
    between evaluation runs).

    Attributes
    ----------
    episode_cum_rewards:
        List of raw (unscaled) cumulative rewards for every episode completed
        since the last :meth:`clear` call.
    episode_lengths:
        List of episode step counts for every completed episode since the last
        :meth:`clear` call.
    task_cum_rewards:
        List of raw cumulative rewards summed across every episode in each
        completed task since the last :meth:`clear` call.
    task_lengths:
        List of total step counts summed across every episode in each completed
        task since the last :meth:`clear` call.
    """

    def __init__(self) -> None:
        self._episode_cum_rewards: list[float] = []
        self._episode_lengths: list[float] = []
        self._task_cum_rewards: list[float] = []
        self._task_lengths: list[float] = []

    def _record(self, cum_reward: float, length: float) -> None:
        self._episode_cum_rewards.append(cum_reward)
        self._episode_lengths.append(length)

    def _record_task(self, cum_reward: float, length: float) -> None:
        self._task_cum_rewards.append(cum_reward)
        self._task_lengths.append(length)

    def clear(self) -> None:
        """Wipe all accumulated episode and task data."""
        self._episode_cum_rewards = []
        self._episode_lengths = []
        self._task_cum_rewards = []
        self._task_lengths = []

    @property
    def episode_cum_rewards(self) -> list[float]:
        """Raw cumulative rewards for completed episodes."""
        return self._episode_cum_rewards

    @property
    def episode_lengths(self) -> list[float]:
        """Episode lengths (step counts) for completed episodes."""
        return self._episode_lengths

    @property
    def task_cum_rewards(self) -> list[float]:
        """Raw cumulative rewards summed across episodes in completed tasks."""
        return self._task_cum_rewards

    @property
    def task_lengths(self) -> list[float]:
        """Total step counts summed across episodes in completed tasks."""
        return self._task_lengths


class _EnvInstance:
    """Internal: wraps a single ``gym.Env`` with the mouse-gym step protocol.

    Each env instance manages its own episode state — step index, episode index,
    and cumulative rewards — and implements the two-frame boundary sequence: a
    terminal step (``done=1/2``) followed by a reset frame (``done=0``,
    ``step_index=0``) on the next ``step()`` call, with the user's action on the
    reset-frame call silently ignored.
    """

    def __init__(
        self,
        env: gym.Env,
        name: str,
        *,
        reset_reward: float = 0.0,
        episode_reset_options: dict | None = None,
        task_reset_options: dict | None = None,
        episodes_per_task: int,
    ):
        self._env = env
        self._name = name
        self._reset_reward = float(reset_reward)
        self._episode_reset_options = dict(episode_reset_options or {})
        self._task_reset_options = dict(task_reset_options or {})
        self._episodes_per_task = int(episodes_per_task)

        # Episode state
        self._needs_initial_reset = True
        self._autoreset_pending = False
        self._task_done_pending = False
        self._step_index = 0
        self._episode_index = 0
        self._task_episode_count = 0  # episodes completed in current task
        self._task_index = 0
        self._episode_cum_reward = 0.0
        self._task_cum_reward = 0.0
        self._task_cum_length = 0.0

        # Spec
        self._obs_channel, self._obs_dtypes, self._output_spec, self._input_spec = (
            self._build_specs()
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_spec(self) -> OutputSpec:
        return self._output_spec

    @property
    def input_spec(self) -> InputSpec:
        return self._input_spec

    def _build_specs(
        self,
    ) -> tuple[
        str | None,
        dict[str, np.dtype],
        OutputSpec,
        InputSpec,
    ]:
        obs_space = self._env.observation_space
        act_space = self._env.action_space

        # --- observation side ---
        if isinstance(obs_space, gym.spaces.Dict):
            obs_dtypes: dict[str, np.dtype] = {
                key: _np_dtype_for_space(sub)
                for key, sub in obs_space.spaces.items()
            }
            single_channel = None
            obs_field: FieldSpec | dict[str, FieldSpec] = {
                key: FieldSpec(
                    dtype=obs_dtypes[key],
                    shape=tuple(getattr(sub, "shape", ()) or ()),
                )
                for key, sub in obs_space.spaces.items()
            }
        else:
            obs_np_dtype = _np_dtype_for_space(obs_space)
            obs_dtypes = {OBS_KEY: obs_np_dtype}
            single_channel = OBS_KEY
            obs_shape = tuple(getattr(obs_space, "shape", ()) or ())
            obs_field = FieldSpec(dtype=obs_np_dtype, shape=obs_shape)

        # --- action side ---
        act_np_dtype = _np_dtype_for_space(act_space)
        if isinstance(act_space, gym.spaces.Discrete):
            act_shape: tuple[int, ...] = ()
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            act_shape = (len(act_space.nvec),)
        else:
            act_shape = tuple(getattr(act_space, "shape", ()) or ())

        output_spec = OutputSpec(
            step_index=FieldSpec(dtype=np.dtype(np.int64), shape=()),
            observation=obs_field,
            reward=FieldSpec(dtype=np.dtype(np.float32), shape=()),
            done=FieldSpec(dtype=np.dtype(np.int64), shape=()),
            episode_index=FieldSpec(dtype=int, shape=()),
            task_index=FieldSpec(dtype=int, shape=()),
        )
        input_spec = InputSpec(action=FieldSpec(dtype=act_np_dtype, shape=act_shape))
        return single_channel, obs_dtypes, output_spec, input_spec

    def _action_array(self, value: Any, *, dtype: np.dtype) -> np.ndarray:
        arr = np.asarray(value, dtype=dtype).flatten()
        if arr.size == 1:
            return np.array(arr.item(), dtype=dtype)
        return np.asarray(value, dtype=dtype)

    def sample_random_input(self) -> dict:
        """Sample a random action as a ``dict`` with a flat ``"action"`` key."""
        raw = self._env.action_space.sample()
        act_dtype = cast(np.dtype, self._input_spec.action.dtype)
        return {ACTION_KEY: self._action_array(raw, dtype=act_dtype)}

    def _require_input(self, input_dict: Any) -> np.ndarray:
        """Extract and validate the ``"action"`` key from an input dict."""
        if not isinstance(input_dict, dict):
            raise ValueError(
                f"input must be a dict with an '{ACTION_KEY}' entry, "
                f"got {type(input_dict).__name__}."
            )
        if ACTION_KEY not in input_dict:
            raise ValueError(
                f"input must contain the '{ACTION_KEY}' key; "
                f"got keys {sorted(input_dict.keys())}."
            )
        value = cast(Any, input_dict)[ACTION_KEY]
        return np.asarray(value)

    def _prepare_action(self, action_np: np.ndarray) -> Any:
        """Convert a numpy action array to the format expected by ``gym.Env.step``."""
        space = self._env.action_space
        if isinstance(space, gym.spaces.Discrete):
            return int(np.asarray(action_np).reshape(-1)[0])
        if isinstance(space, gym.spaces.MultiDiscrete):
            dtype = getattr(space, "dtype", np.int64)
            return np.asarray(action_np, dtype=dtype).reshape(-1)
        dtype = getattr(space, "dtype", None)
        arr = np.asarray(action_np, dtype=dtype) if dtype is not None else np.asarray(action_np)
        return arr.reshape(getattr(space, "shape", ()) or ())

    def _reward_array(self, raw_reward: Any) -> np.ndarray:
        return np.asarray(raw_reward, dtype=np.float32)

    def _obs_entry(self, obs: Any) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Build observation field(s) from a single-env observation."""
        if isinstance(obs, dict):
            return {OBS_KEY: {k: np.asarray(v) for k, v in obs.items()}}
        channel = cast(str, self._obs_channel)
        return {channel: np.asarray(obs)}

    def _reset_options_for_boundary(self, *, task_start: bool) -> dict[str, Any]:
        options = dict(self._episode_reset_options)
        if task_start:
            options.update(self._task_reset_options)
        return options

    def _do_reset(self, *, task_start: bool) -> tuple[dict, None, None]:
        """Call env.reset() and return the reset-frame output; no metric results."""
        reset_options = self._reset_options_for_boundary(task_start=task_start)
        if reset_options:
            obs, info = self._env.reset(options=reset_options)
        else:
            obs, info = self._env.reset()
        self._step_index = 0
        self._episode_cum_reward = 0.0

        output: dict = {
            STEP_INDEX_KEY: np.array(0, dtype=np.int64),
            "reward": np.array(self._reset_reward, dtype=np.float32),
            "done": np.array(DONE_RUNNING, dtype=np.int64),
            "episode_index": self._episode_index,
            "task_index": self._task_index,
        }
        output.update(self._obs_entry(obs))

        output["info"] = info

        return output, None, None

    def step(
        self, input_dict: dict
    ) -> tuple[dict, tuple[float, float] | None, tuple[float, float] | None]:
        """Step this env instance; return ``(output, episode_result, task_result)``.

        ``episode_result`` is ``(cum_reward, length)`` when the episode ended on this
        step, or ``None`` otherwise (including reset frames). ``task_result`` uses the
        same shape when the task ended on this step (``done`` 3/4), or ``None``.
        """
        if self._needs_initial_reset:
            self._needs_initial_reset = False
            return self._do_reset(task_start=True)

        if self._autoreset_pending:
            self._autoreset_pending = False
            task_start = False
            if self._task_done_pending:
                self._task_done_pending = False
                self._task_index += 1
                self._task_episode_count = 0
                self._episode_index = 0
                task_start = True
            else:
                self._episode_index += 1
                self._task_episode_count += 1
            return self._do_reset(task_start=task_start)

        # Regular step — validate and unpack input
        action_np = self._require_input(input_dict)
        action = self._prepare_action(action_np)
        obs, raw_reward, terminated, truncated, info = self._env.step(action)

        # Track raw cumulative reward (unscaled) for metrics
        raw_reward_f = float(raw_reward)
        self._episode_cum_reward += raw_reward_f

        self._step_index += 1

        # Determine done code — codes 3/4 fire when this episode is the last in the task.
        # episodes_per_task == 0 means unlimited: task boundary never fires automatically.
        task_done = self._episodes_per_task > 0 and (
            self._task_episode_count + 1 == self._episodes_per_task
        )
        if terminated:
            done = DONE_TASK_TERMINATED if task_done else DONE_EPISODE_TERMINATED
        elif truncated:
            done = DONE_TASK_TRUNCATED if task_done else DONE_EPISODE_TRUNCATED
        else:
            done = DONE_RUNNING

        output: dict = {
            STEP_INDEX_KEY: np.array(self._step_index, dtype=np.int64),
            "reward": self._reward_array(raw_reward),
            "done": np.array(done, dtype=np.int64),
            "episode_index": self._episode_index,
            "task_index": self._task_index,
        }
        output.update(self._obs_entry(obs))

        output["info"] = info

        episode_result: tuple[float, float] | None
        task_result: tuple[float, float] | None
        if done != DONE_RUNNING:
            episode_result = (self._episode_cum_reward, float(self._step_index))
            self._task_cum_reward += self._episode_cum_reward
            self._task_cum_length += float(self._step_index)
            if task_done:
                task_result = (self._task_cum_reward, self._task_cum_length)
                self._task_cum_reward = 0.0
                self._task_cum_length = 0.0
            else:
                task_result = None
            self._autoreset_pending = True
            self._task_done_pending = task_done
        else:
            episode_result = None
            task_result = None

        return output, episode_result, task_result

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
    reset frame (``done=0``, ``step_index=0``, input ignored). After each episode or
    task ends, the next ``step`` is also a reset frame. There is no public
    ``reset()`` — including at task boundaries — so training loops stay a single
    ``step()`` stream.

    Episode statistics are accumulated automatically in :attr:`metrics`
    (:class:`Metrics`). Call ``env.metrics.clear()`` to wipe accumulated
    data between evaluation runs.

    Every output dict contains:
        step_index (int64 array)  — step index within the episode (0-based; resets on episode restart)
        observation (array/dict)  — the observation emitted by the env
        reward (float32 array)    — raw env reward from the underlying Gymnasium step
        done (int64 array)        — 0=running, 1=episode terminated, 2=episode truncated,
                                    3=task terminated, 4=task truncated
        episode_index (int)       — episode counter within the current task (resets at task end)
        task_index (int)          — task counter
        info (dict)               — Gymnasium info dict from the underlying env step/reset
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
    def output_spec(self) -> OutputSpec:
        """Output contract for this env."""
        return self._env_instance.output_spec

    @property
    def input_spec(self) -> InputSpec:
        """Input contract for this env."""
        return self._env_instance.input_spec

    @property
    def action_space(self) -> gym.Space:
        """The underlying Gymnasium action space."""
        return self._env_instance._env.action_space

    @property
    def observation_space(self) -> gym.Space:
        """The underlying Gymnasium observation space."""
        return self._env_instance._env.observation_space

    def reset(self, **_kwargs: Any) -> None:
        """Not supported — episode and task transitions happen inside ``step()``."""
        raise NotImplementedError(
            "SingleEnv does not support public reset(); call step() only. "
            "Episode and task boundaries are handled internally and appear as "
            "reset frames in the step() output — including when a task ends."
        )

    def sample_random_input(self) -> dict:
        """Sample a random action dict for this env. Pass directly to ``step()``."""
        return self._env_instance.sample_random_input()

    def step(self, input: dict) -> dict:
        """Step the env and return one output dict.

        On the first call and on any call immediately after an episode ends, the
        input is ignored and a reset frame is returned instead.

        Completed-episode statistics are recorded automatically into :attr:`metrics`.
        Call ``env.metrics.clear()`` to reset between runs.
        """
        output, episode_result, task_result = self._env_instance.step(input)
        if episode_result is not None:
            cum_reward, length = episode_result
            self._metrics._record(cum_reward, length)
        if task_result is not None:
            task_cum_reward, task_length = task_result
            self._metrics._record_task(task_cum_reward, task_length)
        return output

    def render(self) -> list:
        """Return rendered frames from this env instance."""
        return self._env_instance.render()

    def close(self) -> None:
        """Close the underlying env."""
        self._env_instance.close()


class GroupMetrics:
    """Live read-through view over :class:`Metrics` instances of a :class:`GroupEnv`.

    Stores no episode or task data of its own — all reads delegate to each
    constituent :class:`SingleEnv`'s metrics. Multiple :class:`GroupEnv` instances
    may point to overlapping sets of :class:`SingleEnv` objects without any data
    conflicts.

    Attributes
    ----------
    episode_cum_rewards:
        Per-env list of raw cumulative rewards. ``episode_cum_rewards[i]`` is the
        list from ``envs[i].metrics.episode_cum_rewards``.
    episode_lengths:
        Per-env list of episode step counts. ``episode_lengths[i]`` is the list
        from ``envs[i].metrics.episode_lengths``.
    task_cum_rewards:
        Per-env list of task reward sums. ``task_cum_rewards[i]`` is the list from
        ``envs[i].metrics.task_cum_rewards``.
    task_lengths:
        Per-env list of task length sums. ``task_lengths[i]`` is the list from
        ``envs[i].metrics.task_lengths``.
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

    @property
    def task_cum_rewards(self) -> list[list[float]]:
        """Per-env task reward sums, read live from each env's metrics."""
        return [e.metrics.task_cum_rewards for e in self._envs]

    @property
    def task_lengths(self) -> list[list[float]]:
        """Per-env task length sums, read live from each env's metrics."""
        return [e.metrics.task_lengths for e in self._envs]

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

    ``step`` and ``sample_random_input`` use a flat list indexed by position.
    ``inputs[i]`` is the input dict for the i-th env. ``step`` returns a flat
    ``list[dict]`` — one per env.

    With ``max_threads=0`` (default), ``step`` runs every env on the calling thread.
    With ``max_threads > 0``, ``step`` distributes envs across up to ``max_threads``
    worker threads (capped at ``num_envs``) and preserves output order.
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
    def output_specs(self) -> list[OutputSpec]:
        """One :class:`OutputSpec` per env instance."""
        return [e.output_spec for e in self._envs]

    @property
    def input_specs(self) -> list[InputSpec]:
        """One :class:`InputSpec` per env instance."""
        return [e.input_spec for e in self._envs]

    @property
    def action_space(self) -> gym.spaces.Tuple:
        """Gymnasium tuple action space, one subspace per env instance."""
        return gym.spaces.Tuple(tuple(e.action_space for e in self._envs))

    @property
    def observation_space(self) -> gym.spaces.Tuple:
        """Gymnasium tuple observation space, one subspace per env instance."""
        return gym.spaces.Tuple(tuple(e.observation_space for e in self._envs))

    def sample_random_input(self) -> list[dict]:
        """Sample random inputs for every env instance.

        Returns a flat ``list[dict]`` — one dict per env. Pass directly to ``step()``.
        """
        return [e.sample_random_input() for e in self._envs]

    def step(self, inputs: list[dict]) -> list[dict]:
        """Step all env instances and return outputs.

        ``inputs[i]`` is the input dict for env instance ``i``. Returns a flat
        ``list[dict]`` — one output dict per env. On the first call and on any call
        immediately after an episode ends, the corresponding input is ignored and a
        reset frame is returned instead.

        With ``max_threads=0``, steps run sequentially on the calling thread. With
        ``max_threads > 0``, steps are distributed across worker threads; results stay
        index-aligned with ``inputs``.
        """
        if not isinstance(inputs, list):
            raise ValueError(
                f"inputs must be a list with one dict per env instance; got {type(inputs).__name__}."
            )
        if len(inputs) != self.num_envs:
            raise ValueError(
                f"inputs must contain exactly {self.num_envs} entries, got {len(inputs)}."
            )
        if self._executor is None:
            return [e.step(inp) for e, inp in zip(self._envs, inputs)]
        return list(self._executor.map(_step_env, self._envs, inputs))

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


def _step_env(env: SingleEnv, inp: dict) -> dict:
    """Step one env; module-level helper for :class:`ThreadPoolExecutor.map`."""
    return env.step(inp)
