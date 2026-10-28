"""Track observable rollout progress and completed episode/task statistics."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

StepOutput = tuple[Any, float, bool, bool, dict[str, Any]]


@dataclass(frozen=True)
class EpisodeSummary:
    """A completed episode, including its reset-frame reward.

    ``length`` excludes the initial/reset frame. ``episode_index`` counts
    ordinary episodes, or inner episodes within ``task_index`` for a repeated
    task. Both indices start at zero and are unaffected by clearing summaries.
    """

    episode_index: int
    reward: float
    length: int
    terminated: bool
    truncated: bool
    task_index: int | None = None


@dataclass(frozen=True)
class TaskSummary:
    """A completed repeated task, including rewards from all reset frames.

    ``length`` excludes the task's initial frame but includes subsequent inner
    reset frames. ``completed_episodes`` counts actual inner episode endings;
    ending a task midway through an episode does not complete that episode.
    """

    task_index: int
    reward: float
    length: int
    terminated: bool
    truncated: bool
    completed_episodes: int


@dataclass(frozen=True)
class _Frame:
    output: StepOutput
    episode_index: int
    step_index: int
    episode_start: bool
    cumulative_reward: float


class RolloutTracker:
    """One ordinary MDP/POMDP output stream and its completed episodes.

    Pass each completed output to :meth:`update`, including reset frames. A
    tracker can consume environment outputs or synthetic decision frames.
    The tracker observes outputs; it never advances or resets an environment.

    ``output`` preserves the complete Gymnasium tuple. Observations and info
    are deep-copied on update so later caller/env mutations cannot change the
    recorded frame. ``cumulative_reward`` includes reset-frame rewards.
    """

    def __init__(self) -> None:
        self.output: StepOutput | None = None
        self.observation: Any = None
        self.reward = 0.0
        self.info: dict[str, Any] = {}
        self.terminated = False
        self.truncated = False
        self.episode_index = -1
        self.step_index = -1
        self.episode_start = False
        self.action_requested = False
        self.cumulative_reward = 0.0
        self._episodes: list[EpisodeSummary] = []

    @property
    def ended(self) -> bool:
        """Whether the latest outer episode ended by either boundary flag."""
        return self.terminated or self.truncated

    @property
    def episodes(self) -> tuple[EpisodeSummary, ...]:
        """Completed episodes since the last :meth:`clear` call."""
        return tuple(self._episodes)

    def clear(self) -> None:
        """Clear completed summaries, preserving the current rollout."""
        self._episodes.clear()

    def _prepare(self, *, output: StepOutput) -> _Frame:
        episode_start = self.output is None or self.ended
        if not isinstance(output, tuple) or len(output) != 5:
            raise ValueError("output must be a Gymnasium five-element tuple")
        observation, reward, terminated, truncated, info = deepcopy(output)
        if not isinstance(info, dict):
            raise TypeError("output info must be a dict")
        reward = float(reward)
        copied_output = observation, reward, bool(terminated), bool(truncated), info
        return _Frame(
            output=copied_output,
            episode_index=self.episode_index + int(episode_start),
            step_index=0 if episode_start else self.step_index + 1,
            episode_start=episode_start,
            cumulative_reward=reward if episode_start else self.cumulative_reward + reward,
        )

    def _commit(self, frame: _Frame) -> None:
        self.output = frame.output
        self.observation, self.reward, self.terminated, self.truncated, self.info = frame.output
        self.episode_index = frame.episode_index
        self.step_index = frame.step_index
        self.episode_start = frame.episode_start
        self.cumulative_reward = frame.cumulative_reward
        self.action_requested = not self.ended

    def update(self, *, output: StepOutput) -> None:
        """Record a frame, rejecting invalid output atomically.

        The first frame and the frame after either ending flag are reset
        frames with ``step_index=0``. Other frames advance the index by one.
        Feed every frame in order; the stream carries no independent index
        with which to detect missing frames or an unexpected external reset.
        """
        frame = self._prepare(output=output)
        _, _, terminated, truncated, _ = frame.output
        summary = (
            EpisodeSummary(
                episode_index=frame.episode_index,
                reward=frame.cumulative_reward,
                length=frame.step_index,
                terminated=terminated,
                truncated=truncated,
            )
            if terminated or truncated
            else None
        )
        self._commit(frame)
        if summary is not None:
            self._episodes.append(summary)


class RepeatTaskTracker(RolloutTracker):
    """Track inner episodes and outer repeated tasks separately.

    Outputs must wrap their observation in a mapping with ``observation``,
    ``episode_terminated``, and ``episode_truncated`` keys. ``observation``
    exposes the underlying value; ``output`` retains the full wrapper tuple.
    No dependency on a particular repeated-task environment is required.

    The inherited ``episode_index``, ``step_index``, ``ended``, and
    ``cumulative_reward`` describe the outer task. ``repeat_*`` fields describe
    its current inner episode. :attr:`episodes` contains inner summaries and
    :attr:`tasks` contains outer summaries. An outer ending alone never
    fabricates an inner ending. ``action_requested`` is false at either kind
    of boundary, including when the next frame will reset an inner episode.
    """

    def __init__(self) -> None:
        super().__init__()
        self.repeat_episode_index = -1
        self.repeat_episode_start = False
        self.repeat_episode_terminated = False
        self.repeat_episode_truncated = False
        self.repeat_step_index = -1
        self.repeat_cumulative_reward = 0.0
        self._tasks: list[TaskSummary] = []
        self._task_completed_episodes = 0

    @property
    def tasks(self) -> tuple[TaskSummary, ...]:
        """Completed tasks since the last :meth:`clear` call."""
        return tuple(self._tasks)

    def clear(self) -> None:
        """Clear completed episodes/tasks while preserving active totals."""
        super().clear()
        self._tasks.clear()

    def update(self, *, output: StepOutput) -> None:
        """Record an outer frame and any actual inner/outer completions."""
        frame = self._prepare(output=output)
        wrapped_observation, reward, terminated, truncated, _ = frame.output
        if not isinstance(wrapped_observation, Mapping):
            raise TypeError("Repeated-task observations must be a mapping")
        required = {"observation", "episode_terminated", "episode_truncated"}
        if not required.issubset(wrapped_observation):
            raise ValueError(
                "Repeated-task observations require observation, episode_terminated, "
                "and episode_truncated keys"
            )
        observation = wrapped_observation["observation"]
        repeat_terminated = bool(wrapped_observation["episode_terminated"])
        repeat_truncated = bool(wrapped_observation["episode_truncated"])
        repeat_start = (
            frame.episode_start
            or self.repeat_episode_terminated
            or self.repeat_episode_truncated
        )
        repeat_episode_index = (
            0 if frame.episode_start else self.repeat_episode_index + int(repeat_start)
        )
        repeat_step_index = 0 if repeat_start else self.repeat_step_index + 1
        repeat_reward = reward if repeat_start else self.repeat_cumulative_reward + reward
        completed_episodes = 0 if frame.episode_start else self._task_completed_episodes
        episode_summary = None
        if repeat_terminated or repeat_truncated:
            completed_episodes += 1
            episode_summary = EpisodeSummary(
                episode_index=repeat_episode_index,
                reward=repeat_reward,
                length=repeat_step_index,
                terminated=repeat_terminated,
                truncated=repeat_truncated,
                task_index=frame.episode_index,
            )
        task_summary = (
            TaskSummary(
                task_index=frame.episode_index,
                reward=frame.cumulative_reward,
                length=frame.step_index,
                terminated=terminated,
                truncated=truncated,
                completed_episodes=completed_episodes,
            )
            if terminated or truncated
            else None
        )

        # Validate and calculate everything before advancing either level.
        self._commit(frame)
        self.observation = observation
        self.repeat_episode_index = repeat_episode_index
        self.repeat_episode_start = repeat_start
        self.repeat_episode_terminated = repeat_terminated
        self.repeat_episode_truncated = repeat_truncated
        self.repeat_step_index = repeat_step_index
        self.repeat_cumulative_reward = repeat_reward
        self._task_completed_episodes = completed_episodes
        self.action_requested = not (self.ended or repeat_terminated or repeat_truncated)
        if episode_summary is not None:
            self._episodes.append(episode_summary)
        if task_summary is not None:
            self._tasks.append(task_summary)
