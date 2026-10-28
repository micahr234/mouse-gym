from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from mouse_gym.tracker import EpisodeSummary, RepeatTaskTracker, RolloutTracker, TaskSummary


def ordinary_output(
    *, reward: float = 0.0, terminated: bool = False, truncated: bool = False
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    return [7], reward, terminated, truncated, {}


def repeated_output(
    *,
    reward: float = 0.0,
    inner_terminated: bool = False,
    inner_truncated: bool = False,
    terminated: bool = False,
    truncated: bool = False,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    return (
        {
            "observation": [7],
            "episode_terminated": inner_terminated,
            "episode_truncated": inner_truncated,
        },
        reward,
        terminated,
        truncated,
        {},
    )


@pytest.mark.parametrize("tracker_type", [RolloutTracker, RepeatTaskTracker])
def test_tracker_starts_without_an_output_or_action_request(tracker_type):
    tracker = tracker_type()
    assert tracker.output is None and tracker.observation is None
    assert tracker.episode_index == tracker.step_index == -1
    assert tracker.reward == tracker.cumulative_reward == 0.0
    assert not tracker.ended and not tracker.episode_start
    assert not tracker.action_requested
    assert tracker.episodes == ()
    if isinstance(tracker, RepeatTaskTracker):
        assert tracker.repeat_episode_index == tracker.repeat_step_index == -1
        assert not tracker.repeat_episode_start
        assert tracker.repeat_cumulative_reward == 0.0
        assert tracker.tasks == ()


@pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True), (True, True)])
def test_ordinary_boundaries_keep_both_flags_and_include_reset_rewards(terminated, truncated):
    tracker = RolloutTracker()
    tracker.update(output=ordinary_output(reward=2.0))
    assert tracker.episode_index == tracker.step_index == 0
    assert tracker.episode_start and tracker.action_requested
    tracker.update(output=ordinary_output(reward=3.0))
    assert tracker.step_index == 1 and not tracker.episode_start
    tracker.update(output=ordinary_output(reward=5.0, terminated=terminated, truncated=truncated))
    assert tracker.ended and not tracker.action_requested
    assert tracker.cumulative_reward == 10.0
    assert tracker.episodes == (EpisodeSummary(0, 10.0, 2, terminated, truncated),)
    tracker.update(output=ordinary_output(reward=11.0))
    assert tracker.episode_index == 1 and tracker.step_index == 0
    assert tracker.cumulative_reward == 11.0
    assert tracker.episode_start and tracker.action_requested and not tracker.ended


def test_repeated_tracker_tracks_inner_episodes_and_outer_tasks():
    tracker = RepeatTaskTracker()
    transitions = [
        (repeated_output(reward=2), (0, 0, 0, True, True)),
        (repeated_output(reward=3), (1, 0, 1, False, True)),
        (repeated_output(reward=5, inner_terminated=True), (2, 0, 2, False, False)),
        (repeated_output(reward=7), (3, 1, 0, True, True)),
        (repeated_output(reward=11), (4, 1, 1, False, True)),
        (
            repeated_output(reward=13, inner_truncated=True, terminated=True, truncated=True),
            (5, 1, 2, False, False),
        ),
    ]
    for output, expected in transitions:
        tracker.update(output=output)
        assert (
            tracker.step_index,
            tracker.repeat_episode_index,
            tracker.repeat_step_index,
            tracker.repeat_episode_start,
            tracker.action_requested,
        ) == expected
        assert tracker.episode_index == 0
        assert tracker.observation == [7]
        assert tracker.output is not None and tracker.output[0] == output[0]
    assert tracker.episodes == (
        EpisodeSummary(0, 10.0, 2, True, False, task_index=0),
        EpisodeSummary(1, 31.0, 2, False, True, task_index=0),
    )
    assert tracker.tasks == (TaskSummary(0, 41.0, 5, True, True, completed_episodes=2),)
    assert tracker.repeat_cumulative_reward == 31.0 and tracker.cumulative_reward == 41.0
    tracker.update(output=repeated_output(reward=17))
    assert tracker.episode_index == 1 and tracker.repeat_episode_index == 0
    assert tracker.step_index == tracker.repeat_step_index == 0
    assert tracker.episode_start and tracker.repeat_episode_start
    assert tracker.repeat_cumulative_reward == tracker.cumulative_reward == 17.0
    assert tracker.action_requested


@pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True), (True, True)])
def test_inner_ending_preserves_both_flags_without_ending_task(terminated, truncated):
    tracker = RepeatTaskTracker()
    tracker.update(output=repeated_output())
    tracker.update(
        output=repeated_output(reward=3, inner_terminated=terminated, inner_truncated=truncated)
    )
    assert tracker.repeat_episode_terminated is terminated
    assert tracker.repeat_episode_truncated is truncated
    assert tracker.episodes == (EpisodeSummary(0, 3.0, 1, terminated, truncated, task_index=0),)
    assert not tracker.ended and not tracker.action_requested
    assert tracker.tasks == ()
    tracker.update(output=repeated_output(reward=5))
    assert tracker.repeat_episode_index == 1 and tracker.repeat_step_index == 0
    assert tracker.repeat_episode_start and tracker.action_requested
    assert tracker.repeat_cumulative_reward == 5.0 and tracker.cumulative_reward == 8.0


def test_task_can_end_mid_episode_without_fabricating_an_inner_summary():
    tracker = RepeatTaskTracker()
    tracker.update(output=repeated_output(reward=2))
    tracker.update(output=repeated_output(reward=3, terminated=True))
    assert tracker.ended and not tracker.action_requested
    assert not tracker.repeat_episode_terminated and not tracker.repeat_episode_truncated
    assert tracker.episodes == ()
    assert tracker.tasks == (TaskSummary(0, 5.0, 1, True, False, completed_episodes=0),)


def test_task_can_end_on_an_inner_reset_frame():
    tracker = RepeatTaskTracker()
    tracker.update(output=repeated_output(reward=2))
    tracker.update(output=repeated_output(reward=3, inner_terminated=True))
    tracker.update(output=repeated_output(reward=7, truncated=True))
    assert tracker.repeat_episode_start and tracker.repeat_step_index == 0
    assert tracker.ended and not tracker.action_requested
    assert tracker.episodes == (EpisodeSummary(0, 5.0, 1, True, False, task_index=0),)
    assert tracker.tasks == (TaskSummary(0, 12.0, 2, False, True, completed_episodes=1),)


def test_task_can_end_on_its_initial_frame():
    tracker = RepeatTaskTracker()
    tracker.update(output=repeated_output(reward=2, terminated=True))
    assert tracker.episode_start and tracker.repeat_episode_start
    assert tracker.episodes == ()
    assert tracker.tasks == (TaskSummary(0, 2.0, 0, True, False, completed_episodes=0),)


def test_clear_preserves_current_ordinary_progress_and_summary_snapshots():
    tracker = RolloutTracker()
    tracker.update(output=ordinary_output())
    tracker.update(output=ordinary_output(reward=2, terminated=True))
    completed = tracker.episodes
    tracker.update(output=ordinary_output(reward=3))
    tracker.update(output=ordinary_output(reward=5))
    tracker.clear()
    assert tracker.episodes == ()
    assert completed == (EpisodeSummary(0, 2.0, 1, True, False),)
    assert tracker.episode_index == 1 and tracker.step_index == 1
    assert tracker.cumulative_reward == 8.0 and tracker.action_requested
    tracker.update(output=ordinary_output(reward=7, truncated=True))
    assert tracker.episodes == (EpisodeSummary(1, 15.0, 2, False, True),)


def test_clear_preserves_inner_rewards_and_completed_episode_task_count():
    tracker = RepeatTaskTracker()
    tracker.update(output=repeated_output(reward=2))
    tracker.update(output=repeated_output(reward=3, inner_terminated=True))
    tracker.update(output=repeated_output(reward=5))
    tracker.clear()
    assert tracker.episodes == tracker.tasks == ()
    assert tracker.repeat_episode_index == 1 and tracker.repeat_step_index == 0
    assert tracker.repeat_cumulative_reward == 5.0 and tracker.cumulative_reward == 10.0
    tracker.update(output=repeated_output(reward=7, inner_truncated=True, terminated=True))
    assert tracker.episodes == (EpisodeSummary(1, 12.0, 1, False, True, task_index=0),)
    assert tracker.tasks == (TaskSummary(0, 17.0, 3, True, False, completed_episodes=2),)
    tracker.clear()
    tracker.update(output=repeated_output(reward=11))
    tracker.update(output=repeated_output(reward=13, truncated=True))
    assert tracker.episode_index == 1
    assert tracker.episodes == ()
    assert tracker.tasks == (TaskSummary(1, 24.0, 1, False, True, completed_episodes=0),)


class InvalidFlag:
    def __bool__(self):
        raise ValueError("invalid flag")


@pytest.mark.parametrize(
    "bad_output",
    [
        ([1], 0.0, False, False),
        [[1], 0.0, False, False, {}],
        ([1], object(), False, False, {}),
        ([1], 1.0, InvalidFlag(), False, {}),
        ([1], 1.0, False, InvalidFlag(), {}),
        ([1], 1.0, False, False, []),
    ],
)
@pytest.mark.parametrize("at_boundary", [False, True])
def test_invalid_ordinary_output_does_not_partially_advance_tracker(bad_output, at_boundary):
    tracker = RolloutTracker()
    tracker.update(output=ordinary_output(reward=2))
    if at_boundary:
        tracker.update(output=ordinary_output(reward=3, terminated=True))
    before = deepcopy(tracker.__dict__)
    with pytest.raises((ValueError, TypeError)):
        tracker.update(output=bad_output)
    assert tracker.__dict__ == before
    tracker.update(output=ordinary_output(reward=5))
    assert tracker.step_index == (0 if at_boundary else 1)
    assert tracker.cumulative_reward == (5.0 if at_boundary else 7.0)


@pytest.mark.parametrize(
    "bad_observation",
    [
        [1],
        {"observation": [1]},
        {"observation": [1], "episode_terminated": False},
        {"observation": [1], "episode_terminated": False, "episode_truncated": InvalidFlag()},
    ],
)
def test_invalid_repeated_output_leaves_outer_and_inner_progress_unchanged(bad_observation):
    tracker = RepeatTaskTracker()
    tracker.update(output=repeated_output(reward=2))
    tracker.update(output=repeated_output(reward=3, inner_terminated=True))
    before = deepcopy(tracker.__dict__)
    with pytest.raises((ValueError, TypeError)):
        tracker.update(output=(bad_observation, 7.0, True, False, {}))
    assert tracker.__dict__ == before
    tracker.update(output=repeated_output(reward=5))
    assert tracker.repeat_episode_index == 1 and tracker.repeat_episode_start
    assert tracker.step_index == 2 and tracker.cumulative_reward == 10.0


@pytest.mark.parametrize("tracker_type", [RolloutTracker, RepeatTaskTracker])
def test_output_snapshot_is_independent_of_nested_caller_mutation(tracker_type):
    tracker = tracker_type()
    observation = {"values": [1, 2]}
    wrapped: dict[str, Any] = (
        {"observation": observation, "episode_terminated": False, "episode_truncated": False}
        if isinstance(tracker, RepeatTaskTracker)
        else observation
    )
    info = {"nested": {"values": [3, 4]}}
    tracker.update(output=(wrapped, 2.0, False, False, info))
    observation["values"][0] = 99
    info["nested"]["values"][0] = 88
    if isinstance(tracker, RepeatTaskTracker):
        wrapped["episode_terminated"] = True
    assert tracker.observation == {"values": [1, 2]}
    assert tracker.info == {"nested": {"values": [3, 4]}}
    assert tracker.output is not None and tracker.output[0] != wrapped
    assert tracker.action_requested


def test_deepcopied_tracker_branches_have_independent_progress_and_history():
    tracker = RepeatTaskTracker()
    tracker.update(output=repeated_output(reward=2))
    tracker.update(output=repeated_output(reward=3, inner_terminated=True))
    branch = deepcopy(tracker)
    branch.observation[0] = 99
    branch.clear()
    branch.update(output=repeated_output(reward=5))
    branch.update(output=repeated_output(reward=7, inner_truncated=True, truncated=True))
    assert tracker.observation == [7]
    assert tracker.step_index == 1 and tracker.repeat_episode_index == 0
    assert tracker.cumulative_reward == 5.0
    assert tracker.episodes == (EpisodeSummary(0, 5.0, 1, True, False, task_index=0),)
    assert tracker.tasks == ()
    assert branch.tasks == (TaskSummary(0, 17.0, 3, False, True, completed_episodes=2),)


@pytest.mark.parametrize(
    "summary",
    [EpisodeSummary(0, 2.0, 1, True, False), TaskSummary(0, 2.0, 1, True, False, 1)],
)
def test_completed_summaries_are_immutable(summary):
    with pytest.raises(FrozenInstanceError):
        summary.reward = 7.0
