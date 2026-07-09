"""Smoke tests for mouse-gym — offline, no external downloads."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest

from mouse_gym import EnvConfig, FieldSpec, InputSpec, Metrics, OutputSpec, make_env, make_group_env
from mouse_gym.format import (
    DONE_EPISODE_TERMINATED,
    DONE_EPISODE_TRUNCATED,
    DONE_TASK_TERMINATED,
    DONE_TASK_TRUNCATED,
)


def _rollout(env, steps: int = 5) -> list:
    """Roll out a GroupEnv for ``steps`` steps; returns the final list[dict]."""
    outputs = env.step(env.sample_random_input())
    for _ in range(steps - 1):
        outputs = env.step(env.sample_random_input())
    return outputs


def _rollout_single(env, steps: int = 5) -> dict:
    """Roll out a SingleEnv for ``steps`` steps; returns the final output dict."""
    output = env.step(env.sample_random_input())
    for _ in range(steps - 1):
        output = env.step(env.sample_random_input())
    return output


def test_cartpole_step_contract() -> None:
    cfgs = [
        EnvConfig(
            id="CartPole-v1",
            name=f"train-cartpole_{i}",
            reset_seed=i,
            episodes_per_task=5,
        )
        for i in range(3)
    ]
    env = make_group_env(cfgs)
    try:
        outputs = _rollout(env)
        assert len(outputs) == 3
        assert env.names[0] == "train-cartpole_0"
        assert env.names == ("train-cartpole_0", "train-cartpole_1", "train-cartpole_2")
        sampled = env.sample_random_input()
        assert "action" in sampled[0]
        assert sampled[0]["action"].ndim == 0
        for r in outputs:
            assert set(r.keys()) >= {
                "time",
                "observation",
                "reward",
                "done",
                "episode_index",
                "task_index",
            }
            assert "id" not in r
            assert "name" not in r
            assert "action" not in r
        for per_env in env.metrics.episode_cum_rewards:
            assert all(isinstance(v, float) for v in per_env)
    finally:
        env.close()


def test_group_env_exposes_gym_tuple_spaces() -> None:
    env = make_group_env(
        [
            EnvConfig(id="CartPole-v1", reset_seed=0, episodes_per_task=5),
            EnvConfig(id="CartPole-v1", reset_seed=1, episodes_per_task=5),
        ]
    )
    try:
        assert isinstance(env.action_space, gym.spaces.Tuple)
        assert isinstance(env.observation_space, gym.spaces.Tuple)
        assert len(env.action_space.spaces) == 2
        assert len(env.observation_space.spaces) == 2
        assert isinstance(env.action_space.spaces[0], gym.spaces.Discrete)
        assert isinstance(env.observation_space.spaces[0], gym.spaces.Box)
        assert not hasattr(env, "action_spaces")
    finally:
        env.close()


def test_output_spec_and_input_spec_cartpole() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
    )
    env = make_env(cfg)
    try:
        ospec = env.output_spec
        ispec = env.input_spec

        assert isinstance(ospec, OutputSpec)
        assert isinstance(ispec, InputSpec)

        assert isinstance(ospec.observation, FieldSpec)
        assert ospec.observation.dtype == np.dtype(np.float32)
        assert ospec.observation.shape == (4,)

        assert ospec.time.dtype == np.dtype(np.int64)
        assert ospec.time.shape == ()
        assert ospec.reward.dtype == np.dtype(np.float32)
        assert ospec.done.dtype == np.dtype(np.int64)
        assert ospec.episode_index.dtype == int
        assert ospec.task_index.dtype == int
        assert not hasattr(ospec, "q_star")
        assert not hasattr(ospec, "ns_params")

        assert isinstance(ispec.action, FieldSpec)
        assert ispec.action.dtype == np.dtype(np.int64)
        assert ispec.action.shape == ()
    finally:
        env.close()


def test_pendulum_continuous_step_contract() -> None:
    env = make_group_env(
        [
            EnvConfig(id="Pendulum-v1", reset_seed=0, episodes_per_task=5),
            EnvConfig(id="Pendulum-v1", reset_seed=1, episodes_per_task=5),
        ]
    )
    try:
        assert env.input_specs[0].action.shape == (1,)
        sampled = env.sample_random_input()
        action = sampled[0]
        assert "action" in action
        assert action["action"].dtype == np.float32
        assert action["action"].ndim == 0

        assert env.input_specs[0].action.dtype == np.dtype(np.float32)
        obs_spec = env.output_specs[0].observation
        assert isinstance(obs_spec, FieldSpec)
        assert obs_spec.dtype == np.dtype(np.float32)

        outputs = _rollout(env)
        assert len(outputs) == 2
        for r in outputs:
            assert "observation" in r
            assert "action" not in r
    finally:
        env.close()


def test_action_input_contract_is_enforced() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
    )
    env = make_env(cfg)
    try:
        env.step(env.sample_random_input())
        with pytest.raises(ValueError, match="must be a dict"):
            env.step(cast(Any, [np.array(0)]))
        with pytest.raises(ValueError, match="action"):
            env.step({"wrong": np.array(0)})
    finally:
        env.close()


def test_single_env_reset_is_not_implemented() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
    )
    env = make_env(cfg)
    try:
        with pytest.raises(NotImplementedError, match="does not support public reset"):
            env.reset()
    finally:
        env.close()


def test_dict_obs_dtype_follows_space_not_key_name() -> None:
    class DictObsEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Dict(
                {
                    "pos": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "tile": gym.spaces.Box(low=0, high=9, shape=(1,), dtype=np.int32),
                }
            )
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {"pos": np.zeros(2, np.float32), "tile": np.array([3], np.int32)}, {}

        def step(self, action):
            return (
                {"pos": np.zeros(2, np.float32), "tile": np.array([3], np.int32)},
                0.0,
                False,
                False,
                {},
            )

    cfg = EnvConfig(
        reset_seed=0,
        episodes_per_task=5,
        env_fn=DictObsEnv,
    )
    env = make_env(cfg)
    try:
        output = _rollout_single(env, steps=2)
        assert "pos" not in output
        assert "tile" not in output
        obs = output["observation"]
        assert isinstance(obs, dict)
        assert obs["pos"].dtype == np.float32
        assert obs["tile"].dtype == np.int32

        ospec = env.output_spec
        assert isinstance(ospec.observation, dict)
        assert ospec.observation["pos"].dtype == np.dtype(np.float32)
        assert ospec.observation["tile"].dtype == np.dtype(np.int32)
    finally:
        env.close()


def test_info_keys_passthrough() -> None:
    class InfoEmittingEnv(gym.Env):
        observation_space = gym.spaces.Discrete(4)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return 0, {"foo": 1, "q_star": np.array([1.0, 0.0], dtype=np.float64)}

        def step(self, action):
            return 1, 0.0, False, False, {"foo": 2, "q_star": np.array([0.0, 1.0], dtype=np.float64)}

    cfg = EnvConfig(
        reset_seed=0,
        episodes_per_task=5,
        env_fn=InfoEmittingEnv,
    )
    env = make_env(cfg)
    try:
        reset_frame = env.step(env.sample_random_input())
        step_frame = env.step(env.sample_random_input())
        assert reset_frame["info"]["foo"] == 1
        assert step_frame["info"]["foo"] == 2
        assert np.asarray(reset_frame["info"]["q_star"]).tolist() == [1.0, 0.0]
        assert np.asarray(step_frame["info"]["q_star"]).tolist() == [0.0, 1.0]
    finally:
        env.close()


def test_action_space_can_be_seeded_for_random_inputs() -> None:
    def _sampled_actions(*, reset_seed: int, action_space_seed: int) -> list[int]:
        cfg = EnvConfig(
            id="CartPole-v1",
            reset_seed=reset_seed,
            episodes_per_task=5,
        )
        env = make_env(cfg)
        try:
            env.action_space.seed(action_space_seed)
            return [int(env.sample_random_input()["action"].item()) for _ in range(12)]
        finally:
            env.close()

    assert _sampled_actions(reset_seed=10, action_space_seed=20) == _sampled_actions(
        reset_seed=11, action_space_seed=20
    )
    assert _sampled_actions(reset_seed=10, action_space_seed=20) != _sampled_actions(
        reset_seed=10, action_space_seed=21
    )


def test_reset_seed_controls_internal_reset_stream() -> None:
    def _first_obs(*, reset_seed: int) -> np.ndarray:
        cfg = EnvConfig(id="CartPole-v1", reset_seed=reset_seed, episodes_per_task=5)
        env = make_env(cfg)
        try:
            return env.step(env.sample_random_input())["observation"]
        finally:
            env.close()

    assert np.array_equal(_first_obs(reset_seed=3), _first_obs(reset_seed=3))
    assert not np.array_equal(_first_obs(reset_seed=3), _first_obs(reset_seed=30))


def test_autoreset_frame_uses_reset_reward() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
    )
    env = make_env(cfg)
    try:
        output, _step = _roll_until_autoreset(env)
        assert output["time"].item() == 0
        assert output["reward"].item() == 0.0
        assert output["done"].item() == 0
        assert len(env.metrics.episode_cum_rewards) >= 1
    finally:
        env.close()


def test_initial_reset_frame_uses_reset_reward() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
        reset_reward=-1.0,
    )
    env = make_env(cfg)
    try:
        output = env.step(env.sample_random_input())
        assert output["time"].item() == 0
        assert "action" not in output
        assert output["reward"].item() == -1.0
        assert output["done"].item() == 0
        assert output["task_index"] == 0
        assert env.metrics.episode_cum_rewards == []
    finally:
        env.close()


def _roll_until_autoreset(env, *, max_steps: int = 500) -> tuple[dict, int]:
    output = env.step(env.sample_random_input())
    for step in range(1, max_steps):
        prev_time = output["time"].item()
        output = env.step(env.sample_random_input())
        if output["time"].item() == 0 and prev_time > 0:
            return output, step
    raise AssertionError(f"no autoreset frame within {max_steps} steps")


def test_reset_frame_contract() -> None:
    def make_cartpole() -> gym.Env:
        env = gym.make("CartPole-v1", max_episode_steps=50)
        return gym.wrappers.TransformObservation(
            env, lambda o: np.zeros_like(o), env.observation_space
        )

    cfgs = [
        EnvConfig(
            name=f"CartPole-custom_{i}",
            reset_seed=i,
            episodes_per_task=5,
            env_fn=make_cartpole,
        )
        for i in range(2)
    ]
    env = make_group_env(cfgs)
    try:
        outputs = _rollout(env, steps=2)
        assert len(outputs) == 2
        assert env.names == ("CartPole-custom_0", "CartPole-custom_1")
        obs = outputs[0]["observation"]
        assert np.all(obs == 0.0)
    finally:
        env.close()


def test_box_observation_preserves_native_uint8_dtype() -> None:
    class Uint8ImageEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 255, shape=(2, 3), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.full((2, 3), 7, dtype=np.uint8), {}

        def step(self, action):
            return np.full((2, 3), 9, dtype=np.uint8), 1.0, False, False, {}

    cfg = EnvConfig(
        reset_seed=0,
        episodes_per_task=5,
        env_fn=Uint8ImageEnv,
    )
    env = make_env(cfg)
    try:
        output = _rollout_single(env, steps=2)
        assert "observation" in output
        assert output["observation"].dtype == np.uint8
        obs_spec = env.output_spec.observation
        assert isinstance(obs_spec, FieldSpec)
        assert obs_spec.dtype == np.dtype(np.uint8)
        assert obs_spec.shape == (2, 3)
    finally:
        env.close()


def test_box_action_preserves_native_float64_dtype() -> None:
    class Float64ActionEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)

        def __init__(self):
            self.last_action = None

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            self.last_action = action
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    cfg = EnvConfig(
        reset_seed=0,
        episodes_per_task=5,
        env_fn=Float64ActionEnv,
    )
    env = make_env(cfg)
    try:
        input = env.sample_random_input()
        assert input["action"].dtype == np.float64
        assert env.input_spec.action.dtype == np.dtype(np.float64)

        env.step(input)
        input = {"action": np.array([0.25, -0.25], dtype=np.float64)}
        env.step(input)
        inner_env = cast(Float64ActionEnv, env._env_instance._env.unwrapped)
        assert inner_env.last_action is not None
        assert inner_env.last_action.dtype == np.float64
    finally:
        env.close()


def test_task_done_codes_fire_at_task_boundary() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=2,
        kwargs={"max_episode_steps": 10},
    )
    env = make_env(cfg)
    try:
        episode_dones: list[int] = []
        task_dones: list[int] = []
        for _ in range(300):
            output = env.step(env.sample_random_input())
            done = int(output["done"].item())
            if done in (DONE_EPISODE_TERMINATED, DONE_EPISODE_TRUNCATED):
                episode_dones.append(done)
            elif done in (DONE_TASK_TERMINATED, DONE_TASK_TRUNCATED):
                task_dones.append(done)
        assert len(task_dones) > 0, "expected some task-done steps within 300 steps"
        assert len(episode_dones) > 0, "expected some episode-done steps within 300 steps"
        output = env.step(env.sample_random_input())
        assert output["task_index"] >= 0
    finally:
        env.close()


def test_env_config_rejects_id_and_env_fn() -> None:
    with pytest.raises(ValueError, match="not both"):
        EnvConfig(id="CartPole-v1", reset_seed=0, env_fn=lambda: gym.make("CartPole-v1"))


def test_env_config_requires_id_or_env_fn() -> None:
    with pytest.raises(ValueError, match="at least one"):
        EnvConfig(reset_seed=0)


def test_env_fn_default_name_from_callable() -> None:
    class NamedEnv(gym.Env):
        observation_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return 0, {}

        def step(self, action):
            return 0, 0.0, True, False, {}

    env = make_env(EnvConfig(reset_seed=0, env_fn=NamedEnv))
    try:
        assert env.name == "NamedEnv"
    finally:
        env.close()


def test_anonymous_env_fn_requires_name() -> None:
    with pytest.raises(ValueError, match="anonymous callable"):
        make_env(EnvConfig(reset_seed=0, env_fn=lambda: gym.make("CartPole-v1")))


def test_metrics_accumulates_and_clears() -> None:
    cfg = EnvConfig(
        id="CartPole-v1",
        reset_seed=0,
        episodes_per_task=5,
        kwargs={"max_episode_steps": 10},
    )
    env = make_env(cfg)
    try:
        assert isinstance(env.metrics, Metrics)
        assert env.metrics.episode_cum_rewards == []
        assert env.metrics.episode_lengths == []

        for _ in range(200):
            env.step(env.sample_random_input())
            if env.metrics.episode_cum_rewards:
                break
        else:
            raise AssertionError("no episode completed within 200 steps")

        rewards = env.metrics.episode_cum_rewards
        lengths = env.metrics.episode_lengths
        assert len(rewards) >= 1
        assert len(lengths) == len(rewards)
        assert all(isinstance(r, float) for r in rewards)
        assert all(isinstance(l, float) for l in lengths)

        env.metrics.clear()
        assert env.metrics.episode_cum_rewards == []
        assert env.metrics.episode_lengths == []
    finally:
        env.close()


def test_group_env_max_threads_default_is_main_thread() -> None:
    env = make_group_env(
        [EnvConfig(id="CartPole-v1", reset_seed=i, episodes_per_task=5) for i in range(3)]
    )
    try:
        assert env.max_threads == 0
        outputs = _rollout(env, steps=3)
        assert len(outputs) == 3
        assert all(int(o["time"]) >= 0 for o in outputs)
    finally:
        env.close()


def test_group_env_max_threads_distributes_steps() -> None:
    env = make_group_env(
        [EnvConfig(id="CartPole-v1", reset_seed=i, name=f"cp-{i}", episodes_per_task=5) for i in range(4)],
        max_threads=2,
    )
    try:
        assert env.max_threads == 2
        outputs = _rollout(env, steps=5)
        assert len(outputs) == 4
        assert env.names == ("cp-0", "cp-1", "cp-2", "cp-3")
        for o in outputs:
            assert "observation" in o
            assert "done" in o
    finally:
        env.close()


def test_group_env_max_threads_matches_sequential_contract() -> None:
    """Threaded and sequential groups with the same seeds produce matching trajectories."""
    cfgs = [
        EnvConfig(id="CartPole-v1", reset_seed=i, episodes_per_task=5, kwargs={"max_episode_steps": 8})
        for i in range(3)
    ]
    sequential = make_group_env(cfgs, max_threads=0)
    threaded = make_group_env(cfgs, max_threads=3)
    try:
        for _ in range(20):
            seq_in = sequential.sample_random_input()
            # Reuse the same actions so trajectories stay comparable.
            thr_in = [{"action": inp["action"].copy()} for inp in seq_in]
            seq_out = sequential.step(seq_in)
            thr_out = threaded.step(thr_in)
            assert len(seq_out) == len(thr_out)
            for s, t in zip(seq_out, thr_out):
                assert int(s["done"]) == int(t["done"])
                assert int(s["time"]) == int(t["time"])
                assert int(s["episode_index"]) == int(t["episode_index"])
                assert int(s["task_index"]) == int(t["task_index"])
                np.testing.assert_array_equal(s["observation"], t["observation"])
                np.testing.assert_array_equal(s["reward"], t["reward"])
    finally:
        sequential.close()
        threaded.close()


def test_group_env_rejects_negative_max_threads() -> None:
    with pytest.raises(ValueError, match="max_threads"):
        make_group_env(
            [EnvConfig(id="CartPole-v1", reset_seed=0)],
            max_threads=-1,
        )