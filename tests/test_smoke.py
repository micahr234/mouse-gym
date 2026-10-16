"""Smoke tests for mouse-gym — offline, no external downloads."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest

from mouse_gym import EnvConfig, GroupEnv, Metrics, make_env, make_group_env


def _rollout(env, steps: int = 5) -> list:
    """Roll out a GroupEnv for ``steps`` steps; returns the final list of tuples."""
    outputs = env.step(env.action_space.sample())
    for _ in range(steps - 1):
        outputs = env.step(env.action_space.sample())
    return outputs


def _rollout_single(env, steps: int = 5) -> tuple:
    """Roll out a SingleEnv for ``steps`` steps; returns the final tuple."""
    output = env.step(env.action_space.sample())
    for _ in range(steps - 1):
        output = env.step(env.action_space.sample())
    return output


def test_cartpole_step_contract() -> None:
    cfgs = [
        EnvConfig(id="CartPole-v1", name=f"train-cartpole_{i}", seed=i, seed_mode="constant")
        for i in range(3)
    ]
    env = make_group_env(cfgs)
    try:
        outputs = _rollout(env)
        assert len(outputs) == 3
        assert env.names[0] == "train-cartpole_0"
        assert env.names == ("train-cartpole_0", "train-cartpole_1", "train-cartpole_2")
        for observation, reward, terminated, truncated, info in outputs:
            assert observation.shape == (4,)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
        for per_env in env.metrics.episode_cum_rewards:
            assert all(isinstance(v, float) for v in per_env)
    finally:
        env.close()


def test_group_env_exposes_gym_tuple_spaces() -> None:
    env = make_group_env(
        [
            EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant"),
            EnvConfig(id="CartPole-v1", seed=1, seed_mode="constant"),
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


def test_pendulum_continuous_step_contract() -> None:
    env = make_group_env(
        [
            EnvConfig(id="Pendulum-v1", seed=0, seed_mode="constant"),
            EnvConfig(id="Pendulum-v1", seed=1, seed_mode="constant"),
        ]
    )
    try:
        assert env.action_space.spaces[0].shape == (1,)
        action = env.action_space.sample()[0]
        assert action.dtype == np.float32
        assert action.shape == (1,)

        outputs = _rollout(env)
        assert len(outputs) == 2
        for observation, reward, _terminated, _truncated, _info in outputs:
            assert observation.shape == (3,)
            assert isinstance(reward, float)
    finally:
        env.close()


def test_group_env_rejects_wrong_action_count() -> None:
    env = make_group_env(
        [
            EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant"),
            EnvConfig(id="CartPole-v1", seed=1, seed_mode="constant"),
        ]
    )
    try:
        env.step(env.action_space.sample())
        with pytest.raises(ValueError, match="list or tuple"):
            env.step(cast(Any, 0))
        with pytest.raises(ValueError, match="exactly 2"):
            env.step([0])
    finally:
        env.close()


def test_single_env_reset_is_not_implemented() -> None:
    env = make_env(EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant"))
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

    env = make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=DictObsEnv))
    try:
        observation, _reward, _terminated, _truncated, _info = _rollout_single(env, steps=2)
        assert isinstance(observation, dict)
        assert "episode_done" not in observation
        assert observation["pos"].dtype == np.float32
        assert observation["tile"].dtype == np.int32
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

    env = make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=InfoEmittingEnv))
    try:
        _obs, _reward, _terminated, _truncated, reset_info = env.step(env.action_space.sample())
        _obs, _reward, _terminated, _truncated, step_info = env.step(env.action_space.sample())
        assert reset_info["foo"] == 1
        assert step_info["foo"] == 2
        assert np.asarray(reset_info["q_star"]).tolist() == [1.0, 0.0]
        assert np.asarray(step_info["q_star"]).tolist() == [0.0, 1.0]
    finally:
        env.close()


def test_same_seed_repeats_first_observation() -> None:
    def _first_obs(*, seed: int) -> np.ndarray:
        env = make_env(EnvConfig(id="CartPole-v1", seed=seed, seed_mode="constant"))
        try:
            observation, *_rest = env.step(env.action_space.sample())
            return observation
        finally:
            env.close()

    assert np.array_equal(_first_obs(seed=3), _first_obs(seed=3))
    assert not np.array_equal(_first_obs(seed=3), _first_obs(seed=30))


class _ProceduralEnv(gym.Env):
    """Records every ``reset`` seed and redraws its map from that seed."""

    observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self.received_seeds: list[int | None] = []
        self.received_options: list[dict | None] = []
        self.map: np.ndarray | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.received_seeds.append(seed)
        self.received_options.append(options)
        if self.map is None or seed is not None:
            self.map = self.np_random.integers(0, 10, size=4)
        assert self.map is not None
        start = np.array([self.np_random.random()], dtype=np.float32)
        return start, {"map": self.map.copy()}

    def step(self, action):
        return np.zeros(1, np.float32), 0.0, True, False, {}


def _reset_frames(env, *, steps: int) -> list[tuple]:
    frames = []
    for _ in range(steps):
        output = env.step(env.action_space.sample())
        if env.step_index == 0:
            frames.append(output)
    return frames


def test_seed_is_forwarded_on_every_reset() -> None:
    env = make_env(
        EnvConfig(
            env_fn=_ProceduralEnv,
            seed=7,
            seed_mode="constant",
            reset_options={"mode": "keep"},
        )
    )
    try:
        frames = _reset_frames(env, steps=8)
        inner = cast(_ProceduralEnv, env._env_instance._env.unwrapped)
        assert len(inner.received_seeds) >= 3
        assert inner.received_seeds == [7] * len(inner.received_seeds)
        assert all(options == {"mode": "keep"} for options in inner.received_options)

        maps = [info["map"] for _obs, _reward, _terminated, _truncated, info in frames]
        for later in maps[1:]:
            np.testing.assert_array_equal(later, maps[0])
        starts = [float(observation[0]) for observation, *_rest in frames]
        assert len(set(starts)) == 1
    finally:
        env.close()


def _received_seeds(env) -> list[int | None]:
    frames = _reset_frames(env, steps=8)
    assert len(frames) >= 3
    inner = cast(_ProceduralEnv, env._env_instance._env.unwrapped)
    return list(inner.received_seeds)


def test_reset_options_nested_mutation_stays_on_that_reset() -> None:
    class MutatingOptionsEnv(gym.Env):
        observation_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(2)

        def __init__(self) -> None:
            self.seen: list[tuple[bool, list[str]]] = []

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            assert isinstance(options, dict)
            self.seen.append((options["nested"]["flag"], list(options["tags"])))
            options["nested"]["flag"] = True
            options["tags"].append("mutated")
            return 0, {}

        def step(self, action):
            return 0, 0.0, True, False, {}

    options = {"nested": {"flag": False}, "tags": ["a"]}
    env = make_env(EnvConfig(env_fn=MutatingOptionsEnv, name="opts", reset_options=options))
    try:
        for _ in range(5):
            env.step(0)
        options["nested"]["flag"] = True
        options["tags"].append("caller")
        env.step(0)
        env.step(0)
        inner = cast(MutatingOptionsEnv, env._env_instance._env.unwrapped)
        assert inner.seen == [(False, ["a"])] * 4
        assert options == {"nested": {"flag": True}, "tags": ["a", "caller"]}
    finally:
        env.close()


def test_stream_draws_reset_seeds_from_the_initializer() -> None:
    def _seeds(seed: int) -> list[int | None]:
        env = make_env(EnvConfig(env_fn=_ProceduralEnv, seed=seed, seed_mode="stream"))
        try:
            return _received_seeds(env)
        finally:
            env.close()

    same_a = _seeds(7)
    same_b = _seeds(7)
    other = _seeds(8)
    assert same_a == same_b
    assert same_a != other
    assert len(set(same_a)) == len(same_a)
    assert all(isinstance(seed, int) and seed >= 0 for seed in same_a)


def test_default_seed_mode_passes_none() -> None:
    assert EnvConfig(id="CartPole-v1").seed_mode == "none"
    env = make_env(EnvConfig(env_fn=_ProceduralEnv))
    try:
        seeds = _received_seeds(env)
        assert seeds == [None] * len(seeds)
    finally:
        env.close()


def test_seed_mode_rejects_mismatched_seed() -> None:
    with pytest.raises(ValueError, match="must be an int"):
        EnvConfig(id="CartPole-v1", seed_mode="constant")
    with pytest.raises(ValueError, match="must be an int"):
        EnvConfig(id="CartPole-v1", seed_mode="stream")
    with pytest.raises(ValueError, match="must be None"):
        EnvConfig(id="CartPole-v1", seed=1)
    with pytest.raises(ValueError, match="must be None"):
        EnvConfig(id="CartPole-v1", seed=1, seed_mode="none")
    with pytest.raises(ValueError, match="seed_mode"):
        EnvConfig(id="CartPole-v1", seed=1, seed_mode=cast(Any, "fixed"))


def test_first_reset_map_is_deterministic() -> None:
    def _first_map(*, seed: int) -> np.ndarray:
        env = make_env(EnvConfig(env_fn=_ProceduralEnv, seed=seed, seed_mode="constant"))
        try:
            _obs, _reward, _terminated, _truncated, info = env.step(env.action_space.sample())
            return info["map"]
        finally:
            env.close()

    np.testing.assert_array_equal(_first_map(seed=1), _first_map(seed=1))
    assert not np.array_equal(_first_map(seed=1), _first_map(seed=2))


def test_autoreset_frame_reward_defaults_to_zero() -> None:
    env = make_env(
        EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant", kwargs={"max_episode_steps": 8})
    )
    try:
        (_obs, reward, terminated, truncated, _info), _step = _roll_until_autoreset(env)
        assert env.step_index == 0
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert len(env.metrics.episode_cum_rewards) >= 1
    finally:
        env.close()


def test_initial_reset_frame_reward_from_transform() -> None:
    def transform(*, step_index: int, reward: float, **_kwargs: object) -> float:
        return -1.0 if step_index == 0 else reward

    env = make_env(EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant", reward_transform=transform))
    try:
        _obs, reward, terminated, truncated, _info = env.step(env.action_space.sample())
        assert env.step_index == 0
        assert reward == -1.0
        assert terminated is False
        assert truncated is False
        assert env.metrics.episode_cum_rewards == ()
    finally:
        env.close()


def _roll_until_autoreset(env, *, max_steps: int = 500) -> tuple[tuple, int]:
    env.step(env.action_space.sample())
    for step in range(1, max_steps):
        prev_step_index = env.step_index
        output = env.step(env.action_space.sample())
        if env.step_index == 0 and prev_step_index > 0:
            return output, step
    raise AssertionError(f"no autoreset frame within {max_steps} steps")


def test_reset_frame_contract() -> None:
    def make_cartpole() -> gym.Env:
        env = gym.make("CartPole-v1", max_episode_steps=50)
        return gym.wrappers.TransformObservation(
            env, lambda o: np.zeros_like(o), env.observation_space
        )

    cfgs = [
        EnvConfig(name=f"CartPole-custom_{i}", seed=i, seed_mode="constant", env_fn=make_cartpole)
        for i in range(2)
    ]
    env = make_group_env(cfgs)
    try:
        outputs = _rollout(env, steps=2)
        assert len(outputs) == 2
        assert env.names == ("CartPole-custom_0", "CartPole-custom_1")
        observation = outputs[0][0]
        assert np.all(observation == 0.0)
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

    env = make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=Uint8ImageEnv))
    try:
        observation, _reward, _terminated, _truncated, _info = _rollout_single(env, steps=2)
        assert observation.dtype == np.uint8
        assert observation.shape == (2, 3)
    finally:
        env.close()


def test_box_action_is_forwarded_unchanged() -> None:
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

    env = make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=Float64ActionEnv))
    try:
        sampled = env.action_space.sample()
        assert sampled.dtype == np.float64

        env.step(sampled)
        action = np.array([0.25, -0.25], dtype=np.float64)
        env.step(action)
        inner_env = cast(Float64ActionEnv, env._env_instance._env.unwrapped)
        assert inner_env.last_action is action
    finally:
        env.close()


class ImmediateTerminateEnv(gym.Env):
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 0, 1.0, True, False, {}


class BothFlagsEnv(gym.Env):
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 0, 1.0, True, True, {}


def test_gym_flags_pass_through_together() -> None:
    env = make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=BothFlagsEnv))
    try:
        env.step(env.action_space.sample())
        _obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        assert terminated is True
        assert truncated is True
        assert env.metrics.episode_cum_rewards == (1.0,)
        assert env.metrics.episode_lengths == (1.0,)
    finally:
        env.close()


def test_reward_transform_applies_before_metrics() -> None:
    env = make_env(
        EnvConfig(
            seed=0,
            seed_mode="constant",
            env_fn=ImmediateTerminateEnv,
            reward_transform=lambda reward, **_kwargs: reward * 10.0,
        )
    )
    try:
        _obs, reward, _terminated, _truncated, _info = env.step(env.action_space.sample())
        assert reward == 0.0
        _obs, reward, _terminated, _truncated, _info = env.step(env.action_space.sample())
        assert reward == 10.0
        assert env.metrics.episode_cum_rewards == (10.0,)
        assert not hasattr(env.metrics, "task_cum_rewards")
    finally:
        env.close()


def test_reward_transform_receives_transition_kwargs() -> None:
    class TwoStepEnv(gym.Env):
        observation_space = gym.spaces.Discrete(3)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return 0, {}

        def step(self, action):
            self.t += 1
            return self.t, float(self.t), self.t >= 2, False, {}

    seen: list[dict[str, Any]] = []

    def transform(**kwargs: Any) -> float:
        seen.append(dict(kwargs))
        return float(kwargs["reward"])

    env = make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=TwoStepEnv, reward_transform=transform))
    try:
        env.step(0)
        env.step(1)
        env.step(0)
        env.step(0)
    finally:
        env.close()

    assert len(seen) == 4
    assert set(seen[0]) == {
        "step_index",
        "state",
        "action",
        "reward",
        "terminated",
        "truncated",
        "next_state",
    }
    assert seen[0]["step_index"] == 0
    assert seen[0]["state"] == 0
    assert seen[0]["action"] is None
    assert seen[0]["reward"] == 0.0
    assert seen[0]["terminated"] is False
    assert seen[0]["truncated"] is False
    assert seen[0]["next_state"] == 0
    assert seen[1]["step_index"] == 1
    assert seen[1]["state"] == 0
    assert seen[1]["action"] == 1
    assert seen[1]["next_state"] == 1
    assert seen[1]["reward"] == 1.0
    assert seen[1]["terminated"] is False
    assert seen[1]["truncated"] is False
    assert seen[2]["step_index"] == 2
    assert seen[2]["state"] == 1
    assert seen[2]["action"] == 0
    assert seen[2]["next_state"] == 2
    assert seen[2]["reward"] == 2.0
    assert seen[2]["terminated"] is True
    assert seen[2]["truncated"] is False
    assert seen[3]["step_index"] == 0
    assert seen[3]["state"] == 2
    assert seen[3]["next_state"] == 0
    assert seen[3]["reward"] == 0.0
    assert seen[3]["terminated"] is False
    assert seen[3]["truncated"] is False
    assert seen[3]["action"] is None


def test_reward_transform_accepts_kwargs_function() -> None:
    def scale(*, reward: float, **_kwargs: object) -> float:
        return reward * 10.0

    env = make_env(
        EnvConfig(seed=0, seed_mode="constant", env_fn=ImmediateTerminateEnv, reward_transform=scale)
    )
    try:
        env.step(env.action_space.sample())
        _obs, reward, _terminated, _truncated, _info = env.step(env.action_space.sample())
        assert reward == 10.0
        assert env.metrics.episode_cum_rewards == (10.0,)
    finally:
        env.close()


def test_reset_frame_reward_from_transform_is_in_metrics() -> None:
    def transform(*, step_index: int, reward: float, **_kwargs: object) -> float:
        return -1.0 if step_index == 0 else reward * 10.0

    env = make_env(
        EnvConfig(seed=0, seed_mode="constant", env_fn=ImmediateTerminateEnv, reward_transform=transform)
    )
    try:
        _obs, reward, _terminated, _truncated, _info = env.step(env.action_space.sample())
        assert reward == -1.0
        assert env.metrics.episode_cum_rewards == ()
        _obs, reward, _terminated, _truncated, _info = env.step(env.action_space.sample())
        assert reward == 10.0
        assert env.metrics.episode_cum_rewards == (9.0,)
        assert env.metrics.episode_lengths == (1.0,)
    finally:
        env.close()


def test_later_reset_frames_use_the_same_transform() -> None:
    def transform(*, step_index: int, reward: float, **_kwargs: object) -> float:
        return -1.0 if step_index == 0 else reward

    env = make_env(
        EnvConfig(seed=0, seed_mode="constant", env_fn=ImmediateTerminateEnv, reward_transform=transform)
    )
    try:
        _obs, reward, _terminated, _truncated, _info = env.step(env.action_space.sample())
        assert reward == -1.0
        env.step(env.action_space.sample())
        _obs, reward, terminated, _truncated, _info = env.step(env.action_space.sample())
        assert env.step_index == 0
        assert reward == -1.0
        assert terminated is False
    finally:
        env.close()


def test_env_config_rejects_non_callable_reward_transform() -> None:
    with pytest.raises(ValueError, match="reward_transform"):
        EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant", reward_transform=cast(Any, 2.0))


def test_step_index_resets_after_gym_done() -> None:
    env = make_env(
        EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant", kwargs={"max_episode_steps": 8})
    )
    try:
        _obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        assert env.step_index == 0
        saw_nonzero = False
        for _ in range(100):
            prev = env.step_index
            _obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
            cur = env.step_index
            if cur == 0 and prev > 0:
                assert terminated is False
                assert truncated is False
                return
            if cur > 0:
                saw_nonzero = True
                assert cur == prev + 1
        raise AssertionError(
            f"no episode restart within 100 steps (saw_nonzero={saw_nonzero})"
        )
    finally:
        env.close()


def test_observation_episode_flags_do_not_end_the_segment() -> None:
    """Flags inside the observation are data. Only Gymnasium flags end a segment."""

    class FlagInObsEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                    "episode_terminated": gym.spaces.Discrete(2),
                    "episode_truncated": gym.spaces.Discrete(2),
                }
            )
            self.action_space = gym.spaces.Discrete(2)
            self.resets = 0
            self.n = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.resets += 1
            self.n = 0
            return self._obs(False), {}

        def step(self, action):
            self.n += 1
            episode_flag = self.n == 1
            task_end = self.n == 3
            return self._obs(episode_flag), 1.0, task_end, False, {}

        def _obs(self, episode_terminated: bool) -> dict[str, Any]:
            return {
                "observation": np.zeros(1, dtype=np.float32),
                "episode_terminated": episode_terminated,
                "episode_truncated": False,
            }

    env = make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=FlagInObsEnv))
    try:
        observation, _reward, terminated, _truncated, _info = env.step(0)
        assert env.step_index == 0
        assert env.metrics.episode_cum_rewards == ()

        observation, _reward, terminated, _truncated, _info = env.step(0)
        assert bool(observation["episode_terminated"]) is True
        assert terminated is False
        assert env.step_index == 1
        assert env.metrics.episode_cum_rewards == ()

        observation, _reward, _terminated, _truncated, _info = env.step(0)
        assert env.step_index == 2
        assert bool(observation["episode_terminated"]) is False

        _observation, _reward, terminated, _truncated, _info = env.step(0)
        assert terminated is True
        assert env.step_index == 3
        assert env.metrics.episode_cum_rewards == (3.0,)
        assert env.metrics.episode_lengths == (3.0,)

        _observation, _reward, terminated, _truncated, _info = env.step(0)
        assert env.step_index == 0
        assert terminated is False
        inner = cast(FlagInObsEnv, env._env_instance._env.unwrapped)
        assert inner.resets == 2
    finally:
        env.close()


def test_env_config_rejects_id_and_env_fn() -> None:
    with pytest.raises(ValueError, match="not both"):
        EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant", env_fn=lambda: gym.make("CartPole-v1"))


def test_env_config_requires_id_or_env_fn() -> None:
    with pytest.raises(ValueError, match="at least one"):
        EnvConfig(seed=0, seed_mode="constant")


def test_env_config_rejects_kwargs_with_env_fn() -> None:
    def make_cartpole() -> gym.Env:
        return gym.make("CartPole-v1")

    with pytest.raises(ValueError, match="kwargs only apply to id configs"):
        EnvConfig(seed=0, seed_mode="constant", env_fn=make_cartpole, kwargs={"max_episode_steps": 5})


def test_env_fn_default_name_from_callable() -> None:
    class NamedEnv(gym.Env):
        observation_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return 0, {}

        def step(self, action):
            return 0, 0.0, True, False, {}

    env = make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=NamedEnv))
    try:
        assert env.name == "NamedEnv"
    finally:
        env.close()


def test_anonymous_env_fn_requires_name() -> None:
    with pytest.raises(ValueError, match="anonymous callable"):
        make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=lambda: gym.make("CartPole-v1")))


def test_metrics_accumulates_and_clears() -> None:
    env = make_env(
        EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant", kwargs={"max_episode_steps": 10})
    )
    try:
        assert isinstance(env.metrics, Metrics)
        assert env.metrics.episode_cum_rewards == ()
        assert env.metrics.episode_lengths == ()
        assert not hasattr(env.metrics, "task_cum_rewards")
        assert not hasattr(env.metrics, "task_lengths")

        for _ in range(200):
            env.step(env.action_space.sample())
            if env.metrics.episode_cum_rewards:
                break
        else:
            raise AssertionError("no episode completed within 200 steps")

        rewards = env.metrics.episode_cum_rewards
        lengths = env.metrics.episode_lengths
        assert len(rewards) >= 1
        assert len(lengths) == len(rewards)
        assert all(isinstance(r, float) for r in rewards)
        assert all(isinstance(length, float) for length in lengths)

        env.metrics.clear()
        assert env.metrics.episode_cum_rewards == ()
        assert env.metrics.episode_lengths == ()
    finally:
        env.close()


def test_group_env_max_threads_default_is_main_thread() -> None:
    env = make_group_env([EnvConfig(id="CartPole-v1", seed=i, seed_mode="constant") for i in range(3)])
    try:
        assert env.max_threads == 0
        outputs = _rollout(env, steps=3)
        assert len(outputs) == 3
        assert all(single.step_index >= 0 for single in env.envs)
    finally:
        env.close()


def test_group_env_max_threads_distributes_steps() -> None:
    env = make_group_env(
        [EnvConfig(id="CartPole-v1", seed=i, seed_mode="constant", name=f"cp-{i}") for i in range(4)],
        max_threads=2,
    )
    try:
        assert env.max_threads == 2
        outputs = _rollout(env, steps=5)
        assert len(outputs) == 4
        assert env.names == ("cp-0", "cp-1", "cp-2", "cp-3")
        for observation, _reward, terminated, truncated, info in outputs:
            assert observation.shape == (4,)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
    finally:
        env.close()


def test_group_env_max_threads_matches_sequential_contract() -> None:
    """Threaded and sequential groups with the same seeds produce matching trajectories."""
    cfgs = [
        EnvConfig(id="CartPole-v1", seed=i, seed_mode="constant", kwargs={"max_episode_steps": 8})
        for i in range(3)
    ]
    sequential = make_group_env(cfgs, max_threads=0)
    threaded = make_group_env(cfgs, max_threads=3)
    try:
        for _ in range(20):
            actions = sequential.action_space.sample()
            seq_out = sequential.step(actions)
            thr_out = threaded.step(actions)
            assert len(seq_out) == len(thr_out)
            for i, (seq, thr) in enumerate(zip(seq_out, thr_out)):
                assert seq[2] == thr[2]
                assert seq[3] == thr[3]
                assert sequential.envs[i].step_index == threaded.envs[i].step_index
                np.testing.assert_array_equal(seq[0], thr[0])
                assert seq[1] == thr[1]
    finally:
        sequential.close()
        threaded.close()


def test_group_env_rejects_negative_max_threads() -> None:
    with pytest.raises(ValueError, match="max_threads"):
        make_group_env([EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant")], max_threads=-1)


def test_group_env_rejects_shared_or_repeated_envs() -> None:
    env_a = make_env(EnvConfig(id="CartPole-v1", seed=0, seed_mode="constant", name="a"))
    env_b = make_env(EnvConfig(id="CartPole-v1", seed=1, seed_mode="constant", name="b"))
    fresh = make_env(EnvConfig(id="CartPole-v1", seed=2, seed_mode="constant", name="fresh"))
    group = GroupEnv([env_a, env_b])
    try:
        with pytest.raises(ValueError, match="already in a GroupEnv"):
            GroupEnv([env_a])
        with pytest.raises(ValueError, match="unique"):
            GroupEnv([fresh, fresh])
        assert group.envs == (env_a, env_b)
        assert group.action_space is group.action_space
    finally:
        group.close()
        fresh.close()


def test_reused_observation_and_info_are_copied() -> None:
    class ReuseEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def __init__(self) -> None:
            self.obs = np.zeros(1, np.float32)
            self.info = {"n": 0}

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.obs[:] = 0
            self.info["n"] = 0
            return self.obs, self.info

        def step(self, action):
            self.obs[:] = 1
            self.info["n"] += 1
            return self.obs, 0.0, False, False, self.info

    env = make_env(EnvConfig(env_fn=ReuseEnv, name="reuse"))
    try:
        obs0, _reward, _terminated, _truncated, info0 = env.step(0)
        obs1, _reward, _terminated, _truncated, info1 = env.step(0)
        assert obs0 is not obs1
        assert info0 is not info1
        assert float(obs0[0]) == 0.0
        assert float(obs1[0]) == 1.0
        assert info0["n"] == 0
        assert info1["n"] == 1
    finally:
        env.close()


def test_render_returns_the_env_value() -> None:
    class FrameEnv(gym.Env):
        observation_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return 0, {}

        def step(self, action):
            return 0, 0.0, False, False, {}

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    single = make_env(EnvConfig(env_fn=FrameEnv, name="frame"))
    group = make_group_env(
        [
            EnvConfig(env_fn=FrameEnv, name="frame-a"),
            EnvConfig(env_fn=FrameEnv, name="frame-b"),
        ]
    )
    try:
        frame = single.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (2, 2, 3)
        frames = group.render()
        assert isinstance(frames, tuple)
        assert len(frames) == 2
        assert frames[0].shape == (2, 2, 3)
    finally:
        single.close()
        group.close()


@pytest.mark.parametrize("max_threads", [0, 2])
def test_executed_action_metadata_covers_outer_and_inner_resets(max_threads):
    RepeatTaskEnv = pytest.importorskip("repeat_task_gym").RepeatTaskEnv
    calls = []
    def transform(**transition):
        calls.append(transition["action"])
        return transition["reward"]
    env = make_group_env([
        EnvConfig(seed=i, seed_mode="stream", name=str(i),
                  env_fn=lambda: RepeatTaskEnv(ImmediateTerminateEnv(), max_task_episodes=2),
                  reward_transform=transform)
        for i in range(2)
    ], max_threads=max_threads)
    try:
        for inputs, expected in [([99, None], [None, None]), ([0, 1], [0, 1]),
                                 ([99, None], [None, None]), ([1, 0], [1, 0]),
                                 ([None, 99], [None, None])]:
            outputs = env.step(inputs)
            assert [out[-1]["executed_action"] for out in outputs] == expected
        assert calls.count(None) == 6
        assert calls.count(0) == calls.count(1) == 2
    finally:
        env.close()


def test_plain_env_executed_action_records_reset_and_real_step():
    env = make_env(EnvConfig(seed=0, seed_mode="constant", env_fn=ImmediateTerminateEnv))
    try:
        assert env.step(99)[-1]["executed_action"] is None
        assert env.step(1)[-1]["executed_action"] == 1
        assert env.step(None)[-1]["executed_action"] is None
        assert env.step(0)[-1]["executed_action"] == 0
    finally:
        env.close()
