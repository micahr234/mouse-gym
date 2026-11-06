"""Real spawned workers: isolation, buffer lifetimes, and failure cleanup."""
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os

import gymnasium as gym
import numpy as np
import pytest

from mouse_gym import EnvConfig, GroupEnv, make_env, make_group_env


class ArrayEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(-1, 1, shape=(128, 128), dtype=np.float64)

    def __init__(self):
        self.count = 0
        self.array = np.zeros((128, 128, 3), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        self.array.fill(0)
        return self.array, {"pid": os.getpid()}

    def step(self, action):
        self.count += 1
        self.array.fill(self.count)
        return self.array, float(np.sum(action)), False, False, {
            "pid": os.getpid(), "nested": {"array": self.array},
            "variable": np.arange(self.count * 100000, dtype=np.int32),
        }

    def render(self):
        return self.array


def assert_released(names):
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name, track=False)


def test_shared_memory_resize_preserves_arrays_and_actions_after_close():
    group = make_group_env([EnvConfig(env_fn=ArrayEnv) for _ in range(3)], num_workers=2)
    try:
        assert group._workers is not None
        initial = group.step([None] * 3)
        pids = {out[-1]["pid"] for out in initial}
        assert len(pids) == 2 and os.getpid() not in pids
        original_names = [slot.name for slot in group._workers._slots]
        actions = [np.full((128, 128), i / 4, dtype=np.float64) for i in range(3)]
        first = group.step(actions)
        actions[1][:] = -1
        second = group.step(actions)
        assert [out[1] for out in first] == [0, 4096, 8192]
        assert second[1][1] == -16384
        assert first[1][-1]["executed_action"].dtype == np.float64
        assert first[1][-1]["executed_action"][0, 0] == .25
        for out in first:
            np.testing.assert_array_equal(out[0], 1)
            np.testing.assert_array_equal(out[-1]["nested"]["array"], 1)
            assert len(out[-1]["variable"]) == 100000
        for frame in group.render():
            np.testing.assert_array_equal(frame, 2)
        final_names = [slot.name for slot in group._workers._slots]
        assert set(original_names).isdisjoint(final_names)
    finally:
        group.close()
    group.close()
    assert_released(original_names + final_names)
    np.testing.assert_array_equal(initial[0][0], 0)
    np.testing.assert_array_equal(first[0][0], 1)
    with pytest.raises(RuntimeError, match="closed"):
        group.step([None] * 3)


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_local_factories_transforms_spaces_and_individual_handles(workers):
    scale = 3
    configs = [EnvConfig(name="cart", env_fn=lambda: gym.make("CartPole-v1"),
                         seed=1, seed_mode="stream",
                         reward_transform=lambda **kw: kw["reward"] * scale),
               EnvConfig(id="MountainCar-v0", seed=2, seed_mode="constant")]
    inline = make_group_env(configs)
    process = make_group_env(configs, num_workers=workers)
    try:
        assert process.names == inline.names
        assert [space.shape for space in process.observation_space.spaces] == [(4,), (2,)]
        for _ in range(12):
            for a, b in zip(inline.step([1, 2]), process.step([1, 2])):
                np.testing.assert_array_equal(a[0], b[0])
                assert a[1:] == b[1:]
        for index, action in enumerate([0, 1]):
            a = inline.envs[index].step(action)
            b = process.envs[index].step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:] == b[1:]
        process.envs[0].close()
        process.envs[0].close()
    finally:
        inline.close()
        process.close()


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_worker_count(value):
    with pytest.raises(ValueError, match="num_workers"):
        make_group_env([EnvConfig(id="CartPole-v1")], num_workers=value)


def test_processes_require_factories_instead_of_live_simulator_instances():
    single = make_env(EnvConfig(id="CartPole-v1"))
    try:
        with pytest.raises(ValueError, match="EnvConfig"):
            GroupEnv([single], num_workers=1)
        assert single._group is None
        single.step(0)
    finally:
        single.close()


def test_worker_exception_closes_all_workers_and_shared_memory(tmp_path):
    marker = tmp_path / "closed"
    class BrokenEnv(ArrayEnv):
        def step(self, action):
            raise ValueError("deliberate step failure")
        def close(self):
            marker.write_text("closed")
    group = make_group_env([EnvConfig(env_fn=BrokenEnv)], num_workers=1)
    group.step([None])
    assert group._workers is not None
    names = [slot.name for slot in group._workers._slots]
    pids = {process.pid for process in group._workers._processes}
    with pytest.raises(RuntimeError, match="deliberate step failure"):
        group.step([0])
    assert marker.read_text() == "closed"
    assert not pids.intersection(process.pid for process in mp.active_children())
    assert_released(names)
    group.close()


def test_worker_exit_is_reported_without_hanging():
    class ExitEnv(ArrayEnv):
        def step(self, action):
            os._exit(7)
    group = make_group_env([EnvConfig(env_fn=ExitEnv)], num_workers=1)
    try:
        group.step([None])
        assert group._workers is not None
        names = [slot.name for slot in group._workers._slots]
        with pytest.raises(RuntimeError, match="disconnected|exited"):
            group.step([0])
        assert_released(names)
    finally:
        group.close()


def test_constructor_failure_closes_previously_built_environment(tmp_path):
    marker = tmp_path / "closed"
    class ClosingEnv(ArrayEnv):
        def close(self):
            marker.write_text("closed")
    def broken_factory():
        raise ValueError("deliberate constructor failure")
    before = {process.pid for process in mp.active_children()}
    with pytest.raises(RuntimeError, match="deliberate constructor failure"):
        make_group_env([EnvConfig(env_fn=ClosingEnv), EnvConfig(env_fn=broken_factory)], num_workers=1)
    assert marker.read_text() == "closed"
    assert {process.pid for process in mp.active_children()} == before
