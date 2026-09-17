"""Benchmark ``mouse_gym`` env ``step`` on CartPole-v1.

Times ``sample_random_input`` + ``step`` for a ``SingleEnv`` and for
``GroupEnv`` widths in ``--num-envs``, each with every ``--threads`` cap
(``0`` = all envs on the calling thread). Reports median / min / max wall
time, first-call time, group-steps/second, and env-steps/second (group
rate × width).

    .venv/bin/python bench/bench_env.py
    .venv/bin/python bench/bench_env.py --num-envs 1 4 16 --threads 0 4
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

from mouse_gym import EnvConfig, GroupEnv, SingleEnv, make_env, make_group_env


def _config(i: int, *, max_task_episodes: int) -> EnvConfig:
    return EnvConfig(
        id="CartPole-v1",
        name=f"cartpole_{i}",
        seed=i,
        max_task_episodes=max_task_episodes,
    )


def _timed(fn: Callable[[], Any], iters: int) -> tuple[float, float, float, float]:
    """(first-call ms, median ms, min ms, max ms)."""
    t0 = time.perf_counter()
    fn()
    first = (time.perf_counter() - t0) * 1e3
    fn()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return first, statistics.median(times), min(times), max(times)


def _bench_env(env: SingleEnv | GroupEnv, iters: int, label: str, width: int) -> None:
    def tick() -> None:
        env.step(env.sample_random_input())

    first, med, lo, hi = _timed(tick, iters)
    env_steps_s = width / med * 1e3
    print(
        f"  {label:28s} | {med:7.2f} ms [{lo:.2f},{hi:.2f}] | "
        f"{1e3 / med:>8,.0f} group/s | {env_steps_s:>8,.0f} env-step/s | first {first:.0f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--threads", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-task-episodes", type=int, default=5)
    args = parser.parse_args()

    if any(n < 1 for n in args.num_envs):
        raise SystemExit("--num-envs values must be >= 1")
    if any(t < 0 for t in args.threads):
        raise SystemExit("--threads values must be >= 0")

    print(f"CartPole-v1 | max_task_episodes={args.max_task_episodes}")

    single = make_env(_config(0, max_task_episodes=args.max_task_episodes))
    print("\n### SingleEnv")
    _bench_env(single, args.iters, "make_env", 1)
    single.close()

    for n in args.num_envs:
        print(f"\n### GroupEnv n={n}")
        for threads in args.threads:
            configs = [_config(i, max_task_episodes=args.max_task_episodes) for i in range(n)]
            env = make_group_env(configs, max_threads=threads)
            _bench_env(env, args.iters, f"max_threads={threads}", n)
            env.close()


if __name__ == "__main__":
    main()
