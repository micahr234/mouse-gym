"""Compare inline and process stepping with fixed actions and output digests.

Install the examples extra for FrozenLake. Rates count all environment frames,
including reset frames. Action generation, trackers, and hashing are outside the
step timer; startup is reported separately. This does not measure a full data
collector or transformer inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import gymnasium as gym
import numpy as np
from mouse_gym import EnvConfig, RepeatTaskTracker, make_group_env


def configuration(index, workload):
    if workload == "cartpole":
        return EnvConfig(id="CartPole-v1", seed=index, seed_mode="stream")

    def lake():
        import procedural_frozenlake  # registers the environment in each worker
        from repeat_task_gym import RepeatTaskEnv

        return RepeatTaskEnv(
            gym.make("Procedural-FrozenLake-v1", width=8, height=8,
                     max_episode_steps=24, q_star_horizon=24, map_seed=index,
                     slippery_success_rate=1., permute_obs=False, permute_actions=False,
                     emit_q_star=True, emit_map=True, q_star_gamma=.99),
            max_task_episodes=20, episode_seed_mode="per_task",
            episode_reset_options={"regenerate_map": False},
        )

    return EnvConfig(name=f"lake{index}", env_fn=lake, seed=index,
                     seed_mode="stream", reset_options={"regenerate_map": True})


def canonical(value):
    """Compare values rather than pickle's object identity/aliasing metadata."""
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    return value


def benchmark(*, workload, num_workers, width, steps):
    started = time.perf_counter()
    configs = [configuration(i, workload) for i in range(width)]
    group = make_group_env(configs, num_workers=num_workers)
    startup = time.perf_counter() - started
    rng = np.random.default_rng(1)
    trackers = [RepeatTaskTracker() for _ in configs]
    digest = hashlib.sha256()
    times = []
    warmup = []
    try:
        for step in range(steps + 20):
            actions = [int(x) for x in rng.integers(4 if workload == "frozenlake" else 2, size=width)]
            if workload == "frozenlake":
                actions = [action if tracker.action_requested else None
                           for action, tracker in zip(actions, trackers)]
            started = time.perf_counter()
            outputs = group.step(actions)
            duration = time.perf_counter() - started
            (times if step >= 20 else warmup).append(duration)
            if workload == "frozenlake":
                for tracker, output in zip(trackers, outputs):
                    tracker.update(output=output)
            digest.update(json.dumps(canonical(outputs), sort_keys=True, separators=(",", ":")).encode())
    finally:
        group.close()
    import mouse_gym.format
    source = Path(mouse_gym.format.__file__)
    return {
        "workload": workload, "num_workers": num_workers, "width": width, "steps": steps,
        "python": sys.version, "gil_enabled": sys._is_gil_enabled(),
        "source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "startup_seconds": startup, "warmup_seconds": sum(warmup),
        "step_seconds": sum(times), "env_steps_per_second": width * steps / sum(times),
        "median_step_ms": statistics.median(times) * 1000, "output_sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["cartpole", "frozenlake"], default="frozenlake")
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.envs, args.steps, args.repeats) < 1 or any(n < 0 for n in args.workers):
        parser.error("envs, steps, repeats must be positive; workers must be nonnegative")
    results = []
    for repeat in range(args.repeats):
        for workers in args.workers:
            row = benchmark(workload=args.workload, num_workers=workers, width=args.envs, steps=args.steps)
            results.append({"repeat": repeat, **row})
            print(json.dumps(results[-1]), flush=True)
    if len({row["output_sha256"] for row in results}) != 1:
        raise RuntimeError("Worker counts produced different trajectories")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
