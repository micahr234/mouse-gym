# Changelog

All notable changes to mouse-gym are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- `EnvConfig.reset_seed` renamed to `seed`, and the stream now advances once per task instead of once per episode: the drawn value is passed to the underlying `env.reset(seed=...)` at the task-start reset only, and episode resets within a task pass no seed (Gymnasium's seed-once-per-session convention, applied per task). A whole task is reproducible from its seed while episode-level randomness still varies; envs that regenerate their problem instance when `reset` receives an explicit seed present the same instance to every episode in the task. Previously a fresh seed was drawn and passed on every episode reset.
- Step output dict key order is now `task_index`, `episode_index`, `step_index`, `reward`, `task_done`, `episode_done`, `observation`, `info`.
- `EnvConfig` now raises on `kwargs` or `render` combined with `env_fn` (both apply to `id` configs only) and on negative `episodes_per_task`, instead of silently ignoring them.

### Fixed
- `sample_random_input()` now returns action arrays matching `input_spec.action.shape`; size-1 Box actions (e.g. Pendulum's `(1,)`) were previously collapsed to 0-d scalars, contradicting the spec.

## [1.0.0] - 2026-08-18

First stable release. The public step contract (`episode_done` / `task_done`, task grouping, reset-free `step()` stream) is the API this version commits to.

### Added
- `GroupEnv` / `make_group_env` accept `max_threads` to step constituent envs on a thread pool (`0` = calling thread, default).
- `Metrics.task_cum_rewards` / `Metrics.task_lengths` and matching `GroupMetrics` properties accumulate reward and length sums for each completed task (`task_done` 2).

### Changed
- Step output field `done` renamed to `episode_done` (`0`/`1`/`2` from Gymnasium only). New `task_done` field uses the same codes (`0`; reserved `1`; `2` when `episodes_per_task` is reached). The last episode of a task now emits both.
- Minimum Python version raised from 3.12 to 3.13.
- Step output field `time` renamed to `step_index` (0-based within the episode; resets on episode restart).
- `episode_index` now counts episodes within the current task and resets to `0` when a new task starts.
- Step output dict key order is now `task_index`, `episode_index`, `step_index`, `reward`, `episode_done`, `task_done`, `observation`, `info`.

## [0.1.0] - 2026-07-07

First beta release of mouse-gym as a standalone package, split out of the pre-split mouse repo (which ended at 0.5.0). Versioning restarts at 0.1.0. Changes below are relative to that pre-split repo.

### Added
- Info passthrough smoke test confirming the Gymnasium `info` dict surfaces under `info`.

### Changed
- README clarifies that mouse-gym has no public `reset()` (including at task boundaries); episode and task transitions appear as reset frames from `step()`.
- `EnvConfig` now requires exactly one of `id` or `env_fn` (mutually exclusive). Display `name` defaults to `id`, or to the factory callable's `__name__` for named functions and classes; anonymous `lambda` factories require an explicit `name`.
- Step I/O uses NumPy arrays instead of PyTorch tensors, matching Gymnasium's native types.
- Repository scope narrowed to reset-free rollout infrastructure (`mouse-gym` / `mouse_gym`).
- Step outputs now forward the Gymnasium `info` dict verbatim under `info` instead of flattening keys to `info_<key>`.
- Renamed `env.tracker` / `Tracker` / `GroupTracker` to `env.metrics` / `Metrics` / `GroupMetrics`.

### Removed
- `env.tracker`, `Tracker`, and `GroupTracker` (replaced by `env.metrics`, `Metrics`, and `GroupMetrics`).
- First-party environment implementations and the `worlds/` package.
- Expert Q* machinery: `EnvConfig.q_star_source`, `QStarWrapper`, and the entire `experts/` package.
- Dependencies `stable-baselines3`, `huggingface_hub`, `pillow`, and `multiprocess` from the core package.
- Example notebooks for expert Q*, synthetic env, and procedural FrozenLake.
- Built-in partial observability (`observation_indices`, `ObservationSliceWrapper`) and reward shaping (`reward_scale`, `reward_shift`).
- Example notebooks for non-stationary envs, Atari preprocessing, partial observability, and reward shaping; remaining examples renumbered 01–04.
- Optional extras `atari` and `non-stationary`, and their integration tests.
