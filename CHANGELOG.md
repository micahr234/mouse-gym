# Changelog

All notable changes to mouse-gym are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
