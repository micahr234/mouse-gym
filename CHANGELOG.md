# Changelog

All notable changes to mouse-gym are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Info passthrough smoke test confirming the Gymnasium `info` dict surfaces under `info`.

### Changed
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

## [0.5.0] - 2026-06-25

### Added
- `EnvConfig.reset_seed` now controls the package's internal Gymnasium reset seeding.
- `GroupEnv` now subclasses `gymnasium.Env` and exposes dynamic tuple `action_space` and `observation_space` attributes for underlying env instances.
- Added an RNG seeding control notebook showing how to reproduce or vary reset behavior independently.
- `EnvConfig.episode_reset_options` forwards options to every internal `env.reset(options=...)`; `EnvConfig.task_reset_options` overlays options only when a reset starts a new task.
- `episodes_per_task` now defaults to `0` (unlimited) — the task boundary (done codes 3/4) never fires automatically. Passing any positive integer restores the previous fixed-count behaviour.
- `MetricsTracker` class attached to env wrappers as `env.tracker`; accumulates per-env `episode_cum_rewards` and `episode_lengths` automatically on every `step()` call and can be cleared with `env.tracker.clear()`.
- All Gymnasium `info` dict keys are now forwarded verbatim as `info_<key>` in every step output. For example, `info["map"]` appears as `outputs[i]["info_map"]` and `info["ns_params"]` as `outputs[i]["info_ns_params"]`. No env-specific filtering is applied.

### Changed
- Repeated env instances are now created by passing an explicit `list[EnvConfig]` to `make_env`; each `EnvConfig` builds exactly one env instance.
- Per-instance action-space access now uses the standard Gymnasium tuple space API (`env.action_space.spaces[i]`) instead of the removed `action_spaces` helper.
- Renamed flattened-env terminology to "env instance" / "env index"; the flat `outputs[i]`, `inputs[i]`, `env.names`, `env.input_specs`, and `env.output_specs` API shape is unchanged.
- `OutputSpec` no longer has dedicated fields for individual info keys; info keys are dynamic and discovered from step outputs.
- `step()` now returns `list[dict]` (outputs only) instead of `tuple[list[dict], list[dict]]` (outputs, metrics). Episode statistics are no longer returned inline; read them from `env.tracker` instead.

### Fixed
- All example notebooks updated to use `episodes_per_task` (required field) instead of the removed `max_episode_steps` on `EnvConfig`. Episode time limits moved to `kwargs` where needed.
- Notebook 06 removed stale `reward_episodic` output field references.

### Removed
- `EnvConfig.num_envs`; use one `EnvConfig` per env instance so per-env seeds are explicit.
- Per-env `action_spaces` helper; use `env.action_space.spaces[i]` or drill down into the underlying Gymnasium env instance instead.
- `RolloutMetrics` TypedDict removed from the public API (no longer exported).

## [0.4.1] - 2026-06-24
