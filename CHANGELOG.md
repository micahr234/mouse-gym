# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The `examples` extra installs notebook tools (`jupyter`, `ipykernel`,
  `matplotlib`), Gymnasium's `classic-control` extra (`pygame-ce`, imported
  as `pygame`) so CartPole can render, `repeat-task-gym` from
  `https://github.com/micahr234/repeat-task-gym`, and `procedural-frozenlake`
  from `https://github.com/micahr234/procedural-frozenlake`. Notebook
  [04 — Repeat a task](examples/04_repeat_task_env.ipynb) wraps
  `Procedural-FrozenLake-v1` in a `RepeatTaskEnv` and passes that env to
  `make_env` through `env_fn`.

### Changed
- `step()` takes a Gymnasium action and returns
  `(observation, reward, terminated, truncated, info)`. `terminated` and
  `truncated` are the wrapped env's flags as Python bools. The observation
  and `info` dict are deep-copied. `env.step_index` is `0` on a
  reset frame. Random actions come from `action_space.sample()`.
- `EnvConfig.seed_mode` chooses the reset seed and defaults to `none`,
  which passes `seed=None`. `constant` passes `seed` on every reset.
  `stream` draws each reset seed from an RNG initialized with `seed`.
- `EnvConfig.reset_options` is forwarded to every reset mouse-gym issues.
  Each reset gets a deep copy of a snapshot taken when the env is built.
- `reward_transform` is called with `step_index`, `state`, `action`,
  `reward`, `terminated`, `truncated`, and `next_state`. On a reset frame,
  `state` is the reset observation the first time and the previous
  observation after that. `reward` is `0.0`. `terminated` and `truncated`
  are false. `action` is `None` because that call does not apply an action.
- `Metrics` records `episode_cum_rewards` and `episode_lengths` when the
  wrapped env reports `terminated` or `truncated`. Those flags mean whatever
  the wrapped env uses them for. Both properties return tuples.
- `GroupEnv.step` returns a tuple of per-env results. `GroupEnv.envs`
  returns a tuple. `action_space` and `observation_space` are built once
  when the group is created. Each `SingleEnv` belongs to at most one
  `GroupEnv`.
- `render()` returns the wrapped env's own value. `GroupEnv.render()`
  returns one result per env. Set `render_mode` through `kwargs` or inside
  `env_fn`.

### Removed
- `EnvConfig.render`.
- Sharing one `SingleEnv` across more than one `GroupEnv`.
- `scripts/worker.sh`. The Cursor My Machines worker script lives only in
  `mouse-experiment` (`scripts/worker.sh` there still registers sibling
  `mouse-core` / `mouse-experiment` / `mouse-gym` checkouts).
- First-reset-only seeding. Example notebook
  `examples/03_rng_seeding_control.ipynb` is gone; metrics is
  [03 — Metrics](examples/03_metrics.ipynb) and the repeat-task notebook is
  [04 — Repeat a task](examples/04_repeat_task_env.ipynb).
- Dict step I/O: `sample_random_input`, `input_spec`, `output_spec`,
  `input_specs`, `output_specs`, `FieldSpec`, `InputSpec`, `OutputSpec`,
  and `StepOutput`.
- Task grouping: `max_task_episodes`, `terminate_task`, `task_done`,
  `task_index`, `episode_index`, `episode_done`, `episode_reset_options`,
  `task_reset_options`, the per-task seed stream, and
  `Metrics.task_cum_rewards` / `Metrics.task_lengths` (including the
  matching `GroupMetrics` properties).

## [1.1.0] - 2026-09-17

Reward and task boundaries are now callables: shape each step with `reward_transform`, end a task with `terminate_task` (`task_done=1`), or time out after `max_task_episodes` (`task_done=2`).

### Added
- ``bench/bench_env.py``: ``SingleEnv`` / ``GroupEnv`` step rate on
  CartPole-v1, including ``max_threads``.
- `EnvConfig.terminate_task`: optional callable invoked on episode-end
  steps with the same transition kwargs as `reward_transform`. A truthy
  return emits `task_done=1`. `max_task_episodes` remains the task-length
  timeout (`task_done=2`). If both would fire, termination wins.
- `EnvConfig.reward_transform`: optional callable applied to each
  Gymnasium step reward and each reset-frame reward before the step
  output and before episode and task metrics accumulate. Invoked with
  `step_index`, `episode_index`, `state`, `action`, `reward`, `done`
  (`episode_done` 0/1/2), and `next_state`. Reset frames pass
  `step_index=0`, `reward=0.0`, and `action=None`; a task-start reset
  is also `episode_index=0`.
- Example notebook [05 — Reward transform and task termination](examples/05_reward_and_task.ipynb)
  for `reward_transform`, `terminate_task`, and `max_task_episodes`.

### Changed
- Minimum Python version raised from 3.13 to 3.14. Development setup (`scripts/install.sh`), CI, publish workflow, and Pyright now target free-threaded Python 3.14t; Python 3.13 is no longer supported. NumPy floor raised to 2.5.3 for 3.14 wheels.
- `EnvConfig.reset_seed` renamed to `seed`, and the stream now advances once per task instead of once per episode: the drawn value is passed to the underlying `env.reset(seed=...)` at the task-start reset only, and episode resets within a task pass no seed (Gymnasium's seed-once-per-session convention, applied per task). A whole task is reproducible from its seed while episode-level randomness still varies; envs that regenerate their problem instance when `reset` receives an explicit seed present the same instance to every episode in the task. Previously a fresh seed was drawn and passed on every episode reset.
- Step output dict key order is now `task_index`, `episode_index`, `step_index`, `reward`, `task_done`, `episode_done`, `observation`, `info`.
- `EnvConfig.episodes_per_task` renamed to `max_task_episodes` (task-length cap, parallel to Gymnasium's `max_episode_steps`).
- `EnvConfig` now raises on `kwargs` or `render` combined with `env_fn` (both apply to `id` configs only) and on negative `max_task_episodes`, instead of silently ignoring them.

### Removed
- `EnvConfig.reset_reward`. Reset-frame reward is `0` unless
  `reward_transform` returns a value when `step_index` is `0`.

### Fixed
- `scripts/install.sh` `cd`s to the repo root so it works from any cwd, prints the venv activate hint, and uses `return` throughout so a sourced failure does not kill the calling shell.
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
