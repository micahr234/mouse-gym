# Mouse Gym 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-gym/main/mouse-gym.png" width="400"/></p>

> **Warning:** Mouse Gym is in early development and is not yet ready for production use. APIs may change without notice.

Imagine an agent deployed to real users. It attempts a task, gets feedback, and learns something useful about how to complete that task. Then the session ends. Some time later, a similar situation recurs. But in the standard episodic setup, both the agent and the environment are treated as if they are starting from scratch. The task begins again, the history is gone, and the agent has to repeat the same learning process all over again. It must relearn what it already discovered because the continuity between interactions has been erased.

Humans do not learn this way. In the real world, people improve after attempting tasks multiple times. They learn on-the-job, make mistakes, adapt, and become better as they work. If we want agents that do not feel like retraining a new hire every single day, we have to train them to improve as related experience accumulates.

That shifts the focus from **zero-shot performance** to **few-shot improvement**. The question is no longer how well an agent does the first time it sees a problem, but how quickly it gets better when a similar situation comes around again. To measure that, you have to watch performance across repeated episodes, not judge a single isolated trial.

Gymnasium's standard loop treats every episode as an isolated trial: reset the environment, rollout the episode out, repeat. **mouse-gym** groups consecutive episodes on the same environment into a **task** and runs the whole thing as one continuous stream. You only ever call `step()`, and episode and task boundaries show up as fields in the output.


## News

- **2026-07-07 — NumPy I/O** Step inputs and outputs use NumPy arrays (Gymnasium-native types). PyTorch is no longer required.
- **2026-07-02 — `tracker` → `metrics`** Renamed `env.tracker` / `Tracker` / `GroupTracker` to `env.metrics` / `Metrics` / `GroupMetrics`. Example notebook renamed to [04 — Metrics](examples/04_metrics.ipynb).
- **2026-06-29 — Repository split** Mouse Gym is now its own package. Reset-free rollout infrastructure lives here; custom environment implementations live elsewhere.
- **2026-06-26 — `SingleEnv` / `GroupEnv`** `make_env` returns one env; `make_group_env` handles parallel rollouts.

See [CHANGELOG.md](CHANGELOG.md) for the full release history.


## Install

**Requirements:** Python 3.12+, Gymnasium ≥ 1.3, NumPy ≥ 1.26 (NumPy is also pulled in by Gymnasium).

```bash
pip install mouse-gym
```

For development:

```bash
git clone https://github.com/micahr234/mouse-gym.git
cd mouse-gym
source scripts/install.sh
```


## Quick start

```python
from mouse_gym import EnvConfig, make_env

cfg = EnvConfig(
    id="CartPole-v1",
    reset_seed=0,
    episodes_per_task=5,
)
env = make_env(cfg)

for _ in range(1000):
    output = env.step(env.sample_random_input())

print(env.metrics.episode_cum_rewards)
env.close()
```

## Differences from Gymnasium

**mouse-gym** builds on [Gymnasium](https://gymnasium.farama.org/) — you wrap ordinary Gymnasium envs and keep their observations, rewards, and spaces. What's different is the rollout interface:

- **Reset-free rollouts.** Only call `step()` — `SingleEnv.reset()` is not part of the API, including at task boundaries. When an episode or task ends, the next `step()` returns a **reset frame**: initial observation, `time=0`, `done=0`, `reset_reward`; the input is ignored. (Mouse-gym calls the underlying env's Gymnasium `reset()` internally on that step — you never call it yourself.) One continuous loop; boundaries are fields in the output, not a separate `reset()` before `step()`.

- **Dict I/O instead of Gymnasium's action + tuple.** Pass `{"action": ...}` to `step()` (use `sample_random_input()` for random rollouts). Each `step()` returns one dict with named fields — `observation`, `reward`, `done`, `time`, `episode_index`, `task_index`, and `info` — rather than `(observation, reward, terminated, truncated, info)`.

- **Tasks group episodes.** A task is a consecutive run of episodes — length set by `episodes_per_task` in `EnvConfig` (default `0`: no task boundary). When a task ends, the next `step()` is a reset frame with `task_index` incremented. `done` codes `3`/`4` mark the last step of a task.

- **`done` replaces `terminated` / `truncated`.** One integer field per step instead of two booleans:
  - `0` — **Running.** A normal mid-episode step, or a reset frame (`time=0`, `reward=reset_reward`).
  - `1` — **Episode terminated.** Underlying env returned `terminated=True` and this is not the last episode in the task. Do not bootstrap; the next `step()` is a reset frame (next episode, same task).
  - `2` — **Episode truncated.** Underlying env returned `truncated=True` and this is not the last episode in the task. Same semantics as `1`.
  - `3` — **Task terminated.** `terminated=True` on the last episode in the task (per `episodes_per_task`). Bootstrap here; the next `step()` is a reset frame that starts a new task (`task_index` increments).
  - `4` — **Task truncated.** `truncated=True` on the last episode in the task. Same bootstrap semantics as `3`.
  - With `episodes_per_task=0` (default), codes `3` and `4` never fire. Gymnasium has no task grouping or equivalent codes.

## Additions to Gymnasium

On top of the standard env API, mouse-gym adds:

- **Metrics on the env.** Episode returns and lengths accumulate in `env.metrics`, not in the `step()` return value. See [04 — Metrics](examples/04_metrics.ipynb).

- **Grouped envs.** `GroupEnv` steps multiple `SingleEnv` instances sequentially in one `step()` call and returns a flat `list[dict]` — useful for mixed or multi-task setups without a vectorized wrapper. See [02 — Multiple envs](examples/02_multi_env.ipynb).

- **Input/output specs.** `input_spec` and `output_spec` describe the construction-time contract for step dict shapes and dtypes (on `GroupEnv`, `input_specs[i]` and `output_specs[i]`). See [01 — Random rollout](examples/01_random_rollout.ipynb).

## Examples

The notebooks in [`examples/`](examples/) are the detailed reference for `EnvConfig`, input/output fields, and day-to-day usage. Install notebook dependencies with `pip install "mouse-gym[examples]"`, then work through them in order:

**[01 — Random rollout](examples/01_random_rollout.ipynb)** — Start here. Build an env from `EnvConfig`, run the reset-free `step()` loop, and inspect what comes back on each call: input dict (`action`), output dict (`observation`, `reward`, `done`, `time`, `episode_index`, `task_index`, `info`), reset frames, and `done` codes. Also covers `input_spec` / `output_spec` and optional `env_fn` factories.

**[02 — Multiple envs](examples/02_multi_env.ipynb)** — Combine several env instances with `make_group_env`. Step heterogeneous envs (different ids, spaces, and seeds) in one sequential loop; read flat `list[dict]` inputs and outputs; use `env.names`, `input_specs[i]`, and `output_specs[i]`.

**[03 — RNG seeding control](examples/03_rng_seeding_control.ipynb)** — Reproduce or vary behavior with `reset_seed` (internal reset stream) and `env.action_space.seed()` (random action sampling), independently.

**[04 — Metrics](examples/04_metrics.ipynb)** — Read episode returns and lengths from `env.metrics`, clear between eval runs, and aggregate stats across a `GroupEnv`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
