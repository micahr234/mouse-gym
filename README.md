# Mouse Gym

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-gym/main/mouse-gym.png" width="400"/></p>

**mouse-gym** wraps any [Gymnasium](https://gymnasium.farama.org/) environment in a reset-free rollout. You call `step()` in one loop. When the wrapped env reports `terminated` or `truncated`, the next `step()` returns a reset frame. Observations, rewards, and spaces stay those of the env you passed in. Parallel groups, episode metrics, and a small config for spinning up envs live here.


## News

- **2026-10-07 — Gymnasium step** `step()` takes an action and returns `(observation, reward, terminated, truncated, info)`. `env.step_index` is `0` on a reset frame. `EnvConfig.seed_mode` is `constant`, `stream`, or `none`. Task grouping left mouse-gym: `task_done`, `episode_done`, `max_task_episodes`, and `terminate_task` are gone. Repeating episodes inside a task is [repeat-task-gym](https://github.com/micahr234/repeat-task-gym) — build a `RepeatTaskEnv` and pass it through `env_fn`. See [CHANGELOG.md](CHANGELOG.md).
- **2026-09-17 — 1.1.0** Reward and task boundaries were callables on `EnvConfig`: `reward_transform`, `terminate_task`, and `max_task_episodes`. See [CHANGELOG.md](CHANGELOG.md).
- **2026-08-18 — 1.0.0** First stable release. Public step contract is `episode_done` / `task_done` on a reset-free `step()` stream.
- **2026-07-07 — NumPy I/O** Step inputs and outputs use NumPy arrays (Gymnasium-native types). PyTorch is no longer required.
- **2026-07-02 — `tracker` → `metrics`** Renamed `env.tracker` / `Tracker` / `GroupTracker` to `env.metrics` / `Metrics` / `GroupMetrics`. See [03 — Metrics](examples/03_metrics.ipynb).
- **2026-06-29 — Repository split** Mouse Gym is now its own package. Reset-free rollout infrastructure lives here; custom environment implementations live elsewhere.
- **2026-06-26 — `SingleEnv` / `GroupEnv`** `make_env` returns one env; `make_group_env` handles parallel rollouts.

See [CHANGELOG.md](CHANGELOG.md) for the full release history.


## Install

**Requirements:** Python 3.14+ (free-threaded **3.14t** recommended for `GroupEnv` threads), Gymnasium ≥ 1.3, NumPy ≥ 2.5 (NumPy is also pulled in by Gymnasium).

```bash
pip install mouse-gym
```

For development (free-threaded Python **3.14t** via uv):

```bash
git clone https://github.com/micahr234/mouse-gym.git
cd mouse-gym
source scripts/install.sh
```


## Quick start

```python
from mouse_gym import EnvConfig, make_env

env = make_env(EnvConfig(id="CartPole-v1", seed=0))

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

print(env.metrics.episode_cum_rewards)
env.close()
```

## Differences from Gymnasium

**mouse-gym** builds on Gymnasium — you wrap ordinary Gymnasium envs and keep their observations, rewards, and spaces. What's different is the rollout interface:

- **Reset-free rollouts.** Only call `step()`. `SingleEnv.reset()` is not part of the API. `step(action)` returns Gymnasium's `(observation, reward, terminated, truncated, info)`. The first call is a **reset frame**: initial observation, `env.step_index == 0`, `terminated=False`, `truncated=False`; the action is ignored. After the wrapped env reports `terminated` or `truncated`, the next `step()` is another reset frame. Reset-frame reward comes from `reward_transform` (detect a reset with `step_index == 0`) or `0` if no transform is set. Mouse-gym calls the underlying env's Gymnasium `reset()` internally on that step. The observation and `info` dict are forwarded unchanged.

- **Reset seeds.** `seed_mode` chooses the `seed` passed to every internal `reset`. `constant` (default) repeats `seed`. `stream` uses `seed` to start an RNG and draws a new reset seed from it each time. `none` passes `None` (`seed` must be left unset). `reset_options` is forwarded on every reset in all three modes.

## Additions to Gymnasium

On top of the standard env API, mouse-gym adds:

- **Metrics on the env.** When `terminated` or `truncated` is true, the return and length are appended to `env.metrics.episode_cum_rewards` and `env.metrics.episode_lengths`. Those flags mean whatever the wrapped env uses them for. `EnvConfig.reward_transform` is applied to each env step and reset-frame reward before it appears in the step output and before those sums are recorded. It receives `step_index`, `state`, `action`, `reward`, `terminated`, `truncated`, and `next_state`. See [03 — Metrics](examples/03_metrics.ipynb).

- **Grouped envs.** `GroupEnv` steps multiple `SingleEnv` instances in one `step()` call. Pass a list or tuple of actions (`action_space.sample()` returns a tuple) and get back a flat list of `(observation, reward, terminated, truncated, info)` tuples. By default (`max_threads=0`) stepping is sequential on the calling thread; set `max_threads > 0` to distribute envs across that many worker threads. On free-threaded CPython (`3.14t`) those workers can run env steps in parallel. On a GIL-enabled interpreter, cheap Python envs such as CartPole are usually faster with `max_threads=0`. See [02 — Multiple envs](examples/02_multi_env.ipynb). `bench/bench_env.py` times `step` for `SingleEnv` and `GroupEnv` widths.

## Examples

The notebooks in [`examples/`](examples/) are the detailed reference for `EnvConfig`, the step return, and day-to-day usage. Install notebook dependencies with `pip install "mouse-gym[examples]"`. That extra installs Gymnasium's classic-control extra (`pygame-ce`) for CartPole rendering, plus `repeat-task-gym` and `procedural-frozenlake` for the repeat-task notebook. Then work through them in order:

**[01 — Random rollout](examples/01_random_rollout.ipynb)** — Start here. Build an env from `EnvConfig`, run the reset-free `step()` loop, and inspect what comes back on each call: an action in, `(observation, reward, terminated, truncated, info)` out, `env.step_index`, and reset frames. Also covers `action_space` / `observation_space` and optional `env_fn` factories.

**[02 — Multiple envs](examples/02_multi_env.ipynb)** — Combine several env instances with `make_group_env`. Step heterogeneous envs (different ids, spaces, and seeds) in one loop; optionally parallelize with `max_threads`; pass one action per env and read a flat list of Gymnasium tuples; use `env.names` and each env's spaces.

**[03 — Metrics](examples/03_metrics.ipynb)** — Read returns and lengths from `env.metrics`, clear between eval runs, aggregate stats across a `GroupEnv`, and shape rewards with `reward_transform`.

**[04 — Repeat a task](examples/04_repeat_task_env.ipynb)** — Wrap [Procedural Frozen Lake](https://github.com/micahr234/procedural-frozenlake) in a `RepeatTaskEnv` and pass it through `env_fn`. Episode ends stay in the observation. Gymnasium `terminated` and `truncated` mark the task, which is what `env.metrics` records. `reset_options={"regenerate_map": True}` asks for a new lake when mouse-gym starts the next task.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
