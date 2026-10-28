# Mouse Gym

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-gym/main/mouse-gym.png" width="400"/></p>

**mouse-gym** wraps any [Gymnasium](https://gymnasium.farama.org/) environment in a reset-free rollout. You call `step()` in one loop. When the wrapped env reports `terminated` or `truncated`, the next `step()` returns a reset frame. Observations, rewards, and spaces stay those of the env you passed in. Parallel groups, standalone rollout trackers, and a small config for spinning up envs live here.


## News

- **2026-10-28 — Standalone rollout trackers** `RolloutTracker` collects ordinary MDP/POMDP progress and episode summaries. `RepeatTaskTracker` also understands inner episodes and task summaries. Callers own and update trackers explicitly. `Metrics`, `GroupMetrics`, `env.metrics`, and `env.step_index` have been removed.

- **2026-10-07 — Gymnasium step** `step()` takes an action and returns `(observation, reward, terminated, truncated, info)`. `env.step_index` is `0` on a reset frame. `EnvConfig.seed_mode` is `constant`, `stream`, or `none`. Task grouping left mouse-gym: `task_done`, `episode_done`, `max_task_episodes`, and `terminate_task` are gone. Repeating episodes inside a task is [repeat-task-gym](https://github.com/micahr234/repeat-task-gym) — build a `RepeatTaskEnv` and pass it through `env_fn`. See [CHANGELOG.md](CHANGELOG.md).
- **2026-09-17 — 1.1.0** Reward and task boundaries were callables on `EnvConfig`: `reward_transform`, `terminate_task`, and `max_task_episodes`. See [CHANGELOG.md](CHANGELOG.md).
- **2026-08-18 — 1.0.0** First stable release. Public step contract is `episode_done` / `task_done` on a reset-free `step()` stream.
- **2026-07-07 — NumPy I/O** Step inputs and outputs use NumPy arrays (Gymnasium-native types). PyTorch is no longer required.
- **2026-07-02 — `tracker` → `metrics`** Renamed `env.tracker` / `Tracker` / `GroupTracker` to `env.metrics` / `Metrics` / `GroupMetrics`. See [03 — Metrics](examples/03_metrics.ipynb).
- **2026-06-29 — Repository split** Mouse Gym is now its own package. Reset-free rollout infrastructure lives here; custom environment implementations live elsewhere.
- **2026-06-26 — `SingleEnv` / `GroupEnv`** `make_env` returns one env; `make_group_env` handles parallel rollouts.

See [CHANGELOG.md](CHANGELOG.md) for the full release history.


## Install

**Requirements:** Python 3.15+ (free-threaded **3.15t** recommended for `GroupEnv` threads), Gymnasium ≥ 1.4, NumPy ≥ 2.5.4 (NumPy is also pulled in by Gymnasium).

```bash
pip install mouse-gym
```

For development (free-threaded Python **3.15t** via uv):

```bash
git clone https://github.com/micahr234/mouse-gym.git
cd mouse-gym
source scripts/install.sh
```


## Quick start

```python
from mouse_gym import EnvConfig, RolloutTracker, make_env

env = make_env(EnvConfig(id="CartPole-v1"))
tracker = RolloutTracker()

for _ in range(1000):
    action = env.action_space.sample() if tracker.action_requested else None
    output = env.step(action)
    tracker.update(output=output)

print([episode.reward for episode in tracker.episodes])
env.close()
```

## Differences from Gymnasium

**mouse-gym** builds on Gymnasium — you wrap ordinary Gymnasium envs and keep their observations, rewards, and spaces.

- **Reset-free rollouts.** Only call `step()`. `SingleEnv.reset()` is not part of the API. `step(action)` returns Gymnasium's `(observation, reward, terminated, truncated, info)`. The first call is a **reset frame**: initial observation, `tracker.step_index == 0`, `terminated=False`, `truncated=False`; the action is ignored. After the wrapped env reports `terminated` or `truncated`, the next `step()` is another reset frame. Reset-frame reward comes from `reward_transform` (detect a reset with `step_index == 0`) or `0` if no transform is set. Mouse-gym calls the underlying env's Gymnasium `reset()` internally on that step. The observation and `info` dict are deep-copied before they are returned.

- **Reset seeds.** `seed_mode` chooses the `seed` passed to every internal `reset`. `none` (default) passes `None` (`seed` must be left unset). `constant` repeats `seed`. `stream` uses `seed` to start an RNG and draws a new reset seed from it each time. Each reset receives a deep copy of `reset_options`, taken from a snapshot captured when the env is built.

- **Executed actions.** `info["executed_action"]` records the supplied action on ordinary steps and `None` on reset frames. This reserved field also preserves `None` from an inner reset in a wrapper such as `RepeatTaskEnv`. Reward transforms receive `action=None` for both outer and inner resets. Record this field instead of echoing an ignored input action.

- **Separate rollout tracking.** Create a `RolloutTracker` for ordinary MDP/POMDP episodes, or a `RepeatTaskTracker` for a `RepeatTaskEnv`. Call `tracker.update(output=output)` once for every returned frame, starting with the initial reset. The tracker exposes the latest observation, flags, counters, cumulative reward, and `action_requested`. Ordinary completed episodes appear in `tracker.episodes` as immutable `EpisodeSummary` records with return, length, and termination/truncation flags. Repeated trackers collect inner episodes in `episodes` and outer tasks in `tasks`. `tracker.clear()` clears completed summaries while preserving the active rollout. Environments do not own trackers.

- **Reward transforms.** `EnvConfig.reward_transform` runs before the output reaches the tracker, so summaries include transformed rewards, including reset-frame rewards. It receives `step_index`, `state`, `action`, `reward`, `terminated`, `truncated`, and `next_state`. `state` is the reset observation on the first reset frame, and the previous observation on subsequent frames. Reset-frame `reward` is `0.0`, with both flags false. See [03 — Rollout tracking](examples/03_metrics.ipynb).

- **Grouped envs.** `GroupEnv` steps multiple `SingleEnv` instances in one `step()` call. Pass a list or tuple of actions (`action_space.sample()` returns a tuple) and get back a tuple of `(observation, reward, terminated, truncated, info)` results, one per env. Each `SingleEnv` belongs to one group. By default (`max_threads=0`) stepping is sequential on the calling thread; set `max_threads > 0` to distribute envs across that many worker threads. On free-threaded CPython (`3.15t`) those workers can run env steps in parallel. On a GIL-enabled interpreter, cheap Python envs such as CartPole are usually faster with `max_threads=0`. See [02 — Multiple envs](examples/02_multi_env.ipynb). `bench/bench_env.py` times `step` for `SingleEnv` and `GroupEnv` widths. To collect progress and summaries, keep one tracker per env and update each with its corresponding output.

## Examples

The notebooks in [`examples/`](examples/) are the detailed reference for `EnvConfig`, the step return, and day-to-day usage. Install them with `pip install "mouse-gym[examples]"`. That extra installs Jupyter, ipykernel, and Matplotlib, Gymnasium's classic-control extra (`pygame-ce`) for CartPole rendering, plus `repeat-task-gym` and `procedural-frozenlake` for the repeat-task notebook. Then work through them in order:

**[01 — Random rollout](examples/01_random_rollout.ipynb)** — Start here. Build an env from `EnvConfig`, run the reset-free `step()` loop, and inspect what comes back on each call: an action in, `(observation, reward, terminated, truncated, info)` out, `tracker.step_index`, and reset frames. Also covers `action_space` / `observation_space` and optional `env_fn` factories.

**[02 — Multiple envs](examples/02_multi_env.ipynb)** — Combine several env instances with `make_group_env`. Step heterogeneous envs (different ids, spaces, and seeds) in one loop; optionally parallelize with `max_threads`; pass one action per env and read a tuple of Gymnasium results; use `env.names` and each env's spaces.

**[03 — Rollout tracking](examples/03_metrics.ipynb)** — Collect returns and lengths with a separate tracker, clear completed summaries between evaluation windows, aggregate results across a `GroupEnv`, and shape rewards with `reward_transform`.

**[04 — Repeat a task](examples/04_repeat_task_env.ipynb)** — Wrap [Procedural Frozen Lake](https://github.com/micahr234/procedural-frozenlake) in a `RepeatTaskEnv` and pass it through `env_fn`. Pair it with a separate `RepeatTaskTracker`. Gymnasium `terminated` and `truncated` mark the task; episode boundaries stay in the observation. Call `step(None)` whenever `tracker.action_requested` is false, including the interval between episodes. Episode resets pass `regenerate_map: False`. `reset_options={"regenerate_map": True}` asks for a new lake when mouse-gym starts the next task.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

`RolloutTracker` treats Gymnasium `terminated` and `truncated` as episode
boundaries. `RepeatTaskTracker` treats those flags as task boundaries and
uses the wrapped observation's inner episode flags to track inner progress.
`RepeatTaskEnv` remains responsible for repeating episodes and choosing reset
policy; the standalone tracker only interprets outputs and collects summaries.
