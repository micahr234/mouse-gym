# Mouse Gym 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-gym/main/mouse-gym.png" width="400"/></p>

> **Warning:** Mouse Gym is in early development and is not yet ready for production use. APIs may change without notice.

In the real world, experience is not usually divided into clean, isolated episodes with hard stops. Instead, it is closer to a continuous stream of experience, where similar situations can repeat over time.

Most reinforcement learning environments, however, are episodic: the agent acts until the episode ends, the caller invokes `reset()` each time.

**mouse-gym** was created to bridge that gap. It provides a standardized environment interface for turning episodic Gymnasium environments into continuing, reset-free streams of experience, while still making the underlying episode and task structure visible.

Many reinforcement learning experiments care about behavior **across episodes**, not just within a single episode. You could stitch episodes together manually on top of Gymnasium, but then several important design choices become ad hoc:

- whether reset observations are included in the data
- how episode boundaries are marked
- how task boundaries are defined

**mouse-gym** makes these choices explicit and consistent.



## News

- **2026-06-29 — Repository split** Mouse Gym is now its own package. Reset-free rollout infrastructure lives here; custom environment implementations live elsewhere.
- **2026-06-26 — `SingleEnv` / `GroupEnv`** `make_env` returns one env; `make_group_env` handles parallel rollouts.

See [CHANGELOG.md](CHANGELOG.md) for the full release history.


## Install

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

print(env.tracker.episode_cum_rewards)
env.close()
```


## Examples

Runnable notebooks in [`examples/`](examples/):

| Notebook | What it covers |
|----------|----------------|
| [01 — Random rollout](examples/01_random_rollout.ipynb) | End-to-end loop; output fields; `done` codes; reset frames; `EnvConfig`; `input_spec`/`output_spec` |
| [02 — Multiple envs](examples/02_multi_env.ipynb) | `make_group_env`; heterogeneous specs; env instance names |
| [03 — RNG seeding control](examples/03_rng_seeding_control.ipynb) | `reset_seed`; reproducible internal resets and action-space sampling |
| [04 — Tracker](examples/04_metrics_tracker.ipynb) | `env.tracker`; episode returns and lengths; `clear()` between eval runs; multi-env aggregation |


## Core API

Mouse Gym wraps ordinary Gymnasium environments. Under the hood you still get standard `action_space` and `observation_space` objects, native dtypes, and any env built with `gymnasium.make` or your own `env_fn` factory.

**The rollout interface follows Gymnasium conventions except for the points below.** If you are coming from `gymnasium.Env`, read this section first.

### Differences from Gymnasium

- **No public `reset()`.** `reset()` raises `NotImplementedError`. The first `step()` performs an internal reset and returns the initial observation. After an episode ends, the next `step()` emits a reset frame before normal stepping resumes.
- **`step()` takes a dict, not a raw action.** Pass `{"action": tensor}`. Use `env.sample_random_input()` to draw random actions from the underlying action space.
- **`step()` returns a dict, not a tuple.** Each output contains `observation`, `reward`, `done`, `time`, `episode_index`, `task_index`, and `info` — not `(obs, reward, terminated, truncated, info)`.
- **`done` is an integer code, not separate booleans.** Gymnasium's `terminated` / `truncated` pair is replaced by one field:
  - `0` — **Running.** A normal mid-episode step, or a reset frame at the start of a new episode/task (`time=0`, `reward=reset_reward`). No boundary; keep stepping.
  - `1` — **Episode terminated.** The underlying env returned `terminated=True` and this is not the last episode in the task. The episode ended naturally; the task continues — do not bootstrap. The next `step()` returns a reset frame (`done=0`).
  - `2` — **Episode truncated.** The underlying env returned `truncated=True` and this is not the last episode in the task. Same semantics as `1` (time limit, etc.); the task continues; the next `step()` is a reset frame.
  - `3` — **Task terminated.** `terminated=True` on the last episode in the task (per `episodes_per_task`). Episode and task both ended — bootstrap here. The next `step()` starts a new task (`task_index` increments).
  - `4` — **Task truncated.** `truncated=True` on the last episode in the task. Same bootstrap semantics as `3`.
  - With `episodes_per_task=0` (default), codes `3` and `4` never fire automatically.
- **Reset frames are part of the output stream.** When a new episode starts, the next output has `time=0`, `done=0`, and `reset_reward` (default `0`) instead of the underlying env's step reward.
- **Construction via `EnvConfig`.** Build envs with `make_env(cfg)` or `make_group_env([cfg, ...])` instead of calling `gym.make` directly. Pass any Gymnasium env id, or supply `env_fn` for custom or pre-wrapped envs. Use `episodes_per_task` to group consecutive episodes into one task.
- **`env_fn` for custom envs.** A zero-arg factory returning a Gymnasium env. Use this for third-party envs, preprocessing wrappers, or any setup you would normally do before stepping.
- **Seeding and reset options.** `EnvConfig.reset_seed` controls the internal `env.reset(seed=...)` stream. `episode_reset_options` and `task_reset_options` are forwarded to internal resets (task options overlay episode options on task starts). Random action sampling still uses the normal Gymnasium API: `env.action_space.seed(...)`.
- **Episode stats live on `env.tracker`.** Cumulative returns and lengths accumulate automatically and are kept separate from the per-step output stream. Call `env.tracker.clear()` between evaluation runs.

### Types

| | `SingleEnv` | `GroupEnv` |
|---|---|---|
| Factory | `make_env(EnvConfig)` | `make_group_env(list[EnvConfig])` |
| `step(...)` | `dict → dict` | `list[dict] → list[dict]` |
| `sample_random_input()` | `dict` | `list[dict]` |
| `tracker` | `Tracker` | `GroupTracker` |
| Name | `env.name` | `env.names` |
| Specs | `input_spec`, `output_spec` | `input_specs[i]`, `output_specs[i]` |

On `GroupEnv`, `action_space` and `observation_space` are Gymnasium `Tuple` spaces over the underlying envs.

### Output fields

Each step (including reset frames) returns:

| Field | Description |
|-------|-------------|
| `observation` | Observation from the underlying env (native dtype preserved) |
| `reward` | Per-step reward from the underlying env; `reset_reward` on reset frames |
| `done` | Integer boundary code (`0`–`4`; see differences above) |
| `time` | Step index within the current episode; `0` on reset frames |
| `episode_index` | Index of the current episode within the task |
| `task_index` | Index of the current task |
| `info` | Underlying Gymnasium `info` dict |

### `EnvConfig`

Required: `id`, `reset_seed`.

| Field | Default | Purpose |
|-------|---------|---------|
| `episodes_per_task` | `0` | Episodes per task; `0` = unlimited |
| `name` | `None` | Display name (`id` if unset) |
| `kwargs` | `None` | Forwarded to `gymnasium.make` |
| `env_fn` | `None` | Factory replacing `gym.make` |
| `reset_reward` | `0.0` | Reward on reset frames |
| `episode_reset_options` | `None` | Passed to every internal `reset(options=...)` |
| `task_reset_options` | `None` | Overlaid on task-start resets |
| `render` | `False` | Enable `"human"` render mode |

Example with a custom factory:

```python
import gymnasium as gym
from mouse_gym import EnvConfig, make_env

def make_cartpole():
    env = gym.make("CartPole-v1", max_episode_steps=500)
    return MyWrapper(env)

cfg = EnvConfig(id="my-cartpole", reset_seed=0, episodes_per_task=5, env_fn=make_cartpole)
env = make_env(cfg)
```

### Tracker

```python
# SingleEnv
env.tracker.episode_cum_rewards   # list[float]
env.tracker.episode_lengths       # list[float]

# GroupEnv
env.tracker.episode_cum_rewards   # list[list[float]]
env.tracker.episode_lengths       # list[list[float]]

env.tracker.clear()
```


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).


## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
