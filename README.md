# Mouse Gym 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-gym/main/mouse-gym.png" width="400"/></p>

> **Warning:** Mouse Gym is in early development and is not yet ready for production use. APIs may change without notice.

Imagine an agent deployed to real users. Today, it attempts a task, gets feedback, and learns something useful about how to complete that task. Then the session ends.

Tomorrow, a similar situation recurs. But in the standard episodic setup, both the agent and the environment are treated as if they are starting from scratch. The task begins again, the history is gone, and the agent has to repeat the same learning process all over again. It may relearn what it discovered yesterday, but it cannot leverage yesterday's experience directly because the continuity between interactions has been erased.

Humans do not learn this way. In the real world, people improve after attempting the task multiple times. They get on-the-job training, make mistakes, adapt, and become better at their jobs as they accumulate experience. Learning systems should be studied in the same way: not only by asking how well they perform immediately, but also by asking how quickly they improve as related experience accumulates.

That shifts the focus from **zero-shot performance** — "How good is the agent immediately?" — to **few-shot improvement** — "How quickly does the agent get better as similar situations repeat?" To study that kind of learning, we need to look across episodes, not just within a single episode.

Most Gymnasium environments are episodic. A typical program calls `reset()`, then calls `step()` until the episode ends, then calls `reset()` again. This is a good interface when each episode is meant to be an independent trial. But for continual learning, in-context reinforcement learning, or repeated-task experiments, we often want a longer stream of experience where episodes happen inside a larger task.

**mouse-gym** provides that stream. It wraps standard Gymnasium environments so the user only calls `step()`. Episode resets happen internally, but the output still makes episode and task boundaries visible. This gives the agent a continuous experience stream while keeping the underlying structure explicit.


## News

- **2026-07-02 — `tracker` → `metrics`** Renamed `env.tracker` / `Tracker` / `GroupTracker` to `env.metrics` / `Metrics` / `GroupMetrics`. Example notebook renamed to [04 — Metrics](examples/04_metrics.ipynb).
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

print(env.metrics.episode_cum_rewards)
env.close()
```

## Examples

Runnable notebooks in [`examples/`](examples/):

| Notebook | What it covers |
|----------|----------------|
| [01 — Random rollout](examples/01_random_rollout.ipynb) | End-to-end loop; output fields; `done` codes; reset frames; `EnvConfig`; `input_spec`/`output_spec` |
| [02 — Multiple envs](examples/02_multi_env.ipynb) | `make_group_env`; heterogeneous specs; env instance names |
| [03 — RNG seeding control](examples/03_rng_seeding_control.ipynb) | `reset_seed`; reproducible internal resets and action-space sampling |
| [04 — Metrics](examples/04_metrics.ipynb) | `env.metrics`; episode returns and lengths; `clear()` between eval runs; multi-env aggregation |


## Differences from Gymnasium

Gymnasium's contract is episodic: call `reset()` to get the first observation, then call `step(action)` with a raw action from `action_space` until the episode ends, then call `reset()` again. Each `step()` returns `(observation, reward, terminated, truncated, info)`.

mouse-gym replaces that with a reset-free rollout protocol. Only the points below differ from standard Gymnasium usage:

- **No public `reset()`.** `SingleEnv.reset()` raises `NotImplementedError`. Gymnasium requires `reset()` before the first step and again after every episode; mouse-gym performs those resets internally inside `step()`.

- **Reset frames come from `step()`, not `reset()`.** The first `step()` and every `step()` immediately after an episode ends return a reset frame: the initial observation with `time=0`, `done=0`, and `reset_reward`. The input dict is ignored on these calls. In Gymnasium, that observation comes from `reset()`, outside the step loop.

- **Two-step episode boundaries.** When an episode ends, one `step()` returns the terminal transition with `done` in `1`–`4`. The next `step()` is a separate reset frame with `done=0`. In Gymnasium, the terminal step sets `terminated` or `truncated`; the caller then calls `reset()` separately.

- **`step()` takes a dict, not a raw action.** Pass `{"action": tensor}`. Gymnasium `step()` accepts an action value directly from `action_space`. Use `sample_random_input()` for random rollouts.

- **`step()` returns a dict, not a five-tuple.** Each output contains `observation`, `reward`, `done`, `time`, `episode_index`, `task_index`, and `info` — not `(observation, reward, terminated, truncated, info)`.

- **`done` replaces `terminated` / `truncated`.** One integer field per step instead of two booleans:
  - `0` — **Running.** A normal mid-episode step, or a reset frame (`time=0`, `reward=reset_reward`).
  - `1` — **Episode terminated.** Underlying env returned `terminated=True` and this is not the last episode in the task. Do not bootstrap; the next `step()` is a reset frame.
  - `2` — **Episode truncated.** Underlying env returned `truncated=True` and this is not the last episode in the task. Same semantics as `1`.
  - `3` — **Task terminated.** `terminated=True` on the last episode in the task (per `episodes_per_task`). Bootstrap here; the next `step()` starts a new task (`task_index` increments).
  - `4` — **Task truncated.** `truncated=True` on the last episode in the task. Same bootstrap semantics as `3`.
  - With `episodes_per_task=0` (default), codes `3` and `4` never fire. Gymnasium has no task grouping or equivalent codes.

- **Extra output fields.** Every step includes `time` (step index within the episode), `episode_index`, and `task_index`. Gymnasium step outputs do not include these.

- **Tensor outputs.** `observation`, `reward`, `done`, and `time` are `torch.Tensor` values; `episode_index` and `task_index` are plain `int`. Gymnasium envs typically return NumPy arrays and Python scalars.

## Types

| | `SingleEnv` | `GroupEnv` |
|---|---|---|
| Factory | `make_env(EnvConfig)` | `make_group_env(list[EnvConfig])` or `GroupEnv(list[SingleEnv])` |
| `step(...)` | `dict → dict` | `list[dict] → list[dict]` |
| `sample_random_input()` | `dict` | `list[dict]` |
| `metrics` | `Metrics` | `GroupMetrics` (read-through; stores no data of its own) |
| Name | `env.name` | `env.names` |
| Specs | `input_spec`, `output_spec` | `input_specs[i]`, `output_specs[i]` |
| Spaces | `action_space`, `observation_space` | `Tuple` spaces; subspace `i` is `env.action_space.spaces[i]` |
| Lifecycle | `render()`, `close()` | `render()` (flattened frames), `close()` |
| Constituents | — | `env.envs`, `env.num_envs` |

On `GroupEnv`, `action_space` and `observation_space` are Gymnasium `Tuple` spaces over the underlying envs. Build envs with `make_env(cfg)` or `make_group_env([cfg, ...])`, or construct `GroupEnv([env_a, env_b, ...])` directly from existing `SingleEnv` instances. Overlapping groups that share the same `SingleEnv` objects are safe.

## Output fields

Each step (including reset frames) returns:

| Field | Description |
|-------|-------------|
| `observation` | Observation from the underlying env (`torch.Tensor`, or `dict[str, torch.Tensor]` for Dict spaces) |
| `reward` | Per-step reward from the underlying env (`torch.float32` tensor); `reset_reward` on reset frames |
| `done` | Integer boundary code (`0`–`4`; see differences above), as `torch.int64` tensor |
| `time` | Step index within the current episode; `0` on reset frames (`torch.int64` tensor) |
| `episode_index` | Index of the current episode within the task |
| `task_index` | Index of the current task |
| `info` | Underlying Gymnasium `info` dict |

Outputs do not include `action`, `name`, or other input-side fields. For Dict observation spaces, the underlying Gymnasium dict is returned under the `observation` key (as `dict[str, torch.Tensor]`).

## EnvConfig

Build envs with `make_env(cfg)` or `make_group_env([cfg, ...])`. Each `EnvConfig` creates one env instance. Pass any Gymnasium env id, or supply `env_fn` — a zero-arg factory returning a Gymnasium env — for custom or pre-wrapped envs. When `env_fn` is set, `id` is used only for naming.

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
| `render` | `False` | Sets `render_mode="human"` when not already in `kwargs` |

`reset_seed` controls the internal `env.reset(seed=...)` stream (you never call `reset()` yourself). `episode_reset_options` and `task_reset_options` are forwarded to internal resets; task options overlay episode options on task starts. Random action sampling uses the normal Gymnasium API: `env.action_space.seed(...)`.

`input_spec` and `output_spec` describe the construction-time contract for input/output dict shapes and dtypes. On `GroupEnv`, use `input_specs[i]` and `output_specs[i]`.

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

## Metrics

Episode statistics accumulate when an episode completes (`done` in `1`–`4`) and are kept separate from the per-step output stream. Values are raw cumulative rewards from the underlying env (unscaled). Call `env.metrics.clear()` between evaluation runs.

```python
# SingleEnv
env.metrics.episode_cum_rewards   # list[float]
env.metrics.episode_lengths       # list[float]

# GroupEnv — GroupMetrics delegates to each env's Metrics
env.metrics.episode_cum_rewards   # list[list[float]]
env.metrics.episode_lengths       # list[list[float]]

env.metrics.clear()
```


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).


## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
