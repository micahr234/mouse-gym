# Mouse Gym 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-gym/main/mouse-gym.png" width="400"/></p>

Traditional episodic reinforcement learning typically measures how well a policy performs within an episode. What it does not directly measure is how quickly an agent can **improve after experiencing multiple episodes**.

**mouse-gym** recasts Gymnasium environments around this idea. Instead of treating each episode as an isolated trial, it groups multiple episodes into a **task**. Episodes within a task share the same underlying problem, so the agent can remember previous attempts, adapt, and improve while it works.

The question becomes not just **How well can an agent perform a task?** but **How quickly can it improve as it gains experience with that task?** Adaptation happens within a task, across episodes — the same thing people do when they learn on the job.

mouse-gym preserves episode boundaries while presenting experience as a continuous stream, so recurrent models, transformers, and other adaptive agents can carry information forward across episodes.


## News

- **2026-08-18 — 1.0.0** First stable release. Public step contract is `episode_done` / `task_done` on a reset-free `step()` stream.
- **2026-07-07 — NumPy I/O** Step inputs and outputs use NumPy arrays (Gymnasium-native types). PyTorch is no longer required.
- **2026-07-02 — `tracker` → `metrics`** Renamed `env.tracker` / `Tracker` / `GroupTracker` to `env.metrics` / `Metrics` / `GroupMetrics`. Example notebook renamed to [04 — Metrics](examples/04_metrics.ipynb).
- **2026-06-29 — Repository split** Mouse Gym is now its own package. Reset-free rollout infrastructure lives here; custom environment implementations live elsewhere.
- **2026-06-26 — `SingleEnv` / `GroupEnv`** `make_env` returns one env; `make_group_env` handles parallel rollouts.

See [CHANGELOG.md](CHANGELOG.md) for the full release history.


## Install

**Requirements:** Python 3.13+, Gymnasium ≥ 1.3, NumPy ≥ 1.26 (NumPy is also pulled in by Gymnasium).

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
    seed=0,
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

- **Reset-free rollouts.** Only call `step()` — `SingleEnv.reset()` is not part of the API, including at task boundaries. When an episode or task ends, the next `step()` returns a **reset frame**: initial observation, `step_index=0`, `episode_done=0`, `task_done=0`, `reset_reward`; the input is ignored. (Mouse-gym calls the underlying env's Gymnasium `reset()` internally on that step — you never call it yourself.) One continuous loop; boundaries are fields in the output, not a separate `reset()` before `step()`.

- **Dict I/O instead of Gymnasium's action + tuple.** Pass `{"action": ...}` to `step()` (use `sample_random_input()` for random rollouts). Each `step()` returns one dict with named fields — `task_index`, `episode_index`, `step_index`, `reward`, `task_done`, `episode_done`, `observation`, and `info` — rather than `(observation, reward, terminated, truncated, info)`.

- **Tasks group episodes.** A task is a consecutive run of episodes — length set by `episodes_per_task` in `EnvConfig` (default `0`: no task boundary). When a task ends, the next `step()` is a reset frame with `task_index` incremented and `episode_index` reset to `0`. `task_done=2` marks the last step of a task (episode budget exhausted). `task_done=1` is reserved and unused.

- **Each task is a fresh, seeded Gymnasium session.** `EnvConfig.seed` seeds a stream that advances once per task: the drawn value is passed to the underlying `env.reset(seed=...)` at the task-start reset only, and episode resets within the task pass no seed — Gymnasium's own seed-once-per-session convention, applied per task. A whole task is reproducible from its seed, while episode-level randomness (e.g. the start position) still varies across episodes as the env's RNG continues. Envs that generate their problem instance when `reset` receives an explicit seed (e.g. a procedural map) present the same instance to every episode in the task — no mouse-gym-specific protocol required.

- **`episode_done` and `task_done` replace `terminated` / `truncated`.** Two independent integer fields per step, both using `0`/`1`/`2`:
  - **`episode_done`** (Gymnasium episode outcome only):
    - `0` — **Running.** A normal mid-episode step, or a reset frame (`step_index=0`, `reward=reset_reward`).
    - `1` — **Episode terminated.** Underlying env returned `terminated=True`. Do not bootstrap; the next `step()` is a reset frame.
    - `2` — **Episode truncated.** Underlying env returned `truncated=True`. Same episode-end semantics as `1`.
  - **`task_done`** (task-boundary outcome; not copied from Gymnasium):
    - `0` — **Not a task boundary.**
    - `1` — **Task terminated.** Reserved; never emitted today (no task-success/failure signal yet).
    - `2` — **Task truncated.** This episode completed and it was the last episode in the task (`episodes_per_task` reached). Bootstrap here; the next `step()` is a reset frame that starts a new task (`task_index` increments, `episode_index` resets to `0`).
  - On the last episode of a task both fields fire together — e.g. `episode_done=1` and `task_done=2`. With `episodes_per_task=0` (default), `task_done` stays `0`. Gymnasium has no task grouping or equivalent codes.

## Additions to Gymnasium

On top of the standard env API, mouse-gym adds:

- **Metrics on the env.** Episode returns and lengths accumulate in `env.metrics`, not in the `step()` return value. When `episodes_per_task > 0`, completed tasks also record reward and length sums in `env.metrics.task_cum_rewards` and `env.metrics.task_lengths`. See [04 — Metrics](examples/04_metrics.ipynb).

- **Grouped envs.** `GroupEnv` steps multiple `SingleEnv` instances in one `step()` call and returns a flat `list[dict]` — useful for mixed or multi-task setups without a vectorized wrapper. By default (`max_threads=0`) stepping is sequential on the calling thread; set `max_threads > 0` to distribute envs across that many worker threads. See [02 — Multiple envs](examples/02_multi_env.ipynb).

- **Input/output specs.** `input_spec` and `output_spec` describe the construction-time contract for step dict shapes and dtypes (on `GroupEnv`, `input_specs[i]` and `output_specs[i]`). See [01 — Random rollout](examples/01_random_rollout.ipynb).

## Examples

The notebooks in [`examples/`](examples/) are the detailed reference for `EnvConfig`, input/output fields, and day-to-day usage. Install notebook dependencies with `pip install "mouse-gym[examples]"`, then work through them in order:

**[01 — Random rollout](examples/01_random_rollout.ipynb)** — Start here. Build an env from `EnvConfig`, run the reset-free `step()` loop, and inspect what comes back on each call: input dict (`action`), output dict (`task_index`, `episode_index`, `step_index`, `reward`, `task_done`, `episode_done`, `observation`, `info`), reset frames, and done codes. Also covers `input_spec` / `output_spec` and optional `env_fn` factories.

**[02 — Multiple envs](examples/02_multi_env.ipynb)** — Combine several env instances with `make_group_env`. Step heterogeneous envs (different ids, spaces, and seeds) in one loop; optionally parallelize with `max_threads`; read flat `list[dict]` inputs and outputs; use `env.names`, `input_specs[i]`, and `output_specs[i]`.

**[03 — RNG seeding control](examples/03_rng_seeding_control.ipynb)** — Reproduce or vary behavior with `seed` (one seed drawn per task; the env is reseeded only at task starts) and `env.action_space.seed()` (random action sampling), independently.

**[04 — Metrics](examples/04_metrics.ipynb)** — Read episode returns and lengths from `env.metrics`, clear between eval runs, and aggregate stats across a `GroupEnv`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
