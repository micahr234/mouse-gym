# Contributing to Mouse Gym

Mouse Gym is actively developed and contributions are very welcome — whether that's bug reports, wrappers, or documentation improvements.

## Ways to contribute

- **Bug reports** — open a GitHub issue with a minimal reproduction and the full error traceback.
- **Feature requests** — open an issue describing the use case. If you have a design idea, sketching it out in the issue first helps align before writing code.
- **Pull requests** — see the workflow below.
- **Documentation** — edits to the README or example notebooks are welcome.

## Development setup

```bash
# Clone and create a free-threaded Python 3.14t virtual environment (via uv)
git clone https://github.com/micahr234/mouse-gym.git
cd mouse-gym
source scripts/install.sh
```

This installs the package in editable mode with the `dev` and `all` extras (`all` bundles `examples`, including Jupyter). Activate with `source .venv/bin/activate`. The install uses free-threaded CPython (`3.14t`) so `GroupEnv(max_threads>0)` can run constituent env steps in parallel.

If you edit notebooks with a different tool (browser Jupyter, `nbconvert`, scripts), clear outputs before committing, e.g. `jupyter nbconvert --clear-output --inplace examples/*.ipynb`.

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Before opening a PR, run:
   ```bash
   pyright
   pytest
   ```
   CI runs the same two checks (plus a wheel/sdist build) on every pull request and on pushes to `main`.
4. Open a pull request against `main` with a clear description of what changed and why.

Tests live under [`tests/`](tests/):

- `test_smoke.py` — core rollout protocol (CartPole, custom `env_fn`, info passthrough, metrics)

If you add a new feature, add or extend a test under [`tests/`](tests/) and/or a notebook under [`examples/`](examples/).

## Code style

- Python 3.14+ (free-threaded `3.14t` for threaded `GroupEnv`), type-annotated throughout.
- Follow the existing patterns: config in `config.py`, build in `build.py`, formatting in `format.py`, public API in `__init__.py`. Third-party envs and Gymnasium wrappers are built by users via `env_fn` rather than bundled integrations. Implementation details belong in code comments and docstrings.
- Avoid silent fallbacks — if a precondition isn't met, raise a clear error.
- Comments should explain *why*, not *what*.

## Releasing to PyPI

Publishing is automated by [`.github/workflows/publish.yml`](.github/workflows/publish.yml) using [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).

### Publishing a version

1. Bump `version` in `pyproject.toml` on `main`.
2. Commit, push, and create an annotated tag matching the version (e.g. `v0.1.1` for version `0.1.1`).
3. Push the tag: `git push origin v0.1.1` — the Publish workflow runs on tag push.

You can also run the workflow manually from the Actions tab (**Publish** → **Run workflow**).

## Questions

Open a GitHub Discussion or issue.
