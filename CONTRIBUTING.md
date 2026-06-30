# Contributing to Mouse Gym

Mouse Gym is actively developed and contributions are very welcome — whether that's bug reports, wrappers, or documentation improvements.

## Ways to contribute

- **Bug reports** — open a GitHub issue with a minimal reproduction and the full error traceback.
- **Feature requests** — open an issue describing the use case. If you have a design idea, sketching it out in the issue first helps align before writing code.
- **Pull requests** — see the workflow below.
- **Documentation** — edits to the README or example notebooks are welcome.

## Development setup

```bash
# Clone and create a virtual environment (Python 3.12, via uv)
git clone https://github.com/micahr234/mouse-gym.git
cd mouse-gym
source scripts/install.sh
```

This installs the package in editable mode with dev dependencies (including Jupyter for [`examples/`](examples/) notebooks).

If you edit notebooks with a different tool (browser Jupyter, `nbconvert`, scripts), clear outputs before committing, e.g. `jupyter nbconvert --clear-output --inplace examples/*.ipynb`.

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Run tests (`.venv/bin/pytest`) and check for linter errors (`pyright src/`) before opening a PR.
4. Open a pull request against `main` with a clear description of what changed and why.

Tests live under [`tests/`](tests/):

- `test_smoke.py` — core rollout protocol (CartPole, custom `env_fn`, info passthrough, tracker)

If you add a new feature, add or extend a test under [`tests/`](tests/) and/or a notebook under [`examples/`](examples/).

## Code style

- Python 3.12+, type-annotated throughout.
- Follow the existing patterns: config in `config.py`, build in `build.py`, wrappers in `wrappers.py`, formatting in `format.py`, public API in `__init__.py`. Third-party envs and Gymnasium wrappers are built by users via `env_fn` rather than bundled integrations. Implementation details belong in code comments and docstrings.
- Avoid silent fallbacks — if a precondition isn't met, raise a clear error.
- Comments should explain *why*, not *what*.

## Questions

Open a GitHub Discussion or issue.
