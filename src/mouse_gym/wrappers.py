"""Single-env wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np


class SeedStreamWrapper(gym.Wrapper):
    """Control mouse-gym's internal reset stream."""

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        *,
        reset_seed: int | None,
    ):
        super().__init__(env_fn())
        self._reset_rng = np.random.default_rng(reset_seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed = int(self._reset_rng.integers(0, 2**31))
        return self.env.reset(seed=seed, options=options)
