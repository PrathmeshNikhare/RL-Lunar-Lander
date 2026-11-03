"""Environment construction for Gymnasium ``LunarLander-v3``.

This module only builds the environment. Seeding happens at ``reset(seed=...)``
(Gymnasium's seeding API) and is the caller's responsibility; see
``lunar_lander_rl.rollout`` for an example. Observation and action semantics
are documented in ``docs/environment-notes.md``.
"""

from __future__ import annotations

from typing import Any, Final, Literal, cast, get_args

import gymnasium as gym
import numpy as np
import numpy.typing as npt

ENV_ID: Final[str] = "LunarLander-v3"

RenderMode = Literal["human", "rgb_array"]
VALID_RENDER_MODES: Final[tuple[str, ...]] = get_args(RenderMode)

Observation = npt.NDArray[np.float32]
LunarLanderEnv = gym.Env[Observation, np.int64]

# Order of the 8 observation components (gymnasium 1.4.0 lunar_lander.py, step()).
OBSERVATION_LABELS: Final[tuple[str, ...]] = (
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
)

# Meanings of the 4 discrete actions (LunarLander docstring, "Action Space").
ACTION_MEANINGS: Final[dict[int, str]] = {
    0: "do_nothing",
    1: "fire_left_orientation_engine",
    2: "fire_main_engine",
    3: "fire_right_orientation_engine",
}


def make_env(
    render_mode: RenderMode | None = None,
    max_episode_steps: int | None = None,
) -> LunarLanderEnv:
    """Create a discrete-action ``LunarLander-v3`` environment with default physics.

    Args:
        render_mode: ``None`` (no rendering), ``"human"`` (pygame window) or
            ``"rgb_array"`` (frames returned by ``render()``).
        max_episode_steps: Optional override of the registered ``TimeLimit``
            (1000 steps). Intended for short tests; ``None`` keeps the default.

    Returns:
        The environment wrapped by Gymnasium's standard ``make`` wrappers
        (``TimeLimit``, ``OrderEnforcing``, ``PassiveEnvChecker``).

    Raises:
        ValueError: If ``render_mode`` or ``max_episode_steps`` is invalid.
    """
    if render_mode is not None and render_mode not in VALID_RENDER_MODES:
        raise ValueError(
            f"render_mode must be None or one of {VALID_RENDER_MODES}, got {render_mode!r}"
        )
    kwargs: dict[str, Any] = {}
    if max_episode_steps is not None:
        if isinstance(max_episode_steps, bool) or not isinstance(
            max_episode_steps, int
        ):
            raise ValueError(
                f"max_episode_steps must be an int or None, got {type(max_episode_steps).__name__}"
            )
        if max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be positive, got {max_episode_steps}"
            )
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(ENV_ID, render_mode=render_mode, **kwargs)
    return cast(LunarLanderEnv, env)
