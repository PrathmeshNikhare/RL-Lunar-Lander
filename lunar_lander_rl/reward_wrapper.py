"""Gymnasium wrapper that adds the configured reward components.

The wrapper sees only what ``step()`` and ``reset()`` return (observation,
reward, terminated, truncated, info). It never reads simulator internals and
never changes the observation, the termination flags or the inner env.

Per step it returns ``original + sum(components)`` as the reward and adds:

- ``info["reward_components"]``: ``original``, each component and ``total``
  for this step;
- ``info["reward_episode"]`` (final step only): the same keys summed over the
  episode, plus ``length``.

In training it is applied *outside* SB3's ``Monitor`` (``make_vec_env``'s
``wrapper_class``), so ``monitor/*.csv`` keeps the original reward.
Evaluation and video recording never use it.
"""

from __future__ import annotations

from typing import Any, Final, assert_never

import gymnasium as gym
import numpy as np

from lunar_lander_rl.config import (
    OriginalReward,
    PotentialShapingReward,
    RewardConfig,
    TimePenaltyReward,
    UprightSlowBonusReward,
)
from lunar_lander_rl.environment import OBSERVATION_LABELS, Observation
from lunar_lander_rl.reward_functions import (
    potential,
    shaping_term,
    time_penalty,
    upright_slow_bonus,
)

STEP_INFO_KEY: Final[str] = "reward_components"
EPISODE_INFO_KEY: Final[str] = "reward_episode"
ORIGINAL_KEY: Final[str] = "original"
TOTAL_KEY: Final[str] = "total"
LENGTH_KEY: Final[str] = "length"
TIME_PENALTY_KEY: Final[str] = "time_penalty"
UPRIGHT_SLOW_BONUS_KEY: Final[str] = "upright_slow_bonus"
POTENTIAL_SHAPING_KEY: Final[str] = "potential_shaping"
_ANGLE: Final[int] = OBSERVATION_LABELS.index("angle")


def component_names(reward: RewardConfig) -> tuple[str, ...]:
    """Names of the custom components added in ``reward``'s mode (may be empty)."""
    if isinstance(reward, OriginalReward):
        return ()
    if isinstance(reward, TimePenaltyReward):
        return (TIME_PENALTY_KEY,)
    if isinstance(reward, UprightSlowBonusReward):
        return (UPRIGHT_SLOW_BONUS_KEY,)
    if isinstance(reward, PotentialShapingReward):
        return (POTENTIAL_SHAPING_KEY,)
    assert_never(reward)


def episode_fields(reward: RewardConfig) -> tuple[str, ...]:
    """Keys of ``info["reward_episode"]``, in a stable order."""
    return (LENGTH_KEY, ORIGINAL_KEY, *component_names(reward), TOTAL_KEY)


class RewardComponentsWrapper(
    gym.Wrapper[Observation, np.int64, Observation, np.int64]
):
    """Add the components of ``reward`` to the environment's original reward.

    Args:
        env: The environment to wrap (unchanged by the wrapper).
        reward: The validated reward configuration.
        gamma: The learner's discount; used only by potential-based shaping and
            must equal ``ppo.gamma`` (D-019).
    """

    def __init__(
        self, env: gym.Env[Observation, np.int64], reward: RewardConfig, gamma: float
    ) -> None:
        super().__init__(env)
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(f"expected a Box observation space, got {space}")
        self._reward = reward
        self._gamma = gamma
        # Phi clips positions/velocities to the declared space; the angle is
        # left unclipped because wrap_angle bounds it (clipping at +/-2*pi first
        # would make every rotation beyond a full turn read as upright).
        self._low = space.low.copy()
        self._high = space.high.copy()
        self._low[_ANGLE], self._high[_ANGLE] = -np.inf, np.inf
        self._potential = 0.0
        self._totals: dict[str, float] = {}
        self._length = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._potential = self._phi(observation)
        self._totals = dict.fromkeys(
            (ORIGINAL_KEY, *component_names(self._reward), TOTAL_KEY), 0.0
        )
        self._length = 0
        return observation, info

    def step(
        self, action: np.int64
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        original = float(reward)  # the terminal +/-100 is a Python int
        components = self._components(observation, bool(terminated))
        total = original + sum(components.values())
        step_values = {ORIGINAL_KEY: original, **components, TOTAL_KEY: total}

        self._length += 1
        for key, value in step_values.items():
            self._totals[key] += value
        info = {**info, STEP_INFO_KEY: step_values}
        if terminated or truncated:
            info[EPISODE_INFO_KEY] = {LENGTH_KEY: self._length, **self._totals}
        return observation, total, terminated, truncated, info

    def _phi(self, observation: Observation) -> float:
        reward = self._reward
        if not isinstance(reward, PotentialShapingReward):
            return 0.0
        # Clipping keeps Phi bounded (docs/reward-wrapper.md).
        return potential(
            np.clip(observation, self._low, self._high),
            reward.distance_weight,
            reward.speed_weight,
            reward.angle_weight,
        )

    def _components(
        self, observation: Observation, terminated: bool
    ) -> dict[str, float]:
        reward = self._reward
        if isinstance(reward, OriginalReward):
            return {}
        if isinstance(reward, TimePenaltyReward):
            return {TIME_PENALTY_KEY: time_penalty(reward.coefficient)}
        if isinstance(reward, UprightSlowBonusReward):
            return {
                UPRIGHT_SLOW_BONUS_KEY: upright_slow_bonus(
                    observation, reward.bonus, reward.max_abs_angle, reward.max_speed
                )
            }
        if not isinstance(reward, PotentialShapingReward):
            assert_never(reward)
        next_potential = self._phi(observation)
        value = shaping_term(
            self._potential,
            next_potential,
            gamma=self._gamma,
            alpha=reward.alpha,
            terminated=terminated,
        )
        self._potential = next_potential
        return {POTENTIAL_SHAPING_KEY: value}
