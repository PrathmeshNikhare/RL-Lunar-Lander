"""Pure reward-component calculations for the custom reward modes.

Every function here depends only on its arguments: no environment, no
simulator state, no randomness. The wrapper in
``lunar_lander_rl.reward_wrapper`` calls them with values taken from
``step()``. Formulas, units and the terminal convention are documented in
``docs/reward-wrapper.md``.

Observations are the 8-value LunarLander vector (order in
``lunar_lander_rl.environment.OBSERVATION_LABELS``). Positions and velocities
are in the environment's scaled observation units, not physical units.
"""

from __future__ import annotations

import math
from typing import Final

from lunar_lander_rl.environment import OBSERVATION_LABELS, Observation

_X: Final[int] = OBSERVATION_LABELS.index("x")
_Y: Final[int] = OBSERVATION_LABELS.index("y")
_VX: Final[int] = OBSERVATION_LABELS.index("vx")
_VY: Final[int] = OBSERVATION_LABELS.index("vy")
_ANGLE: Final[int] = OBSERVATION_LABELS.index("angle")


def wrap_angle(angle: float) -> float:
    """Map an angle in radians to [-pi, pi].

    LunarLander's ``angle`` is the cumulative body angle and is not wrapped
    (docs/environment-notes.md), so a full turn would otherwise not read as
    upright.
    """
    return math.remainder(angle, math.tau)


def speed(observation: Observation) -> float:
    """Norm of (vx, vy) in scaled observation units."""
    return math.hypot(float(observation[_VX]), float(observation[_VY]))


def time_penalty(coefficient: float) -> float:
    """Per-step time cost: ``-coefficient`` on every step."""
    return -coefficient


def is_upright_and_slow(
    observation: Observation, max_abs_angle: float, max_speed: float
) -> bool:
    """True if ``|wrap(angle)| <= max_abs_angle`` and ``speed <= max_speed``.

    Deliberately ignores position and leg contact: the state can be met while
    hovering anywhere (the flaw that Phase 04 probes).
    """
    return (
        abs(wrap_angle(float(observation[_ANGLE]))) <= max_abs_angle
        and speed(observation) <= max_speed
    )


def upright_slow_bonus(
    observation: Observation, bonus: float, max_abs_angle: float, max_speed: float
) -> float:
    """``bonus`` if the observation is upright and slow, else 0."""
    return bonus if is_upright_and_slow(observation, max_abs_angle, max_speed) else 0.0


def potential(
    observation: Observation,
    distance_weight: float,
    speed_weight: float,
    angle_weight: float,
) -> float:
    """``Phi(s) = -(w_d * ||(x, y)|| + w_v * ||(vx, vy)|| + w_a * |wrap(angle)|)``.

    Phi <= 0, and it is 0 only when the lander is at the pad centre, at rest
    and upright. It is bounded when the observation is bounded; the wrapper
    clips observations to the observation space before calling this.
    """
    distance = math.hypot(float(observation[_X]), float(observation[_Y]))
    return -(
        distance_weight * distance
        + speed_weight * speed(observation)
        + angle_weight * abs(wrap_angle(float(observation[_ANGLE])))
    )


def shaping_term(
    potential_before: float,
    potential_after: float,
    *,
    gamma: float,
    alpha: float,
    terminated: bool,
) -> float:
    """``F = alpha * (gamma * Phi(s') - Phi(s))``, with ``Phi(s') = 0`` if terminated.

    Only ``terminated`` (an absorbing state) zeroes the next potential. A
    time-limit truncation is not absorbing: Stable-Baselines3 bootstraps the
    value of the final observation, so the real ``Phi(s')`` is used.
    """
    next_potential = 0.0 if terminated else potential_after
    return alpha * (gamma * next_potential - potential_before)
