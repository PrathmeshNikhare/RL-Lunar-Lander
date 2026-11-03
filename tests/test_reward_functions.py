"""Hand-calculated tests for the pure reward-component functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_lander_rl.environment import Observation
from lunar_lander_rl.reward_functions import (
    is_upright_and_slow,
    potential,
    shaping_term,
    speed,
    time_penalty,
    upright_slow_bonus,
    wrap_angle,
)


def obs(
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    angle: float = 0.0,
    angular_velocity: float = 0.0,
    left: float = 0.0,
    right: float = 0.0,
) -> Observation:
    return np.array([x, y, vx, vy, angle, angular_velocity, left, right], np.float32)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.1, 0.1),
        (math.tau + 0.1, 0.1),
        (-math.tau - 0.1, -0.1),
        (1.5 * math.pi, -0.5 * math.pi),
        (3 * math.tau, 0.0),
    ],
)
def test_wrap_angle(angle: float, expected: float) -> None:
    assert wrap_angle(angle) == pytest.approx(expected, abs=1e-12)


def test_wrap_angle_stays_within_pi() -> None:
    for angle in np.linspace(-20.0, 20.0, 401):
        assert abs(wrap_angle(float(angle))) <= math.pi


def test_speed_is_velocity_norm() -> None:
    assert speed(obs(vx=0.75, vy=-1.0)) == 1.25  # 3-4-5 triangle, exact in float32


def test_time_penalty_is_negative_coefficient() -> None:
    assert time_penalty(0.01) == -0.01
    assert time_penalty(0.25) == -0.25


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (obs(), True),
        (obs(angle=0.25, vy=-0.5), True),  # both exactly at the thresholds
        (obs(angle=0.2501, vy=-0.5), False),
        (obs(angle=-0.2501), False),
        (obs(vx=0.3, vy=-0.4001), False),  # speed just above 0.5
        (obs(angle=math.tau + 0.1), True),  # one full turn: upright again
        (obs(angle=math.pi), False),  # upside down
        # Position and legs are ignored on purpose (the flaw Phase 04 probes).
        (obs(x=0.9, y=1.4, left=0.0, right=0.0), True),
    ],
)
def test_upright_and_slow(observation: Observation, expected: bool) -> None:
    assert is_upright_and_slow(observation, max_abs_angle=0.25, max_speed=0.5) is (
        expected
    )


def test_upright_slow_bonus_values() -> None:
    assert upright_slow_bonus(obs(), 2.0, 0.25, 0.5) == 2.0
    assert upright_slow_bonus(obs(vy=-1.0), 2.0, 0.25, 0.5) == 0.0


def test_potential_hand_calculated() -> None:
    # ||(0.75, 1.0)|| = 1.25, ||(0.75, -1.0)|| = 1.25, |angle| = 0.5
    observation = obs(x=0.75, y=1.0, vx=0.75, vy=-1.0, angle=-0.5, left=1.0)
    assert potential(observation, 2.0, 1.0, 4.0) == -(2.0 * 1.25 + 1.25 + 4.0 * 0.5)
    assert potential(observation, 0.0, 0.0, 1.0) == -0.5
    assert potential(obs(), 2.0, 1.0, 4.0) == 0.0


def test_potential_uses_wrapped_angle() -> None:
    assert potential(obs(angle=math.tau + 0.5), 0.0, 0.0, 1.0) == pytest.approx(
        -0.5, abs=1e-6
    )


def test_shaping_term_hand_calculated() -> None:
    # alpha * (gamma * Phi(s') - Phi(s)) = 2 * (0.5 * -1.5 - -1.0) = 0.5
    assert shaping_term(-1.0, -1.5, gamma=0.5, alpha=2.0, terminated=False) == 0.5
    # Terminated: Phi(s') = 0, so F = 2 * (0 - -1.0) = 2.0
    assert shaping_term(-1.0, -1.5, gamma=0.5, alpha=2.0, terminated=True) == 2.0


def test_shaping_term_with_project_gamma() -> None:
    value = shaping_term(-2.0, -1.0, gamma=0.999, alpha=1.0, terminated=False)
    assert value == pytest.approx(0.999 * -1.0 + 2.0)


@pytest.mark.parametrize("terminated", [True, False])
def test_discounted_shaping_telescopes(terminated: bool) -> None:
    """sum_t gamma^t F_t = alpha * (gamma^T * Phi(s_T) - Phi(s_0)), Phi(s_T)=0 if terminated."""
    gamma, alpha = 0.9, 1.5
    potentials = [-3.0, -2.5, -4.0, -1.0, -0.25]
    steps = len(potentials) - 1
    discounted = 0.0
    for t in range(steps):
        last = t == steps - 1
        discounted += gamma**t * shaping_term(
            potentials[t],
            potentials[t + 1],
            gamma=gamma,
            alpha=alpha,
            terminated=terminated and last,
        )
    final = 0.0 if terminated else potentials[-1]
    assert discounted == pytest.approx(alpha * (gamma**steps * final - potentials[0]))
