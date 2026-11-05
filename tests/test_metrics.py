"""Hand-calculated tests for outcome labels, episode diagnostics and summaries."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_lander_rl.config import HoverConfig
from lunar_lander_rl.metrics import (
    OUTCOMES,
    across_seeds,
    classify_outcome,
    episode_diagnostics,
    hover_steps,
    longest_true_run,
    run_summary,
)

HOVER = HoverConfig(min_height=0.1, max_speed=0.2, min_consecutive_steps=3)


def _obs(
    y: float = 0.5,
    vx: float = 0.0,
    vy: float = 0.0,
    left: float = 0.0,
    right: float = 0.0,
    x: float = 0.0,
) -> list[float]:
    return [x, y, vx, vy, 0.0, 0.0, left, right]


@pytest.mark.parametrize(
    ("terminated", "truncated", "reward", "x", "expected"),
    [
        (True, False, 100.0, 0.0, "success"),
        (True, False, 100.0, 0.2, "success"),  # pad edge is inclusive
        (True, False, 100.0, -0.2, "success"),
        (True, False, 100.0, 0.2001, "rest_off_pad"),
        (True, False, 100.0, -0.9, "rest_off_pad"),
        (True, False, -100.0, 0.1, "crash"),
        (True, False, -100.0, -0.999, "crash"),
        (True, False, -100.0, 1.0, "out_of_bounds"),  # |x| >= 1 boundary
        (True, False, -100.0, -1.3, "out_of_bounds"),
        (False, True, 0.4, 0.0, "timeout"),
        (False, True, 100.0, 0.0, "timeout"),  # not terminated: reward ignored
        (True, True, 100.0, 0.0, "success"),  # environment outcome wins
        (True, True, -100.0, 0.0, "crash"),
    ],
)
def test_classify_outcome(
    terminated: bool, truncated: bool, reward: float, x: float, expected: str
) -> None:
    assert classify_outcome(terminated, truncated, reward, x) == expected


@pytest.mark.parametrize("reward", [0.0, 99.9, -100.5, 2.3])
def test_terminated_without_terminal_reward_is_rejected(reward: float) -> None:
    with pytest.raises(ValueError, match="expected ±100"):
        classify_outcome(True, False, reward, 0.0)


def test_unfinished_episode_is_rejected() -> None:
    with pytest.raises(ValueError, match="neither"):
        classify_outcome(False, False, 100.0, 0.0)


def test_outcomes_are_the_five_documented_labels() -> None:
    assert OUTCOMES == ("success", "rest_off_pad", "crash", "out_of_bounds", "timeout")


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], 0),
        ([False, False], 0),
        ([True], 1),
        ([True, True, False, True], 2),
        ([False, True, True, True, False, True, True], 3),
        ([True, True, True], 3),
    ],
)
def test_longest_true_run(flags: list[bool], expected: int) -> None:
    assert longest_true_run(np.asarray(flags, dtype=bool)) == expected


def test_hover_step_conditions_and_boundaries() -> None:
    obs = np.asarray(
        [
            _obs(),  # hover
            _obs(y=0.1),  # min_height inclusive -> hover
            _obs(y=0.0999),  # too low
            _obs(vx=0.0, vy=0.2),  # speed exactly 0.2 -> hover (max_speed inclusive)
            _obs(vx=0.0, vy=-0.21),  # too fast
            _obs(left=1.0),  # one leg down
            _obs(right=1.0),
        ],
        dtype=np.float32,
    )
    assert hover_steps(obs, HOVER).tolist() == [
        True,
        True,
        False,
        True,
        False,
        False,
        False,
    ]


def test_episode_diagnostics_hand_calculated() -> None:
    reset = _obs(y=1.4, vy=-0.5)  # index 0 is never assessed
    after = [
        _obs(),  # step 1: hover
        _obs(),  # step 2: hover
        _obs(vy=-0.5),  # step 3: falling fast
        _obs(),  # step 4..6: hover x3
        _obs(),
        _obs(),
        _obs(y=0.02, left=1.0),  # step 7: first contact
        _obs(y=0.0, left=1.0, right=1.0),
    ]
    observations = np.asarray([reset, *after], dtype=np.float32)
    actions = np.asarray([2, 2, 0, 1, 3, 2, 0, 0], dtype=np.int8)
    result = episode_diagnostics(observations, actions, HOVER)
    assert result.first_contact_step == 7
    assert result.max_hover_steps == 3
    assert result.hover  # 3 >= min_consecutive_steps
    assert result.main_engine_steps == 3
    assert result.side_engine_steps == 2


def test_episode_diagnostics_no_contact_and_short_hover() -> None:
    observations = np.asarray([_obs(), _obs(), _obs(), _obs(vy=-1.0)], dtype=np.float32)
    actions = np.asarray([0, 0, 0], dtype=np.int8)
    result = episode_diagnostics(observations, actions, HOVER)
    assert result.first_contact_step is None
    assert result.max_hover_steps == 2
    assert not result.hover
    assert result.main_engine_steps == result.side_engine_steps == 0


def test_episode_diagnostics_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="observations"):
        episode_diagnostics(
            np.zeros((3, 8), dtype=np.float32), np.zeros(3, dtype=np.int8), HOVER
        )


def _row(outcome: str, ret: float, length: int, **extra: object) -> dict[str, object]:
    return {
        "original_return": ret,
        "length": length,
        "outcome": outcome,
        "hover": extra.get("hover", False),
        "first_contact_step": extra.get("contact"),
        "main_engine_steps": extra.get("main", 0),
        "side_engine_steps": extra.get("side", 0),
    }


def test_run_summary_hand_calculated() -> None:
    rows = [
        _row("success", 250.0, 300, contact=200, main=100, side=10),
        _row("success", 230.0, 340, contact=240, main=120, side=20),
        _row("timeout", -50.0, 1000, hover=True, main=500, side=30),
        _row("crash", -10.0, 160, contact=150, main=20, side=0),
    ]
    s = run_summary(rows)
    assert s["n_episodes"] == 4
    assert s["mean_return"] == pytest.approx(105.0)
    assert s["std_return"] == pytest.approx(math.sqrt(73900.0 / 3))
    assert s["mean_length"] == pytest.approx(450.0)
    assert (s["n_success"], s["n_timeout"], s["n_crash"]) == (2, 1, 1)
    assert s["n_rest_off_pad"] == s["n_out_of_bounds"] == 0
    assert s["success_rate"] == 0.5 and s["timeout_rate"] == 0.25
    assert sum(s[f"n_{o}"] for o in OUTCOMES) == 4
    assert s["n_hover"] == 1 and s["hover_rate"] == 0.25
    assert s["n_with_contact"] == 3
    assert s["mean_first_contact_step"] == pytest.approx(590 / 3)
    assert s["mean_main_engine_steps"] == pytest.approx(185.0)
    assert s["mean_side_engine_steps"] == pytest.approx(15.0)


def test_run_summary_single_episode_and_no_contact() -> None:
    s = run_summary([_row("timeout", 1.0, 1000, hover=True)])
    assert s["std_return"] == 0.0 and s["std_length"] == 0.0
    assert s["n_with_contact"] == 0 and math.isnan(s["mean_first_contact_step"])
    with pytest.raises(ValueError):
        run_summary([])


def test_across_seeds() -> None:
    result = across_seeds([1.0, 2.0, 6.0])
    assert result["n_seeds"] == 3
    assert result["mean"] == pytest.approx(3.0)
    assert result["std"] == pytest.approx(math.sqrt(7.0))
    assert (result["min"], result["max"]) == (1.0, 6.0)
    assert across_seeds([5.0])["std"] == 0.0
    partial = across_seeds([math.nan, 4.0])
    assert partial["n_seeds"] == 1 and partial["mean"] == 4.0
    empty = across_seeds([math.nan])
    assert empty["n_seeds"] == 0 and math.isnan(empty["mean"])


def test_hover_speed_is_the_euclidean_norm() -> None:
    # Each component is below 0.2, but the norm is 0.212: not a hover step.
    obs = np.asarray([_obs(vx=0.15, vy=0.15), _obs(vx=0.1, vy=-0.1)], dtype=np.float32)
    assert hover_steps(obs, HOVER).tolist() == [False, True]


def test_limits_apply_at_float32_precision() -> None:
    """Observations are float32; the documented limits behave as follows.

    Hover: a float32 array compared with a Python float uses float32 (NumPy 2,
    NEP 50), so a speed of float32(0.2) is within ``max_speed`` 0.2.
    Outcome: ``final_x`` is a float32 widened to float64; float32(0.2) is
    0.2000000030 > 0.2, so it is off the pad.
    """
    edge = np.float32(0.2)
    assert hover_steps(np.asarray([_obs(vy=float(edge))], dtype=np.float32), HOVER)[0]
    assert classify_outcome(True, False, 100.0, float(edge)) == "rest_off_pad"
    assert (
        classify_outcome(True, False, 100.0, float(np.float32(0.19999999))) == "success"
    )
