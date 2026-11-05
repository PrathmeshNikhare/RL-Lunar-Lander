"""Outcome labels, episode diagnostics and summaries for controlled evaluation.

Every function here is pure: it derives metrics from raw evaluation records
(``episodes.csv`` fields and ``trajectories.npz`` steps). The exact rules,
and the environment semantics behind them, are in ``docs/07-metrics.md``.

Outcome (exactly one per episode, decision D-036):

========================  =====================================================
``success``               terminated, final reward +100 (at rest), |x| <= 0.2
``rest_off_pad``          terminated, final reward +100, |x| > 0.2
``crash``                 terminated, final reward -100, |x| < 1
``out_of_bounds``         terminated, final reward -100, |x| >= 1
``timeout``               truncated by the 1000-step time limit, not terminated
========================  =====================================================

Leg-contact flags are not used for outcomes: they are unreliable at
termination (docs/environment-notes.md).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from lunar_lander_rl.config import HoverConfig
from lunar_lander_rl.environment import OBSERVATION_LABELS

Outcome = Literal["success", "rest_off_pad", "crash", "out_of_bounds", "timeout"]
OUTCOMES: Final[tuple[Outcome, ...]] = (
    "success",
    "rest_off_pad",
    "crash",
    "out_of_bounds",
    "timeout",
)
# Environment facts (docs/environment-notes.md), not tunable values.
PAD_HALF_WIDTH: Final[float] = 0.2  # helipad spans x in [-0.2, 0.2]
OUT_OF_BOUNDS_X: Final[float] = 1.0  # terminated with -100 at |x| >= 1
TERMINAL_REWARD: Final[float] = 100.0  # assigned exactly on the terminal step
MAIN_ENGINE_ACTION: Final[int] = 2
SIDE_ENGINE_ACTIONS: Final[tuple[int, ...]] = (1, 3)

_Y = OBSERVATION_LABELS.index("y")
_VX = OBSERVATION_LABELS.index("vx")
_VY = OBSERVATION_LABELS.index("vy")
_LEGS = [
    OBSERVATION_LABELS.index("left_leg_contact"),
    OBSERVATION_LABELS.index("right_leg_contact"),
]


def classify_outcome(
    terminated: bool, truncated: bool, final_reward: float, final_x: float
) -> Outcome:
    """Label an episode from its terminal semantics.

    ``terminated`` takes precedence over ``truncated``: if the environment
    ends the episode on the time limit's last step, the environment outcome
    is what happened. A terminated episode whose final reward is not exactly
    ±100 contradicts the environment's semantics and raises ``ValueError``.
    """
    if terminated:
        if final_reward == TERMINAL_REWARD:
            return "success" if abs(final_x) <= PAD_HALF_WIDTH else "rest_off_pad"
        if final_reward == -TERMINAL_REWARD:
            return "out_of_bounds" if abs(final_x) >= OUT_OF_BOUNDS_X else "crash"
        raise ValueError(
            f"terminated episode with final reward {final_reward}, expected ±100"
        )
    if truncated:
        return "timeout"
    raise ValueError("episode neither terminated nor truncated")


def longest_true_run(flags: np.ndarray) -> int:
    """Length of the longest run of consecutive ``True`` values (0 if none)."""
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


@dataclass(frozen=True)
class EpisodeDiagnostics:
    first_contact_step: int | None  # first step with either leg flag set
    max_hover_steps: int  # longest run of consecutive hover steps
    hover: bool  # max_hover_steps >= min_consecutive_steps
    main_engine_steps: int
    side_engine_steps: int


def hover_steps(observations: np.ndarray, hover: HoverConfig) -> np.ndarray:
    """Per-step hover flags for post-step observations (shape ``(length, 8)``).

    A hover step has both leg flags 0, ``y >= min_height`` and speed
    ``||(vx, vy)|| <= max_speed``.
    """
    airborne = (observations[:, _LEGS] == 0).all(axis=1)
    high = observations[:, _Y] >= hover.min_height
    speed = np.hypot(observations[:, _VX], observations[:, _VY])
    result: np.ndarray = airborne & high & (speed <= hover.max_speed)
    return result


def episode_diagnostics(
    observations: np.ndarray, actions: np.ndarray, hover: HoverConfig
) -> EpisodeDiagnostics:
    """Diagnostics from one episode's steps.

    ``observations`` includes the reset observation at index 0; only the
    observations after each action (steps 1..length) are assessed, so step
    numbers count environment steps.
    """
    if len(observations) != len(actions) + 1:
        raise ValueError(f"{len(observations)} observations for {len(actions)} actions")
    after = observations[1:]
    contact = np.flatnonzero((after[:, _LEGS] > 0).any(axis=1))
    longest = longest_true_run(hover_steps(after, hover))
    return EpisodeDiagnostics(
        first_contact_step=int(contact[0]) + 1 if contact.size else None,
        max_hover_steps=longest,
        hover=longest >= hover.min_consecutive_steps,
        main_engine_steps=int(np.sum(actions == MAIN_ENGINE_ACTION)),
        side_engine_steps=int(np.isin(actions, SIDE_ENGINE_ACTIONS).sum()),
    )


def _std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def run_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise one run's per-episode metric rows (sample std, ddof=1).

    Rows need ``original_return``, ``length``, ``outcome``, ``hover``,
    ``first_contact_step`` (``None`` if no contact), ``main_engine_steps``
    and ``side_engine_steps``.
    """
    if not episodes:
        raise ValueError("cannot summarise zero episodes")
    n = len(episodes)
    returns = [float(e["original_return"]) for e in episodes]
    lengths = [float(e["length"]) for e in episodes]
    contacts = [
        float(e["first_contact_step"])
        for e in episodes
        if e["first_contact_step"] is not None
    ]
    summary: dict[str, Any] = {
        "n_episodes": n,
        "mean_return": statistics.fmean(returns),
        "std_return": _std(returns),
        "mean_length": statistics.fmean(lengths),
        "std_length": _std(lengths),
    }
    for outcome in OUTCOMES:
        count = sum(e["outcome"] == outcome for e in episodes)
        summary[f"n_{outcome}"] = count
        summary[f"{outcome}_rate"] = count / n
    n_hover = sum(bool(e["hover"]) for e in episodes)
    summary["n_hover"] = n_hover
    summary["hover_rate"] = n_hover / n
    summary["n_with_contact"] = len(contacts)
    summary["mean_first_contact_step"] = (
        statistics.fmean(contacts) if contacts else math.nan
    )
    summary["mean_main_engine_steps"] = statistics.fmean(
        float(e["main_engine_steps"]) for e in episodes
    )
    summary["mean_side_engine_steps"] = statistics.fmean(
        float(e["side_engine_steps"]) for e in episodes
    )
    return summary


def across_seeds(values: Sequence[float]) -> dict[str, float | int]:
    """Mean, sample std, min and max of one metric over a config's seeds.

    NaN values (for example no episode with leg contact) are excluded and
    the remaining count is reported as ``n_seeds``.
    """
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return {
            "n_seeds": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "n_seeds": len(finite),
        "mean": statistics.fmean(finite),
        "std": _std(finite),
        "min": min(finite),
        "max": max(finite),
    }
