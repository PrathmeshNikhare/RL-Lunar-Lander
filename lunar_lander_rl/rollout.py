"""Bounded random-policy rollouts for environment exploration and smoke tests.

The rollout uses the environment's original reward only; no reward wrappers
are involved. ``terminated`` (environment terminal state) and ``truncated``
(time limit) are recorded separately and never merged into a single flag
before being reported.
"""

from __future__ import annotations

from dataclasses import dataclass

from lunar_lander_rl.environment import LunarLanderEnv, Observation


@dataclass(frozen=True)
class StepRecord:
    """One transition from a rollout, kept for inspection/printing."""

    step: int
    action: int
    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class EpisodeSummary:
    """Outcome of a single random-policy episode.

    Attributes:
        seed: Seed passed to ``reset`` and the action space.
        steps: Number of ``step`` calls made.
        original_return: Undiscounted sum of the environment's own rewards.
        terminated: Episode ended in an environment terminal state.
        truncated: Episode ended because of the time limit.
        reached_step_cap: The loop stopped at ``max_steps`` without the
            environment signalling ``terminated`` or ``truncated``.
        final_observation: Observation after the last step.
        recorded_steps: The first ``record_steps`` transitions.
    """

    seed: int
    steps: int
    original_return: float
    terminated: bool
    truncated: bool
    reached_step_cap: bool
    final_observation: Observation
    recorded_steps: tuple[StepRecord, ...]


def run_random_episode(
    env: LunarLanderEnv,
    seed: int,
    max_steps: int,
    record_steps: int = 0,
) -> EpisodeSummary:
    """Run one episode with uniformly random actions, bounded by ``max_steps``.

    The environment is reset with ``seed`` and the action space is seeded with
    the same value, so the rollout is reproducible for a given seed.

    Args:
        env: Environment created by ``make_env``. It is reset, not closed.
        seed: Seed for ``env.reset`` and ``env.action_space``.
        max_steps: Hard upper bound on ``step`` calls (outer safety bound in
            addition to the environment's own ``TimeLimit``).
        record_steps: Number of initial transitions to keep in the summary.

    Raises:
        ValueError: If ``max_steps`` is not positive or ``record_steps`` is negative.
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if record_steps < 0:
        raise ValueError(f"record_steps must be non-negative, got {record_steps}")

    observation, _info = env.reset(seed=seed)
    env.action_space.seed(seed)

    original_return = 0.0
    terminated = False
    truncated = False
    steps = 0
    records: list[StepRecord] = []

    for step_index in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _info = env.step(action)
        steps = step_index + 1
        original_return += float(reward)
        if step_index < record_steps:
            records.append(
                StepRecord(
                    step=steps,
                    action=int(action),
                    observation=tuple(float(v) for v in observation),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
        if terminated or truncated:
            break

    return EpisodeSummary(
        seed=seed,
        steps=steps,
        original_return=original_return,
        terminated=bool(terminated),
        truncated=bool(truncated),
        reached_step_cap=not (terminated or truncated),
        final_observation=observation,
        recorded_steps=tuple(records),
    )
