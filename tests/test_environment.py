"""Seeded, short tests for the LunarLander-v3 factory and random rollout.

No rendering, no training, no assertions on stochastic outcomes beyond
reproducibility for a fixed seed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from lunar_lander_rl.environment import (
    ACTION_MEANINGS,
    ENV_ID,
    OBSERVATION_LABELS,
    LunarLanderEnv,
    make_env,
)
from lunar_lander_rl.rollout import run_random_episode

OBSERVATION_SIZE = 8
NUM_ACTIONS = 4
REGISTERED_TIME_LIMIT = 1000
DO_NOTHING = np.int64(0)
SEED = 0


@pytest.fixture
def env() -> Iterator[LunarLanderEnv]:
    environment = make_env()
    try:
        yield environment
    finally:
        environment.close()


def test_spec_matches_target_environment(env: LunarLanderEnv) -> None:
    assert env.spec is not None
    assert env.spec.id == ENV_ID
    assert env.spec.max_episode_steps == REGISTERED_TIME_LIMIT


def test_observation_space_has_eight_float32_components(env: LunarLanderEnv) -> None:
    space = env.observation_space
    assert isinstance(space, spaces.Box)
    assert space.shape == (OBSERVATION_SIZE,)
    assert space.dtype == np.float32
    assert len(OBSERVATION_LABELS) == OBSERVATION_SIZE


def test_action_space_is_discrete_four(env: LunarLanderEnv) -> None:
    space = env.action_space
    assert isinstance(space, spaces.Discrete)
    assert space.n == NUM_ACTIONS
    assert sorted(ACTION_MEANINGS) == list(range(NUM_ACTIONS))


def test_reset_returns_observation_in_space_and_info_dict(env: LunarLanderEnv) -> None:
    observation, info = env.reset(seed=SEED)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)


def test_reset_observation_matches_documented_order(env: LunarLanderEnv) -> None:
    observation, _ = env.reset(seed=SEED)
    labelled = dict(zip(OBSERVATION_LABELS, observation, strict=True))
    # Lander starts high above the pad with neither leg touching the ground.
    assert labelled["y"] > 0.0
    assert labelled["left_leg_contact"] == 0.0
    assert labelled["right_leg_contact"] == 0.0


def test_step_returns_five_tuple_with_separate_flags(env: LunarLanderEnv) -> None:
    env.reset(seed=SEED)
    observation, reward, terminated, truncated, info = env.step(DO_NOTHING)
    assert env.observation_space.contains(observation)
    assert isinstance(reward, (float, np.floating))
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_same_seed_and_actions_reproduce_trajectory() -> None:
    actions = [np.int64(a % NUM_ACTIONS) for a in range(20)]
    trajectories = []
    for _ in range(2):
        environment = make_env()
        try:
            observation, _ = environment.reset(seed=SEED)
            trajectory = [observation.copy()]
            rewards = []
            for action in actions:
                observation, reward, terminated, truncated, _ = environment.step(action)
                trajectory.append(observation.copy())
                rewards.append(float(reward))
                if terminated or truncated:
                    break
        finally:
            environment.close()
        trajectories.append((np.stack(trajectory), rewards))
    np.testing.assert_array_equal(trajectories[0][0], trajectories[1][0])
    assert trajectories[0][1] == trajectories[1][1]


def test_time_limit_sets_truncated_not_terminated() -> None:
    limit = 5
    environment = make_env(max_episode_steps=limit)
    try:
        assert environment.spec is not None
        assert environment.spec.max_episode_steps == limit
        environment.reset(seed=SEED)
        for step in range(1, limit + 1):
            _, _, terminated, truncated, _ = environment.step(DO_NOTHING)
            assert terminated is False, f"lander cannot reach the ground by step {step}"
            assert truncated is (step == limit)
    finally:
        environment.close()


def test_random_episode_finishes_and_env_resets(env: LunarLanderEnv) -> None:
    summary = run_random_episode(env, seed=SEED, max_steps=REGISTERED_TIME_LIMIT)
    assert 1 <= summary.steps <= REGISTERED_TIME_LIMIT
    assert summary.terminated or summary.truncated
    assert not summary.reached_step_cap
    assert np.isfinite(summary.original_return)
    assert env.observation_space.contains(summary.final_observation)
    observation, _ = env.reset(seed=SEED + 1)
    assert env.observation_space.contains(observation)


def test_random_episode_is_reproducible_for_a_seed(env: LunarLanderEnv) -> None:
    first = run_random_episode(env, seed=SEED, max_steps=REGISTERED_TIME_LIMIT)
    second = run_random_episode(env, seed=SEED, max_steps=REGISTERED_TIME_LIMIT)
    assert first.steps == second.steps
    assert first.original_return == second.original_return
    np.testing.assert_array_equal(first.final_observation, second.final_observation)


def test_step_cap_stops_loop_without_terminal_flags(env: LunarLanderEnv) -> None:
    summary = run_random_episode(env, seed=SEED, max_steps=3, record_steps=2)
    assert summary.steps == 3
    assert summary.reached_step_cap
    assert not summary.terminated and not summary.truncated
    assert [record.step for record in summary.recorded_steps] == [1, 2]


def test_rollout_reports_truncation_from_time_limit() -> None:
    limit = 3
    environment = make_env(max_episode_steps=limit)
    try:
        summary = run_random_episode(environment, seed=SEED, max_steps=10)
    finally:
        environment.close()
    assert summary.steps == limit
    assert summary.truncated
    assert not summary.terminated
    assert not summary.reached_step_cap


@pytest.mark.parametrize("render_mode", ["bogus", "HUMAN", ""])
def test_invalid_render_mode_is_rejected(render_mode: Any) -> None:
    with pytest.raises(ValueError, match="render_mode"):
        make_env(render_mode=render_mode)


@pytest.mark.parametrize("max_episode_steps", [0, -1, True, 2.5])
def test_invalid_max_episode_steps_is_rejected(max_episode_steps: Any) -> None:
    with pytest.raises(ValueError, match="max_episode_steps"):
        make_env(max_episode_steps=max_episode_steps)


@pytest.mark.parametrize(("max_steps", "record_steps"), [(0, 0), (-5, 0), (10, -1)])
def test_invalid_rollout_arguments_are_rejected(
    env: LunarLanderEnv, max_steps: int, record_steps: int
) -> None:
    with pytest.raises(ValueError, match="max_steps|record_steps"):
        run_random_episode(
            env, seed=SEED, max_steps=max_steps, record_steps=record_steps
        )
