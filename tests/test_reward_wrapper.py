"""Reward wrapper tests.

Contract tests use a scripted stub environment (no Box2D), so the wrapper can
only rely on what ``reset``/``step`` return. Integration tests use the real
``LunarLander-v3``.
"""

from __future__ import annotations

import csv
import inspect
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from conftest import TEST_REWARDS

from lunar_lander_rl import reward_functions, reward_logging, reward_wrapper
from lunar_lander_rl.config import (
    OriginalReward,
    PotentialShapingReward,
    RewardConfig,
    TimePenaltyReward,
    UprightSlowBonusReward,
)
from lunar_lander_rl.environment import Observation, make_env
from lunar_lander_rl.reward_wrapper import (
    EPISODE_INFO_KEY,
    STEP_INFO_KEY,
    RewardComponentsWrapper,
    component_names,
)

ACTION = np.int64(0)
GAMMA = 0.5  # exact in binary; S0-S2 below are exact in float32 too


def obs(
    x: float = 0.0, y: float = 0.0, vx: float = 0.0, vy: float = 0.0, angle: float = 0.0
) -> Observation:
    return np.array([x, y, vx, vy, angle, 0.0, 0.0, 0.0], np.float32)


Transition = tuple[Observation, Any, bool, bool, dict[str, Any]]


class ScriptedEnv(gym.Env[Observation, np.int64]):
    """Returns a fixed reset observation and a fixed list of transitions."""

    def __init__(self, reset_obs: Observation, transitions: list[Transition]) -> None:
        bound = np.array([2.5, 2.5, 10, 10, 2 * math.pi, 10, 1, 1], np.float32)
        self.observation_space = gym.spaces.Box(-bound, bound, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._reset_obs = reset_obs
        self._transitions = transitions
        self._index = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._index = 0
        return self._reset_obs, {"reset_key": True}

    def step(self, action: np.int64) -> Transition:
        transition = self._transitions[self._index]
        self._index += 1
        return transition


# s0: distance 0.5 -> Phi = -(2 * 0.5) = -1.0 (weights 2, 1, 1)
S0 = obs(x=0.5)
# s1: upright and slow; speed 0.5, angle 0.25 -> Phi = -(0.5 + 0.25) = -0.75
S1 = obs(vy=-0.5, angle=0.25)
# s2: height 0.5, fast -> Phi = -(2 * 0.5 + 1.25) = -2.25
S2 = obs(y=0.5, vx=0.75, vy=-1.0)

POTENTIAL = PotentialShapingReward(
    alpha=2.0, distance_weight=2.0, speed_weight=1.0, angle_weight=1.0
)
BONUS = UprightSlowBonusReward(bonus=3.0, max_abs_angle=0.25, max_speed=0.5)


def scripted(*, terminated: bool = True) -> ScriptedEnv:
    return ScriptedEnv(
        S0,
        [
            (S1, np.float64(1.5), False, False, {"inner": 1}),
            # The real env returns the terminal +/-100 as a Python int.
            (
                S2,
                100 if terminated else np.float64(-0.5),
                terminated,
                not terminated,
                {},
            ),
        ],
    )


def run(reward: RewardConfig, env: ScriptedEnv) -> list[Transition]:
    wrapped = RewardComponentsWrapper(env, reward, GAMMA)
    wrapped.reset(seed=0)
    return [wrapped.step(ACTION) for _ in range(2)]


def test_original_mode_returns_the_original_reward_exactly() -> None:
    raw = scripted()
    raw.reset()
    expected = [raw.step(ACTION) for _ in range(2)]
    for got, want in zip(run(OriginalReward(), scripted()), expected, strict=True):
        assert got[0] is want[0]  # observation passed through untouched
        assert got[1] == float(want[1]) and type(got[1]) is float
        assert got[2:4] == want[2:4]
    steps = run(OriginalReward(), scripted())
    assert steps[0][4][STEP_INFO_KEY] == {"original": 1.5, "total": 1.5}
    assert steps[1][4][EPISODE_INFO_KEY] == {
        "length": 2,
        "original": 101.5,
        "total": 101.5,
    }


def test_time_penalty_hand_calculated() -> None:
    steps = run(TimePenaltyReward(coefficient=0.25), scripted())
    assert [s[1] for s in steps] == [1.25, 99.75]
    assert steps[0][4][STEP_INFO_KEY] == {
        "original": 1.5,
        "time_penalty": -0.25,
        "total": 1.25,
    }
    assert steps[1][4][EPISODE_INFO_KEY] == {
        "length": 2,
        "original": 101.5,
        "time_penalty": -0.5,
        "total": 101.0,
    }


def test_upright_slow_bonus_uses_next_observation() -> None:
    steps = run(BONUS, scripted())
    assert [s[4][STEP_INFO_KEY]["upright_slow_bonus"] for s in steps] == [3.0, 0.0]
    assert [s[1] for s in steps] == [4.5, 100.0]
    assert steps[1][4][EPISODE_INFO_KEY]["upright_slow_bonus"] == 3.0


def test_potential_shaping_terminated_hand_calculated() -> None:
    steps = run(POTENTIAL, scripted(terminated=True))
    shaping = [s[4][STEP_INFO_KEY]["potential_shaping"] for s in steps]
    # F1 = 2 * (0.5 * -0.75 - -1.0) = 1.25; F2 (terminated, Phi=0) = 2 * 0.75 = 1.5
    assert shaping == pytest.approx([1.25, 1.5])
    assert [s[1] for s in steps] == pytest.approx([2.75, 101.5])
    assert steps[1][4][EPISODE_INFO_KEY]["potential_shaping"] == pytest.approx(2.75)


def test_potential_shaping_truncated_uses_real_next_potential() -> None:
    steps = run(POTENTIAL, scripted(terminated=False))
    # F2 (truncated) = 2 * (0.5 * -2.25 - -0.75) = -0.75; total = -0.5 - 0.75
    assert steps[1][4][STEP_INFO_KEY]["potential_shaping"] == -0.75
    assert steps[1][1] == -1.25
    assert steps[1][3] is True and steps[1][2] is False
    episode = steps[1][4][EPISODE_INFO_KEY]
    assert episode["potential_shaping"] == 1.25 - 0.75
    assert episode["total"] == (1.5 + 1.25) + (-0.5 - 0.75)


def test_terminated_and_truncated_on_the_same_step() -> None:
    """Terminated wins: Phi(s') = 0, and the episode summary is emitted once."""
    env = ScriptedEnv(S0, [(S2, 100, True, True, {})])
    wrapped = RewardComponentsWrapper(env, POTENTIAL, GAMMA)
    wrapped.reset()
    _, total, terminated, truncated, info = wrapped.step(ACTION)
    # One-step episode: F = 2 * (0 - -1.0) = 2.0
    assert info[STEP_INFO_KEY]["potential_shaping"] == 2.0
    assert (total, terminated, truncated) == (102.0, True, True)
    assert info[EPISODE_INFO_KEY] == {
        "length": 1,
        "original": 100.0,
        "potential_shaping": 2.0,
        "total": 102.0,
    }


def test_potential_is_reset_with_each_episode() -> None:
    """After reset, F1 uses Phi(s0) again, not the last potential of episode 1."""
    wrapped = RewardComponentsWrapper(scripted(terminated=False), POTENTIAL, GAMMA)
    for _ in range(2):
        wrapped.reset()
        _, _, _, _, info = wrapped.step(ACTION)
        assert info[STEP_INFO_KEY]["potential_shaping"] == 1.25
        wrapped.step(ACTION)


def test_angle_beyond_a_full_turn_is_wrapped_not_clipped() -> None:
    """Regression: clipping the angle to +/-2*pi before wrapping read 7.5 rad as 0."""
    angle_only = PotentialShapingReward(
        alpha=1.0, distance_weight=0.0, speed_weight=0.0, angle_weight=1.0
    )
    env = ScriptedEnv(obs(), [(obs(angle=7.5), 0.0, False, False, {})])
    wrapped = RewardComponentsWrapper(env, angle_only, 1.0)
    wrapped.reset()
    _, _, _, _, info = wrapped.step(ACTION)
    # gamma = 1: F = Phi(s') - Phi(s0) = -|wrap(7.5)| - 0, wrap(7.5) = 7.5 - 2*pi
    assert info[STEP_INFO_KEY]["potential_shaping"] == pytest.approx(
        -(7.5 - math.tau), abs=1e-6
    )


def test_potential_is_computed_on_observation_clipped_to_the_space() -> None:
    far = obs(x=10.0)  # beyond the 2.5 bound
    env = ScriptedEnv(far, [(far, 0.0, False, True, {})])
    wrapped = RewardComponentsWrapper(
        env,
        PotentialShapingReward(
            alpha=1.0, distance_weight=1.0, speed_weight=0.0, angle_weight=0.0
        ),
        GAMMA,
    )
    wrapped.reset()
    _, _, _, _, info = wrapped.step(ACTION)
    # Phi = -2.5 both times: F = 0.5 * -2.5 + 2.5 = 1.25 (not 5.0 from x = 10)
    assert info[STEP_INFO_KEY]["potential_shaping"] == pytest.approx(1.25)


@pytest.mark.parametrize(
    "reward",
    [OriginalReward(), TimePenaltyReward(coefficient=0.1), BONUS, POTENTIAL],
)
def test_components_sum_to_total_and_match_declared_names(
    reward: RewardConfig,
) -> None:
    for _, total, _, _, info in run(reward, scripted()):
        values = info[STEP_INFO_KEY]
        assert tuple(values) == ("original", *component_names(reward), "total")
        assert total == values["total"]
        assert values["total"] == pytest.approx(
            sum(v for k, v in values.items() if k != "total")
        )


def test_info_is_copied_and_inner_keys_are_preserved() -> None:
    env = scripted()
    inner_info = env._transitions[0][4]
    wrapped = RewardComponentsWrapper(env, POTENTIAL, GAMMA)
    _, reset_info = wrapped.reset()
    assert reset_info == {"reset_key": True}
    _, _, _, _, info = wrapped.step(ACTION)
    assert info["inner"] == 1
    assert STEP_INFO_KEY not in inner_info  # the inner env's dict is not mutated
    assert EPISODE_INFO_KEY not in info  # only on the final step


def test_reset_starts_a_new_episode() -> None:
    env = scripted()
    wrapped = RewardComponentsWrapper(env, TimePenaltyReward(coefficient=0.25), GAMMA)
    for _ in range(2):
        wrapped.reset()
        wrapped.step(ACTION)
        *_, info = wrapped.step(ACTION)
        assert info[EPISODE_INFO_KEY]["length"] == 2
        assert info[EPISODE_INFO_KEY]["time_penalty"] == -0.5


def test_invalid_wrapper_arguments() -> None:
    with pytest.raises(ValueError, match="gamma"):
        RewardComponentsWrapper(scripted(), POTENTIAL, 0.0)
    env = scripted()
    env.observation_space = gym.spaces.Discrete(3)  # type: ignore[assignment]
    with pytest.raises(TypeError, match="Box"):
        RewardComponentsWrapper(env, POTENTIAL, GAMMA)


def test_reward_code_does_not_reach_into_the_simulator() -> None:
    """Acceptance criterion: no hidden Box2D internals are accessed.

    The main evidence is that every contract test above runs on
    ``ScriptedEnv``, which has no simulator at all. This string search is only
    a cheap guard against obvious regressions (it cannot catch ``getattr``).
    """
    for module in (reward_wrapper, reward_functions, reward_logging):
        source = inspect.getsource(module)
        for forbidden in (".unwrapped", "Box2D", ".lander", ".world", "game_over"):
            assert forbidden not in source, (module.__name__, forbidden)


# --------------------------------------------------------------------------- #
# Integration with the real environment
# --------------------------------------------------------------------------- #


def test_original_mode_matches_raw_environment_for_a_full_episode() -> None:
    raw, wrapped = (
        make_env(),
        RewardComponentsWrapper(make_env(), OriginalReward(), 0.999),
    )
    try:
        raw_obs, _ = raw.reset(seed=3)
        wrapped_obs, _ = wrapped.reset(seed=3)
        actions = np.random.default_rng(3).integers(0, 4, size=1000)
        for action in actions:
            np.testing.assert_array_equal(raw_obs, wrapped_obs)
            raw_obs, r, term, trunc, _ = raw.step(np.int64(action))
            wrapped_obs, w, w_term, w_trunc, info = wrapped.step(np.int64(action))
            assert w == float(r) and (w_term, w_trunc) == (term, trunc)
            if term or trunc:
                break
        assert term or trunc
        assert info[EPISODE_INFO_KEY]["total"] == info[EPISODE_INFO_KEY]["original"]
    finally:
        raw.close()
        wrapped.close()


def test_monitor_keeps_original_reward_inside_training_vec_env(
    smoke_config: Any, tmp_path: Path
) -> None:
    """Monitor (inside the wrapper) logs the original reward, not the shaped one."""
    from conftest import smoke_config_with_reward

    from lunar_lander_rl.train import build_vec_env

    config = smoke_config_with_reward(smoke_config, "potential_based")
    env = build_vec_env(config, tmp_path)
    episodes: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    try:
        env.reset()
        rng = np.random.default_rng(0)
        while len(episodes) < 4:
            _, _, _, infos = env.step(rng.integers(0, 4, size=env.num_envs))
            for rank, info in enumerate(infos):
                if EPISODE_INFO_KEY in info:
                    episodes.append((rank, info[EPISODE_INFO_KEY], info["episode"]))
    finally:
        env.close()
    for _, shaped, monitor in episodes:
        assert monitor["r"] == pytest.approx(shaped["original"], abs=1e-5)
        assert monitor["l"] == shaped["length"]
        assert shaped["total"] != pytest.approx(shaped["original"])
    monitor_returns = []
    for rank in range(config.training.n_envs):
        with (tmp_path / f"{rank}.monitor.csv").open(encoding="utf-8") as handle:
            next(handle)  # JSON header line
            monitor_returns += [float(row["r"]) for row in csv.DictReader(handle)]
    assert sorted(monitor_returns) == pytest.approx(
        sorted(shaped["original"] for _, shaped, _ in episodes), abs=1e-5
    )


@pytest.mark.parametrize("mode", ["original", "potential_based"])
def test_training_env_wraps_every_mode_with_configured_reward_and_ppo_gamma(
    smoke_config: Any, mode: str
) -> None:
    """The wrapper is really present (also in original mode); gamma = ppo.gamma."""
    from conftest import smoke_config_with_reward

    from lunar_lander_rl.train import build_vec_env

    config = smoke_config_with_reward(smoke_config, mode)
    env = build_vec_env(config, monitor_dir=None)
    try:
        for inner in env.envs:  # type: ignore[attr-defined]
            assert isinstance(inner, RewardComponentsWrapper)
            assert inner._reward == config.reward
            assert inner._gamma == config.ppo.gamma
            assert type(inner.env).__name__ == "Monitor"  # applied outside Monitor
    finally:
        env.close()


def test_evaluation_and_video_environments_are_unshaped(tmp_path: Path) -> None:
    def chain(env: gym.Env[Any, Any]) -> list[type]:
        types = [type(env)]
        while isinstance(env, gym.Wrapper):
            env = env.env
            types.append(type(env))
        return types

    env = make_env()
    try:
        assert RewardComponentsWrapper not in chain(env)
    finally:
        env.close()
    pytest.importorskip("moviepy")
    from lunar_lander_rl.record_video import video_env_factory

    video_env = video_env_factory(tmp_path / "clips")()
    try:
        assert RewardComponentsWrapper not in chain(video_env)
    finally:
        video_env.close()


def test_test_rewards_cover_every_mode() -> None:
    from lunar_lander_rl.config import SUPPORTED_REWARD_MODES

    assert sorted(TEST_REWARDS) == sorted(SUPPORTED_REWARD_MODES)
