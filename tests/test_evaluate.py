"""Evaluation tests: original-reward accounting, schema and write-once outputs."""

from __future__ import annotations

import csv
import inspect
import json
import shutil
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

from lunar_lander_rl.artifacts import read_json, sha256_file
from lunar_lander_rl.config import ExperimentConfig
from lunar_lander_rl.environment import ENV_ID, Observation, make_env
from lunar_lander_rl.evaluate import (
    EPISODE_FIELDS,
    REWARD_THRESHOLD,
    TRAJECTORIES_FILENAME,
    EpisodeResult,
    EpisodeSteps,
    evaluate_run,
    read_trajectories,
    run_episodes,
    summarize,
    write_trajectories,
)
from lunar_lander_rl.train import TrainingResult

SEED_START = 500
EPISODES = 3


def _do_nothing(_observation: Observation) -> int:
    return 0


def _manual_rollout(seed: int) -> tuple[float, int, bool, bool]:
    """Independent reference: sum env rewards for the do-nothing policy."""
    env = make_env()
    try:
        env.reset(seed=seed)
        total, length = 0.0, 0
        while True:
            _, reward, terminated, truncated, _ = env.step(np.int64(0))
            total += float(reward)
            length += 1
            if terminated or truncated:
                return total, length, bool(terminated), bool(truncated)
    finally:
        env.close()


def _read_rows(session: Path) -> list[dict[str, str]]:
    with (session / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_returns_equal_independent_original_reward_rollout() -> None:
    results = run_episodes(_do_nothing, EPISODES, SEED_START)
    assert [r.seed for r in results] == [SEED_START + i for i in range(EPISODES)]
    for result in results:
        total, length, terminated, truncated = _manual_rollout(result.seed)
        assert result.original_return == total
        assert result.length == length
        assert (result.terminated, result.truncated) == (terminated, truncated)


def test_evaluation_uses_plain_environment_factory() -> None:
    default = inspect.signature(run_episodes).parameters["env_factory"].default
    assert default is make_env


def test_reward_threshold_matches_registered_spec() -> None:
    env = make_env()
    try:
        assert env.spec is not None
        assert env.spec.reward_threshold == REWARD_THRESHOLD
    finally:
        env.close()


def test_episode_schema() -> None:
    (result,) = run_episodes(_do_nothing, 1, SEED_START)
    assert EPISODE_FIELDS == (
        "episode",
        "seed",
        "original_return",
        "length",
        "terminated",
        "truncated",
        "final_reward",
        "final_x",
        "final_y",
        "final_left_leg_contact",
        "final_right_leg_contact",
    )
    assert isinstance(result.original_return, float)
    assert isinstance(result.length, int) and result.length >= 1
    assert result.terminated or result.truncated


def test_time_limited_episode_is_reported_as_truncated() -> None:
    (result,) = run_episodes(
        _do_nothing, 1, SEED_START, env_factory=lambda: make_env(max_episode_steps=5)
    )
    assert result.length == 5
    assert result.truncated and not result.terminated


def _result(
    episode: int,
    ret: float,
    length: int,
    final_reward: float,
    *,
    terminated: bool = True,
) -> EpisodeResult:
    return EpisodeResult(
        episode=episode,
        seed=episode,
        original_return=ret,
        length=length,
        terminated=terminated,
        truncated=not terminated,
        final_reward=final_reward,
        final_x=0.0,
        final_y=0.0,
        final_left_leg_contact=1.0,
        final_right_leg_contact=1.0,
    )


def test_summary_statistics_hand_calculated() -> None:
    results = [
        _result(0, 100.0, 200, -100),
        _result(1, 200.0, 300, 100),
        _result(2, 300.0, 400, 100),
    ]
    summary = summarize(results)
    assert summary["n_episodes"] == 3
    assert summary["mean_return"] == pytest.approx(200.0)
    assert summary["std_return"] == pytest.approx(100.0)  # sample std, ddof=1
    assert summary["median_return"] == pytest.approx(200.0)
    assert (summary["min_return"], summary["max_return"]) == (100.0, 300.0)
    assert summary["mean_length"] == pytest.approx(300.0)
    assert summary["std_length"] == pytest.approx(100.0)
    assert summary["n_terminated"] == 3 and summary["n_truncated"] == 0
    assert summary["n_final_reward_plus_100"] == 2
    assert summary["n_final_reward_minus_100"] == 1
    assert summary["n_return_at_least_threshold"] == 2
    assert summary["mean_return_at_least_threshold"] is True


def test_summary_even_median_and_truncation_counts() -> None:
    results = [
        _result(
            0, 10.0, 1000, 100.0, terminated=False
        ),  # truncated: not a +100 terminal
        _result(1, 20.0, 50, -100.0),
        _result(2, 30.0, 60, 100.0),
        _result(3, 40.0, 70, 0.5, terminated=False),
    ]
    summary = summarize(results)
    assert summary["median_return"] == pytest.approx(25.0)
    assert summary["n_truncated"] == 2 and summary["n_terminated"] == 2
    assert summary["n_final_reward_plus_100"] == 1
    assert summary["n_final_reward_minus_100"] == 1
    assert summary["mean_return_at_least_threshold"] is False


def test_single_episode_std_is_zero() -> None:
    assert summarize([_result(0, 5.0, 10, -100)])["std_return"] == 0.0


@pytest.mark.parametrize(("episodes", "seed_start"), [(0, 0), (1, -1)])
def test_invalid_episode_arguments(episodes: int, seed_start: int) -> None:
    with pytest.raises(ValueError):
        run_episodes(_do_nothing, episodes, seed_start)


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize([])


def test_evaluate_run_writes_session_and_never_overwrites(
    smoke_run: TrainingResult, smoke_config: ExperimentConfig
) -> None:
    evaluation = smoke_config.evaluation
    session = evaluate_run(smoke_run.run_dir, timestamp="eval-test")
    assert session == smoke_run.run_dir / "evaluations" / "eval-test"
    rows = _read_rows(session)
    assert len(rows) == evaluation.episodes
    assert tuple(rows[0]) == EPISODE_FIELDS
    assert [int(row["seed"]) for row in rows] == [
        evaluation.seed_start + i for i in range(evaluation.episodes)
    ]
    payload = read_json(session / "summary.json")
    assert payload["reward"] == "original environment reward (no wrappers)"
    assert payload["env_id"] == ENV_ID
    assert payload["max_episode_steps"] == 1000
    assert payload["render_mode"] is None
    assert payload["model_sha256"] == sha256_file(smoke_run.model_path)
    assert payload["settings"] == {
        "episodes": evaluation.episodes,
        "deterministic": evaluation.deterministic,
        "seed_start": evaluation.seed_start,
    }
    assert payload["summary"]["mean_return"] == statistics.fmean(
        float(row["original_return"]) for row in rows
    )
    with pytest.raises(FileExistsError):
        evaluate_run(smoke_run.run_dir, timestamp="eval-test")


@pytest.mark.parametrize("run_fixture", ["smoke_run", "shaped_smoke_run"])
def test_evaluate_run_matches_independent_model_rollout(
    run_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Trained-model returns equal a rollout on a raw ``gym.make`` env.

    ``shaped_smoke_run`` was trained with potential shaping: evaluation must
    still report the original environment reward only.
    """
    run: TrainingResult = request.getfixturevalue(run_fixture)
    session = evaluate_run(run.run_dir, episodes=2, timestamp="eval-reference")
    model = PPO.load(run.model_path, device="cpu")
    for row in _read_rows(session):
        env = gym.make(ENV_ID)
        try:
            observation, _ = env.reset(seed=int(row["seed"]))
            total, length = 0.0, 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(int(action))
                total += float(reward)
                length += 1
        finally:
            env.close()
        assert float(row["original_return"]) == total
        assert int(row["length"]) == length
        assert row["terminated"] == str(terminated)
        assert row["truncated"] == str(truncated)


def test_evaluate_run_overrides_and_missing_model(smoke_run: TrainingResult) -> None:
    session = evaluate_run(
        smoke_run.run_dir, episodes=1, seed_start=77, timestamp="eval-override"
    )
    payload = read_json(session / "summary.json")
    assert payload["settings"]["episodes"] == 1
    assert payload["settings"]["seed_start"] == 77
    with pytest.raises(FileNotFoundError):
        evaluate_run(
            smoke_run.run_dir, model_filename="missing.zip", timestamp="eval-missing"
        )
    assert not (smoke_run.run_dir / "evaluations" / "eval-missing").exists()


def test_old_runs_stay_evaluable_when_other_config_sections_change(
    smoke_run: TrainingResult, tmp_path: Path
) -> None:
    """Only the evaluation section of config.json is read (decision D-020)."""
    copy_dir = tmp_path / smoke_run.run_dir.name
    copy_dir.mkdir()
    shutil.copy2(smoke_run.model_path, copy_dir / "model.zip")
    saved = read_json(smoke_run.run_dir / "config.json")
    saved["reward"] = {"mode": "future_mode", "new_parameter": 1.0}
    saved["future_section"] = {"anything": True}
    (copy_dir / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    session = evaluate_run(copy_dir, episodes=1, timestamp="eval-schema")
    assert len(_read_rows(session)) == 1


def _alternating(observation: Observation) -> int:
    """A deterministic, observation-dependent policy that fires engines."""
    return 2 if observation[3] < -0.2 else (1 if observation[4] > 0 else 3)


def test_step_log_is_consistent_with_episode_results() -> None:
    steps: list[EpisodeSteps] = []
    results = run_episodes(_alternating, EPISODES, SEED_START, step_log=steps)
    assert len(steps) == len(results)
    for result, episode in zip(results, steps, strict=True):
        assert episode.observations.shape == (result.length + 1, 8)
        assert episode.actions.shape == episode.rewards.shape == (result.length,)
        total = 0.0
        for reward in episode.rewards:
            total += float(reward)
        assert total == result.original_return
        assert float(episode.rewards[-1]) == result.final_reward
        assert float(episode.observations[-1][0]) == result.final_x
        assert float(episode.observations[-1][1]) == result.final_y
        env = make_env()
        try:
            reset_obs, _ = env.reset(seed=result.seed)
        finally:
            env.close()
        assert np.array_equal(episode.observations[0], reset_obs)


def test_step_recording_does_not_change_the_rollout() -> None:
    plain = run_episodes(_alternating, EPISODES, SEED_START)
    recorded = run_episodes(_alternating, EPISODES, SEED_START, step_log=[])
    assert plain == recorded


def test_trajectories_round_trip_and_are_write_once(tmp_path: Path) -> None:
    steps: list[EpisodeSteps] = []
    results = run_episodes(_alternating, 2, SEED_START, step_log=steps)
    path = tmp_path / TRAJECTORIES_FILENAME
    write_trajectories(path, [r.seed for r in results], steps)
    seeds, loaded = read_trajectories(path)
    assert seeds == [r.seed for r in results]
    for original, copy in zip(steps, loaded, strict=True):
        assert np.array_equal(original.observations, copy.observations)
        assert np.array_equal(original.actions, copy.actions)
        assert np.array_equal(original.rewards, copy.rewards)
    with pytest.raises(FileExistsError):
        write_trajectories(path, seeds, steps)
    with pytest.raises(ValueError, match="seeds"):
        write_trajectories(tmp_path / "other.npz", seeds[:1], steps)


def test_read_trajectories_rejects_inconsistent_arrays(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        seeds=np.asarray([1]),
        lengths=np.asarray([3]),
        observations=np.zeros((3, 8), dtype=np.float32),  # needs 4
        actions=np.zeros(3, dtype=np.int8),
        rewards=np.zeros(3),
    )
    with pytest.raises(ValueError, match="sizes"):
        read_trajectories(path)


def test_evaluate_run_saves_trajectories(smoke_run: TrainingResult) -> None:
    session = evaluate_run(smoke_run.run_dir, timestamp="eval-trajectories")
    rows = _read_rows(session)
    seeds, steps = read_trajectories(session / TRAJECTORIES_FILENAME)
    assert seeds == [int(row["seed"]) for row in rows]
    assert [len(s.actions) for s in steps] == [int(row["length"]) for row in rows]
