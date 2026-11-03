"""Tests for the per-episode reward-component logging callback."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import smoke_config_with_reward

from lunar_lander_rl.artifacts import read_json
from lunar_lander_rl.config import ExperimentConfig, TimePenaltyReward
from lunar_lander_rl.reward_logging import RewardComponentsLogger
from lunar_lander_rl.train import train

REWARD = TimePenaltyReward(coefficient=0.25)
HEADER = ["timesteps", "env_rank", "length", "original", "time_penalty", "total"]


class FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    def record_mean(self, key: str, value: float) -> None:
        self.records.append((key, value))


def episode(length: int, original: float) -> dict[str, Any]:
    penalty = -0.25 * length
    return {
        "length": length,
        "original": original,
        "time_penalty": penalty,
        "total": original + penalty,
    }


def started(path: Path) -> tuple[RewardComponentsLogger, SimpleNamespace]:
    callback = RewardComponentsLogger(path, REWARD)
    model = SimpleNamespace(logger=FakeLogger(), num_timesteps=0)
    callback.init_callback(model)  # type: ignore[arg-type]
    callback.on_training_start({}, {})
    return callback, model


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_rows_per_finished_episode_with_rank_and_timesteps(tmp_path: Path) -> None:
    path = tmp_path / "reward_components.csv"
    callback, model = started(path)
    model.num_timesteps = 2
    callback.update_locals({"infos": [{}, {}]})  # nothing finished
    callback.on_step()
    model.num_timesteps = 4
    callback.update_locals(
        {
            "infos": [
                {"reward_episode": episode(4, 10.0)},
                {"reward_episode": episode(2, -5.0)},
            ]
        }
    )
    callback.on_step()
    callback.close()
    rows = read_rows(path)
    assert list(rows[0]) == HEADER
    assert [(r["timesteps"], r["env_rank"], r["length"]) for r in rows] == [
        ("4", "0", "4"),
        ("4", "1", "2"),
    ]
    assert float(rows[1]["total"]) == -5.5
    keys = [key for key, _ in model.logger.records]
    assert keys == ["reward/original", "reward/time_penalty", "reward/total"] * 2


def test_refuses_to_overwrite_an_existing_csv(tmp_path: Path) -> None:
    path = tmp_path / "reward_components.csv"
    path.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        started(path)
    assert path.read_text(encoding="utf-8") == "keep me"


def test_close_is_safe_before_start_and_twice(tmp_path: Path) -> None:
    callback = RewardComponentsLogger(tmp_path / "unused.csv", REWARD)
    callback.close()
    assert not (tmp_path / "unused.csv").exists()
    started_callback, _ = started(tmp_path / "used.csv")
    started_callback.close()
    started_callback.close()


def test_interrupted_training_leaves_a_closed_csv_with_rows(
    smoke_config: ExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_on_step = RewardComponentsLogger._on_step

    def interrupt_after_first_episode(self: RewardComponentsLogger) -> bool:
        result = original_on_step(self)
        if any("reward_episode" in info for info in self.locals["infos"]):
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(
        RewardComponentsLogger, "_on_step", interrupt_after_first_episode
    )
    config = smoke_config_with_reward(
        smoke_config, "time_penalty", total_timesteps=2048, checkpoint_freq=0
    )
    with pytest.raises(KeyboardInterrupt):
        train(config, tmp_path, timestamp="interrupted")
    run_dir = (
        tmp_path / config.experiment_id / f"interrupted_seed{config.training.seed}"
    )
    assert read_json(run_dir / "metadata.json")["status"] == "interrupted"
    path = run_dir / "reward_components.csv"
    assert len(read_rows(path)) >= 1  # the row was flushed
    # On Windows an open handle blocks renaming, so this proves it was closed.
    path.rename(run_dir / "renamed.csv")
