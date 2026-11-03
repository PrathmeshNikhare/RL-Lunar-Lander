"""Shared fixtures: a tiny, fast experiment config and one smoke-trained run.

The smoke run trains for a single PPO rollout (2 envs x 64 steps) so that
training, saving, loading, evaluation and video tests stay in the seconds
range. It is never a substitute for a real training run.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from lunar_lander_rl.config import ExperimentConfig
from lunar_lander_rl.train import TrainingResult, train

BASELINE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
SMOKE_TIMESTAMP = "20000101T000000Z"


@pytest.fixture(scope="session")
def baseline_dict() -> dict[str, Any]:
    with BASELINE_CONFIG_PATH.open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return data


@pytest.fixture
def config_dict(baseline_dict: dict[str, Any]) -> dict[str, Any]:
    """A mutable deep copy of the baseline config for per-test modification."""
    return copy.deepcopy(baseline_dict)


@pytest.fixture(scope="session")
def smoke_config(baseline_dict: dict[str, Any]) -> ExperimentConfig:
    data = copy.deepcopy(baseline_dict)
    data["experiment_id"] = "smoke"
    data["training"].update(total_timesteps=128, n_envs=2, checkpoint_freq=64)
    data["ppo"].update(n_steps=64, batch_size=32, n_epochs=1)
    data["evaluation"].update(episodes=2)
    data["video"].update(episodes=1)
    return ExperimentConfig.from_dict(data)


@pytest.fixture(scope="session")
def smoke_run(
    smoke_config: ExperimentConfig, tmp_path_factory: pytest.TempPathFactory
) -> TrainingResult:
    runs_root = tmp_path_factory.mktemp("runs")
    return train(
        smoke_config, runs_root, argv=["pytest-smoke"], timestamp=SMOKE_TIMESTAMP
    )
