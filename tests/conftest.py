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


# Placeholder reward parameters for tests only; Phase 04 chooses real values.
TEST_REWARDS: dict[str, dict[str, Any]] = {
    "original": {"mode": "original"},
    "time_penalty": {"mode": "time_penalty", "coefficient": 0.01},
    "upright_slow_bonus": {
        "mode": "upright_slow_bonus",
        "bonus": 1.0,
        "max_abs_angle": 0.1,
        "max_speed": 0.1,
    },
    "potential_based": {
        "mode": "potential_based",
        "alpha": 1.0,
        "distance_weight": 1.0,
        "speed_weight": 1.0,
        "angle_weight": 1.0,
    },
}


def smoke_config_with_reward(
    smoke_config: ExperimentConfig, mode: str, **training: Any
) -> ExperimentConfig:
    """``smoke_config`` with another reward mode (and optional training overrides)."""
    data = smoke_config.to_dict()
    data["experiment_id"] = f"smoke-{mode.replace('_', '-')}"
    data["reward"] = copy.deepcopy(TEST_REWARDS[mode])
    data["training"].update(training)
    return ExperimentConfig.from_dict(data)


@pytest.fixture(scope="session")
def shaped_smoke_run(
    smoke_config: ExperimentConfig, tmp_path_factory: pytest.TempPathFactory
) -> TrainingResult:
    """A potential-shaping run long enough (4 rollouts) for episodes to finish."""
    config = smoke_config_with_reward(
        smoke_config, "potential_based", total_timesteps=512, checkpoint_freq=0
    )
    return train(
        config,
        tmp_path_factory.mktemp("shaped-runs"),
        argv=["pytest-shaped-smoke"],
        timestamp=SMOKE_TIMESTAMP,
    )
