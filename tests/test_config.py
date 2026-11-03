"""Configuration loading and validation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lunar_lander_rl.config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    parse_episode_set,
)

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_baseline_config_loads_with_documented_values() -> None:
    config = load_config(BASELINE)
    assert config.experiment_id == "baseline"
    assert config.environment.id == "LunarLander-v3"
    assert config.reward.mode == "original"
    assert config.training.total_timesteps == 1_000_000
    assert config.training.seed == 42
    assert config.training.n_envs == 16
    assert config.training.device == "cpu"
    assert config.ppo.policy == "MlpPolicy"
    assert config.ppo.net_arch == (64, 64)
    # RL Zoo LunarLander-v3 values (see configs/baseline.yaml header).
    assert (config.ppo.n_steps, config.ppo.batch_size, config.ppo.n_epochs) == (
        1024,
        64,
        4,
    )
    assert (config.ppo.gamma, config.ppo.gae_lambda, config.ppo.ent_coef) == (
        0.999,
        0.98,
        0.01,
    )


def test_round_trip_through_dict_and_json() -> None:
    config = load_config(BASELINE)
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert (
        ExperimentConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    )


def test_overrides_apply_and_are_revalidated() -> None:
    config = load_config(BASELINE).with_overrides(total_timesteps=5000, seed=7)
    assert config.training.total_timesteps == 5000
    assert config.training.seed == 7
    with pytest.raises(ConfigError, match="total_timesteps"):
        config.with_overrides(total_timesteps=0)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("experiment_id",), "../escape", "experiment_id"),
        (("experiment_id",), "Baseline", "experiment_id"),
        (("environment", "id"), "CartPole-v1", "environment.id"),
        (("reward", "mode"), "time_penalty", "reward.mode"),
        (("training", "total_timesteps"), 0, "total_timesteps"),
        (("training", "total_timesteps"), 1.5, "total_timesteps"),
        (("training", "seed"), -1, "seed"),
        (("training", "n_envs"), 0, "n_envs"),
        (("training", "checkpoint_freq"), -1, "checkpoint_freq"),
        (("training", "checkpoint_freq"), 100001, "checkpoint_freq"),
        (("training", "seed"), True, "seed"),
        (("training", "n_envs"), False, "n_envs"),
        (("evaluation", "episodes"), True, "episodes"),
        (("training", "device"), "cuda", "device"),
        (("ppo", "policy"), "CnnPolicy", "policy"),
        (("ppo", "net_arch"), [], "net_arch"),
        (("ppo", "net_arch"), [64, 0], "net_arch"),
        (("ppo", "net_arch"), "64,64", "net_arch"),
        (("ppo", "learning_rate"), 0, "learning_rate"),
        (("ppo", "learning_rate"), True, "learning_rate"),
        (("ppo", "gamma"), 0.0, "gamma"),
        (("ppo", "gamma"), 1.5, "gamma"),
        (("ppo", "gae_lambda"), -0.1, "gae_lambda"),
        (("ppo", "batch_size"), 100, "batch_size"),
        (("ppo", "n_epochs"), 0, "n_epochs"),
        (("evaluation", "episodes"), 0, "episodes"),
        (("evaluation", "deterministic"), "yes", "deterministic"),
        (("video", "seed_start"), -5, "seed_start"),
    ],
)
def test_invalid_values_are_rejected(
    config_dict: dict[str, Any], path: tuple[str, ...], value: Any, message: str
) -> None:
    set_path(config_dict, path, value)
    with pytest.raises(ConfigError, match=message):
        ExperimentConfig.from_dict(config_dict)


def test_unknown_key_is_rejected(config_dict: dict[str, Any]) -> None:
    config_dict["ppo"]["learning_rat"] = 0.001
    with pytest.raises(ConfigError, match=r"ppo: unknown key\(s\) \['learning_rat'\]"):
        ExperimentConfig.from_dict(config_dict)


def test_missing_key_is_rejected(config_dict: dict[str, Any]) -> None:
    del config_dict["training"]["seed"]
    with pytest.raises(ConfigError, match=r"training: missing key\(s\) \['seed'\]"):
        ExperimentConfig.from_dict(config_dict)


def test_non_mapping_section_is_rejected(config_dict: dict[str, Any]) -> None:
    config_dict["evaluation"] = [100]
    with pytest.raises(ConfigError, match="evaluation: expected a mapping"):
        ExperimentConfig.from_dict(config_dict)


@pytest.mark.parametrize(
    ("section", "seed_start"),
    [("evaluation", 50), ("video", 57), ("video", 10099)],
)
def test_overlapping_seed_ranges_are_rejected(
    config_dict: dict[str, Any], section: str, seed_start: int
) -> None:
    # Training env seeds are 42..57; evaluation is 10000..10099 (100 episodes).
    config_dict[section]["seed_start"] = seed_start
    with pytest.raises(ConfigError, match="seed ranges overlap"):
        ExperimentConfig.from_dict(config_dict)


def test_seed_override_is_checked_for_overlap() -> None:
    with pytest.raises(ConfigError, match="seed ranges overlap"):
        load_config(BASELINE).with_overrides(seed=9990)


def test_episode_set_section_parses_on_its_own() -> None:
    settings = parse_episode_set(
        {"episodes": 5, "deterministic": False, "seed_start": 3}, "evaluation"
    )
    assert (settings.episodes, settings.deterministic, settings.seed_start) == (
        5,
        False,
        3,
    )
    with pytest.raises(ConfigError, match="video: unknown key"):
        parse_episode_set(
            {"episodes": 1, "deterministic": True, "seed_start": 0, "x": 1}, "video"
        )


@pytest.mark.parametrize("text", ["", "[1, 2]\n", "just a string\n"])
def test_non_mapping_yaml_root_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "root.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="config: expected a mapping"):
        load_config(path)


def test_missing_file_and_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("training: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(broken)
