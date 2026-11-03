"""The four Phase 04 experiment configs: valid, selectable and otherwise identical.

Fairness (docs/06-experiment-protocol.md): only ``experiment_id`` and
``reward`` may differ from the baseline, so PPO settings, budget and
evaluation/video seeds stay matched across experiments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lunar_lander_rl.config import SUPPORTED_REWARD_MODES, ExperimentConfig, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = {
    "baseline": "original",
    "time_penalty": "time_penalty",
    "upright_slow_bonus": "upright_slow_bonus",
    "potential_shaping": "potential_based",
}
PHASE04_SEEDS = (42, 142, 242)


def _load(name: str) -> ExperimentConfig:
    return load_config(CONFIGS / f"{name}.yaml")


def test_every_reward_mode_has_exactly_one_experiment_config() -> None:
    assert sorted(EXPERIMENTS.values()) == sorted(SUPPORTED_REWARD_MODES)


@pytest.mark.parametrize(("name", "mode"), sorted(EXPERIMENTS.items()))
def test_config_loads_with_its_id_and_mode(name: str, mode: str) -> None:
    config = _load(name)
    assert config.experiment_id == name
    assert config.reward.mode == mode


@pytest.mark.parametrize("name", sorted(set(EXPERIMENTS) - {"baseline"}))
def test_only_id_and_reward_differ_from_baseline(name: str) -> None:
    baseline = _load("baseline").to_dict()
    other = _load(name).to_dict()
    for section in ("experiment_id", "reward"):
        baseline.pop(section)
        other.pop(section)
    assert other == baseline


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
@pytest.mark.parametrize("seed", PHASE04_SEEDS)
def test_phase04_seed_overrides_are_valid(name: str, seed: int) -> None:
    config = _load(name).with_overrides(seed=seed)
    assert config.training.seed == seed


def test_phase04_training_env_seed_blocks_are_disjoint() -> None:
    n_envs = _load("baseline").training.n_envs
    blocks = [set(range(seed, seed + n_envs)) for seed in PHASE04_SEEDS]
    assert sum(len(b) for b in blocks) == len(set().union(*blocks))
