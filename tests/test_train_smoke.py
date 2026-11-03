"""Short PPO train / save / load smoke tests (one rollout, seconds not minutes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from lunar_lander_rl.artifacts import read_json
from lunar_lander_rl.config import ExperimentConfig
from lunar_lander_rl.environment import make_env
from lunar_lander_rl.train import TrainingResult, build_model, build_vec_env, train

N_PROBE_OBSERVATIONS = 16


def _probe_observations() -> np.ndarray:
    env = make_env()
    try:
        return np.stack(
            [env.reset(seed=seed)[0] for seed in range(N_PROBE_OBSERVATIONS)]
        )
    finally:
        env.close()


def test_run_directory_contains_expected_artifacts(
    smoke_run: TrainingResult, smoke_config: ExperimentConfig
) -> None:
    run_dir = smoke_run.run_dir
    assert run_dir.name.endswith(f"_seed{smoke_config.training.seed}")
    for name in ("config.json", "metadata.json", "model.zip"):
        assert (run_dir / name).is_file(), name
    assert sorted(p.name for p in (run_dir / "monitor").glob("*.monitor.csv")) == [
        f"{rank}.monitor.csv" for rank in range(smoke_config.training.n_envs)
    ]
    assert list((run_dir / "tensorboard").rglob("events.out.tfevents*"))
    # checkpoint_freq=64 total timesteps with 2 envs -> every 32 vector steps.
    assert {p.name for p in (run_dir / "checkpoints").iterdir()} == {
        "ppo_64_steps.zip",
        "ppo_128_steps.zip",
    }


def test_saved_config_round_trips(
    smoke_run: TrainingResult, smoke_config: ExperimentConfig
) -> None:
    saved = ExperimentConfig.from_dict(read_json(smoke_run.run_dir / "config.json"))
    assert saved == smoke_config


def test_metadata_records_completion_and_provenance(
    smoke_run: TrainingResult, smoke_config: ExperimentConfig
) -> None:
    meta = read_json(smoke_run.run_dir / "metadata.json")
    training = smoke_config.training
    assert meta["status"] == "completed"
    assert meta["requested_timesteps"] == training.total_timesteps
    assert meta["num_timesteps"] >= training.total_timesteps
    assert meta["num_timesteps"] == smoke_run.num_timesteps
    assert meta["training_seed"] == training.seed
    assert meta["env_seeds"] == [training.seed + i for i in range(training.n_envs)]
    assert meta["duration_seconds"] > 0
    assert meta["argv"] == ["pytest-smoke"]
    assert isinstance(meta["torch_num_threads"], int)
    for key in ("packages", "source", "python", "started_at", "finished_at"):
        assert key in meta


def test_metadata_records_effective_hyperparameters(
    smoke_run: TrainingResult, smoke_config: ExperimentConfig
) -> None:
    hp = read_json(smoke_run.run_dir / "metadata.json")["effective_hyperparameters"]
    for key, value in smoke_config.ppo.algorithm_kwargs().items():
        assert hp[key] == pytest.approx(value), key
    # Implicit SB3 defaults that the config does not set are still recorded.
    assert hp["normalize_advantage"] is True
    assert hp["target_kl"] is None
    assert hp["use_sde"] is False
    assert hp["policy"]["net_arch"] == [64, 64]
    assert hp["policy"]["activation_fn"] == "Tanh"
    assert hp["policy"]["ortho_init"] is True


def test_loaded_model_has_configured_architecture(smoke_run: TrainingResult) -> None:
    model = PPO.load(smoke_run.model_path, device="cpu")
    linear_out = [
        layer.out_features
        for layer in model.policy.mlp_extractor.policy_net
        if isinstance(layer, torch.nn.Linear)
    ]
    assert linear_out == [64, 64]
    assert model.gamma == pytest.approx(0.999)


def test_save_load_preserves_actions_and_training_is_reproducible(
    smoke_config: ExperimentConfig, smoke_run: TrainingResult, tmp_path: Path
) -> None:
    env = build_vec_env(smoke_config, monitor_dir=None)
    try:
        model = build_model(smoke_config, env, tensorboard_dir=None)
        model.learn(total_timesteps=smoke_config.training.total_timesteps)
        path = tmp_path / "model.zip"
        model.save(path)
    finally:
        env.close()
    loaded = PPO.load(path, device="cpu")
    observations = _probe_observations()
    original_actions, _ = model.predict(observations, deterministic=True)
    loaded_actions, _ = loaded.predict(observations, deterministic=True)
    np.testing.assert_array_equal(original_actions, loaded_actions)
    assert set(np.unique(loaded_actions)) <= {0, 1, 2, 3}
    # Same config and seed on the same machine -> same trained policy.
    smoke_model = PPO.load(smoke_run.model_path, device="cpu")
    smoke_actions, _ = smoke_model.predict(observations, deterministic=True)
    np.testing.assert_array_equal(loaded_actions, smoke_actions)


def test_training_refuses_to_reuse_a_run_directory(
    smoke_run: TrainingResult, smoke_config: ExperimentConfig
) -> None:
    runs_root = smoke_run.run_dir.parents[1]
    model_bytes = smoke_run.model_path.read_bytes()
    with pytest.raises(FileExistsError):
        train(
            smoke_config, runs_root, timestamp=smoke_run.run_dir.name.split("_seed")[0]
        )
    assert smoke_run.model_path.read_bytes() == model_bytes


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [(RuntimeError("boom"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_failed_training_still_writes_metadata(
    smoke_config: ExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    expected_status: str,
) -> None:
    def explode(self: PPO, *args: Any, **kwargs: Any) -> PPO:
        raise error

    monkeypatch.setattr(PPO, "learn", explode)
    with pytest.raises(type(error)):
        train(smoke_config, tmp_path, timestamp="failcase")
    run_dir = (
        tmp_path
        / smoke_config.experiment_id
        / f"failcase_seed{smoke_config.training.seed}"
    )
    meta = read_json(run_dir / "metadata.json")
    assert meta["status"] == expected_status
    assert meta["num_timesteps"] == 0
    assert meta["model_path"] is None
    assert meta["effective_hyperparameters"] is not None
    assert (run_dir / "config.json").is_file()
    assert not (run_dir / "model.zip").exists()


def test_environment_build_failure_still_writes_metadata(
    smoke_config: ExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_env(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("env build failed")

    monkeypatch.setattr("lunar_lander_rl.train.build_vec_env", broken_env)
    with pytest.raises(RuntimeError, match="env build failed"):
        train(smoke_config, tmp_path, timestamp="envfail")
    run_dir = (
        tmp_path
        / smoke_config.experiment_id
        / f"envfail_seed{smoke_config.training.seed}"
    )
    meta = read_json(run_dir / "metadata.json")
    assert meta["status"] == "failed"
    assert meta["effective_hyperparameters"] is None
