"""Configuration-driven PPO training (Stable-Baselines3, ``MlpPolicy``).

Training only: evaluation (``lunar_lander_rl.evaluate``) and video recording
(``lunar_lander_rl.record_video``) are separate and load the saved model.

Each call to :func:`train` creates a new run directory and writes:

- ``config.json``: the effective configuration, written before training starts;
- ``monitor/<rank>.monitor.csv``: per-episode training returns and lengths,
  using the environment's original reward;
- ``reward_components.csv``: per-episode original reward, each custom reward
  component and the shaped total (``lunar_lander_rl.reward_logging``);
- ``tensorboard/``: SB3 training scalars;
- ``checkpoints/``: periodic model snapshots, if enabled;
- ``model.zip``: the final model;
- ``metadata.json``: written once, at the end, even if training fails. It
  holds:
  - the status;
  - timings;
  - the effective hyperparameters read back from the constructed model;
  - versions;
  - the git revision captured at launch.

Seeding: SB3 seeds torch, NumPy and Python ``random`` with ``training.seed``.
Vectorised environment ``i`` is seeded with ``training.seed + i`` at its
first reset; later auto-resets draw from each environment's own RNG.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from lunar_lander_rl.artifacts import (
    CHECKPOINTS_SUBDIR,
    CONFIG_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    MONITOR_SUBDIR,
    REWARD_COMPONENTS_FILENAME,
    TENSORBOARD_SUBDIR,
    create_run_dir,
    environment_metadata,
    write_json,
)
from lunar_lander_rl.config import ExperimentConfig
from lunar_lander_rl.environment import make_env
from lunar_lander_rl.reward_logging import RewardComponentsLogger
from lunar_lander_rl.reward_wrapper import RewardComponentsWrapper

TENSORBOARD_RUN_NAME: Final[str] = "ppo"
CHECKPOINT_PREFIX: Final[str] = "ppo"

# PPO attributes read back from the constructed model for metadata.json, so
# implicit SB3 defaults (e.g. normalize_advantage, target_kl) are recorded too.
_RECORDED_ALGORITHM_ATTRIBUTES: Final[tuple[str, ...]] = (
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "normalize_advantage",
    "target_kl",
    "clip_range_vf",
    "use_sde",
    "sde_sample_freq",
)


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    model_path: Path
    num_timesteps: int
    duration_seconds: float


def build_vec_env(config: ExperimentConfig, monitor_dir: Path | None) -> VecEnv:
    """Create the in-process vectorised training environment.

    Every mode, including ``original``, is wrapped by
    ``RewardComponentsWrapper``. It is passed to ``make_vec_env`` as
    ``wrapper_class``, which SB3 applies *outside* ``Monitor``, so
    ``monitor/*.csv`` keeps logging the original environment reward. Wrapping
    inside the env factory would silently make Monitor log the shaped reward.
    """
    return make_vec_env(
        make_env,
        n_envs=config.training.n_envs,
        seed=config.training.seed,
        monitor_dir=None if monitor_dir is None else str(monitor_dir),
        wrapper_class=functools.partial(
            RewardComponentsWrapper, reward=config.reward, gamma=config.ppo.gamma
        ),
        vec_env_cls=DummyVecEnv,
    )


def build_model(
    config: ExperimentConfig,
    env: VecEnv,
    tensorboard_dir: Path | None,
    verbose: int = 0,
) -> PPO:
    """Construct an untrained PPO model from the configuration."""
    return PPO(
        config.ppo.policy,
        env,
        policy_kwargs={"net_arch": list(config.ppo.net_arch)},
        seed=config.training.seed,
        device=config.training.device,
        tensorboard_log=None if tensorboard_dir is None else str(tensorboard_dir),
        verbose=verbose,
        **config.ppo.algorithm_kwargs(),
    )


def effective_hyperparameters(model: PPO) -> dict[str, Any]:
    """Read the hyperparameters actually used by a constructed PPO model."""
    values: dict[str, Any] = {
        name: getattr(model, name) for name in _RECORDED_ALGORITHM_ATTRIBUTES
    }
    clip_range = model.clip_range  # SB3 stores a schedule; ours is constant
    values["clip_range"] = float(
        clip_range(1.0) if callable(clip_range) else clip_range
    )
    values["stats_window_size"] = model._stats_window_size
    policy = model.policy
    values["policy"] = {
        "class": type(policy).__name__,
        "net_arch": policy.net_arch,
        "activation_fn": policy.activation_fn.__name__,
        "ortho_init": policy.ortho_init,
        "share_features_extractor": policy.share_features_extractor,
        "normalize_images": policy.normalize_images,
        "optimizer": policy.optimizer_class.__name__,
        "optimizer_kwargs": policy.optimizer_kwargs,
    }
    return values


def train(
    config: ExperimentConfig,
    runs_root: Path,
    *,
    argv: Sequence[str] | None = None,
    timestamp: str | None = None,
    verbose: int = 0,
) -> TrainingResult:
    """Train PPO per ``config`` in a new run directory under ``runs_root``."""
    launch_metadata = environment_metadata(argv)  # git revision as of launch
    run_dir = create_run_dir(
        runs_root, config.experiment_id, config.training.seed, timestamp
    )
    write_json(run_dir / CONFIG_FILENAME, config.to_dict())

    started_at = datetime.now(UTC)
    start = time.perf_counter()
    status = "failed"
    num_timesteps = 0
    hyperparameters: dict[str, Any] | None = None
    model_path = run_dir / MODEL_FILENAME
    env: VecEnv | None = None
    reward_logger = RewardComponentsLogger(
        run_dir / REWARD_COMPONENTS_FILENAME, config.reward
    )
    try:
        env = build_vec_env(config, run_dir / MONITOR_SUBDIR)
        model = build_model(config, env, run_dir / TENSORBOARD_SUBDIR, verbose=verbose)
        hyperparameters = effective_hyperparameters(model)
        callbacks: list[BaseCallback] = [reward_logger]
        if config.training.checkpoint_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    # The callback counts vectorised steps, each = n_envs timesteps;
                    # config validation guarantees an exact division.
                    save_freq=config.training.checkpoint_freq // config.training.n_envs,
                    save_path=str(run_dir / CHECKPOINTS_SUBDIR),
                    name_prefix=CHECKPOINT_PREFIX,
                )
            )
        model.learn(
            total_timesteps=config.training.total_timesteps,
            callback=callbacks,
            tb_log_name=TENSORBOARD_RUN_NAME,
        )
        model.save(model_path)
        num_timesteps = int(model.num_timesteps)
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        reward_logger.close()
        if env is not None:
            env.close()
        duration = time.perf_counter() - start
        run_metadata: dict[str, Any] = {
            "experiment_id": config.experiment_id,
            "status": status,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "duration_seconds": round(duration, 3),
            "requested_timesteps": config.training.total_timesteps,
            "num_timesteps": num_timesteps,
            "training_seed": config.training.seed,
            "env_seeds": [
                config.training.seed + i for i in range(config.training.n_envs)
            ],
            "torch_num_threads": torch.get_num_threads(),
            "effective_hyperparameters": hyperparameters,
            "model_path": MODEL_FILENAME if status == "completed" else None,
            **launch_metadata,
        }
        write_json(run_dir / METADATA_FILENAME, run_metadata)

    return TrainingResult(
        run_dir=run_dir,
        model_path=model_path,
        num_timesteps=num_timesteps,
        duration_seconds=duration,
    )
