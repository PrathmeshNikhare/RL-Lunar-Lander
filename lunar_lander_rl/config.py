"""Experiment configuration: loading, strict validation and serialisation.

Every tunable value used by training, evaluation and video recording lives in
an :class:`ExperimentConfig`. Configs are loaded from YAML with
``yaml.safe_load`` and validated at startup:

- every section and key is required;
- unknown keys are rejected;
- types and ranges are checked.

Invalid input raises :class:`ConfigError`, which names the offending key.

The effective config (after command-line overrides) is saved with each run as
JSON by ``lunar_lander_rl.artifacts``. It can be reloaded with
:meth:`ExperimentConfig.from_dict`.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from lunar_lander_rl.environment import ENV_ID

SUPPORTED_REWARD_MODES: Final[tuple[str, ...]] = (
    "original",
    "time_penalty",
    "upright_slow_bonus",
    "potential_based",
)
SUPPORTED_POLICIES: Final[tuple[str, ...]] = ("MlpPolicy",)
SUPPORTED_DEVICES: Final[tuple[str, ...]] = ("cpu",)  # no GPU dependency (CLAUDE.md)
_EXPERIMENT_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ConfigError(ValueError):
    """Raised when a configuration is missing keys or has invalid values."""


@dataclass(frozen=True)
class EnvironmentConfig:
    id: str


# Reward configurations, one per mode. Formulas, units and terminal handling
# are documented in docs/reward-wrapper.md. ``mode`` is first so that the
# saved JSON reads naturally; it is fixed per class, not an __init__ argument.


@dataclass(frozen=True)
class OriginalReward:
    """The unmodified environment reward (no custom components)."""

    mode: Literal["original"] = field(default="original", init=False)


@dataclass(frozen=True)
class TimePenaltyReward:
    """Subtract ``coefficient`` on every step."""

    mode: Literal["time_penalty"] = field(default="time_penalty", init=False)
    coefficient: float


@dataclass(frozen=True)
class UprightSlowBonusReward:
    """Add ``bonus`` on every step whose next observation is upright and slow.

    ``max_abs_angle`` is in radians (angle wrapped to [-pi, pi]); ``max_speed``
    is the norm of (vx, vy) in scaled observation units.
    """

    mode: Literal["upright_slow_bonus"] = field(
        default="upright_slow_bonus", init=False
    )
    bonus: float
    max_abs_angle: float
    max_speed: float


@dataclass(frozen=True)
class PotentialShapingReward:
    """Potential-based shaping ``alpha * (gamma * Phi(s') - Phi(s))``.

    ``gamma`` is not configured here: it is always ``ppo.gamma`` (D-019).
    """

    mode: Literal["potential_based"] = field(default="potential_based", init=False)
    alpha: float
    distance_weight: float
    speed_weight: float
    angle_weight: float


RewardConfig = (
    OriginalReward | TimePenaltyReward | UprightSlowBonusReward | PotentialShapingReward
)


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int
    seed: int
    n_envs: int
    checkpoint_freq: int  # in total environment timesteps; 0 disables checkpoints
    device: str


@dataclass(frozen=True)
class PPOConfig:
    policy: str
    net_arch: tuple[int, ...]
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float

    def algorithm_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``stable_baselines3.PPO`` (excluding policy/net_arch)."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass(frozen=True)
class EpisodeSetConfig:
    """Settings for a set of policy episodes (evaluation or video)."""

    episodes: int
    deterministic: bool
    seed_start: int  # episode i uses seed_start + i


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    environment: EnvironmentConfig
    reward: RewardConfig
    training: TrainingConfig
    ppo: PPOConfig
    evaluation: EpisodeSetConfig
    video: EpisodeSetConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serialisable representation."""
        data = dataclasses.asdict(self)
        data["ppo"]["net_arch"] = list(self.ppo.net_arch)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> ExperimentConfig:
        """Validate a plain mapping (e.g. parsed YAML/JSON) and build a config."""
        return _parse_experiment(data)

    def with_overrides(
        self, *, total_timesteps: int | None = None, seed: int | None = None
    ) -> ExperimentConfig:
        """Return a re-validated copy with command-line overrides applied."""
        data = self.to_dict()
        if total_timesteps is not None:
            data["training"]["total_timesteps"] = total_timesteps
        if seed is not None:
            data["training"]["seed"] = seed
        return _parse_experiment(data)


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration file."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return _parse_experiment(data)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _mapping(value: Any, where: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - set(keys))
    missing = [key for key in keys if key not in value]
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    if missing:
        raise ConfigError(f"{where}: missing key(s) {missing}")
    return value


def _int(data: Mapping[str, Any], key: str, where: str, *, minimum: int) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key}: expected an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{where}.{key}: must be >= {minimum}, got {value}")
    return value


def _float(
    data: Mapping[str, Any],
    key: str,
    where: str,
    *,
    greater_than: float | None = None,
    at_least: float | None = None,
    at_most: float | None = None,
) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):  # YAML accepts .nan / .inf
        raise ConfigError(f"{where}.{key}: must be finite, got {number}")
    if greater_than is not None and not number > greater_than:
        raise ConfigError(f"{where}.{key}: must be > {greater_than}, got {number}")
    if at_least is not None and number < at_least:
        raise ConfigError(f"{where}.{key}: must be >= {at_least}, got {number}")
    if at_most is not None and number > at_most:
        raise ConfigError(f"{where}.{key}: must be <= {at_most}, got {number}")
    return number


def _bool(data: Mapping[str, Any], key: str, where: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{where}.{key}: expected true/false, got {value!r}")
    return value


def _choice(
    data: Mapping[str, Any], key: str, where: str, choices: tuple[str, ...]
) -> str:
    value = data[key]
    if value not in choices:
        raise ConfigError(
            f"{where}.{key}: must be one of {list(choices)}, got {value!r}"
        )
    return str(value)


def _net_arch(data: Mapping[str, Any], where: str) -> tuple[int, ...]:
    value = data["net_arch"]
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value)
    ):
        raise ConfigError(
            f"{where}.net_arch: expected a non-empty list of positive integers, got {value!r}"
        )
    return tuple(value)


def parse_episode_set(value: Any, where: str) -> EpisodeSetConfig:
    """Validate one ``evaluation``/``video`` section on its own.

    Evaluation and video tools load only this section from a run's saved
    ``config.json``, so later schema changes to other sections (for example
    Phase 03 reward parameters) do not make earlier runs unevaluable.
    """
    data = _mapping(value, where, ("episodes", "deterministic", "seed_start"))
    return EpisodeSetConfig(
        episodes=_int(data, "episodes", where, minimum=1),
        deterministic=_bool(data, "deterministic", where),
        seed_start=_int(data, "seed_start", where, minimum=0),
    )


def _parse_reward(value: Any) -> RewardConfig:
    """Validate the ``reward`` section; each mode has its own required keys."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"reward: expected a mapping, got {type(value).__name__}")
    if "mode" not in value:
        raise ConfigError("reward: missing key(s) ['mode']")
    mode = _choice(value, "mode", "reward", SUPPORTED_REWARD_MODES)
    where = "reward"
    if mode == "original":
        _mapping(value, where, ("mode",))
        return OriginalReward()
    if mode == "time_penalty":
        data = _mapping(value, where, ("mode", "coefficient"))
        return TimePenaltyReward(
            coefficient=_float(data, "coefficient", where, greater_than=0.0)
        )
    if mode == "upright_slow_bonus":
        data = _mapping(value, where, ("mode", "bonus", "max_abs_angle", "max_speed"))
        return UprightSlowBonusReward(
            bonus=_float(data, "bonus", where, greater_than=0.0),
            # The angle is wrapped to [-pi, pi], so larger limits are meaningless.
            max_abs_angle=_float(
                data, "max_abs_angle", where, greater_than=0.0, at_most=math.pi
            ),
            max_speed=_float(data, "max_speed", where, greater_than=0.0),
        )
    if mode != "potential_based":  # SUPPORTED_REWARD_MODES and this parser drifted
        raise AssertionError(f"unhandled reward mode {mode!r}")
    data = _mapping(
        value,
        where,
        ("mode", "alpha", "distance_weight", "speed_weight", "angle_weight"),
    )
    weights = {
        key: _float(data, key, where, at_least=0.0)
        for key in ("distance_weight", "speed_weight", "angle_weight")
    }
    if not any(weights.values()):
        raise ConfigError(
            f"{where}: at least one potential weight must be > 0, got {weights}"
        )
    return PotentialShapingReward(
        alpha=_float(data, "alpha", where, greater_than=0.0), **weights
    )


def _parse_experiment(value: Any) -> ExperimentConfig:
    root = _mapping(
        value,
        "config",
        (
            "experiment_id",
            "environment",
            "reward",
            "training",
            "ppo",
            "evaluation",
            "video",
        ),
    )

    experiment_id = root["experiment_id"]
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID_PATTERN.match(
        experiment_id
    ):
        raise ConfigError(
            "config.experiment_id: must match "
            f"{_EXPERIMENT_ID_PATTERN.pattern} (used as a directory name), "
            f"got {experiment_id!r}"
        )

    env_data = _mapping(root["environment"], "environment", ("id",))
    environment = EnvironmentConfig(
        id=_choice(env_data, "id", "environment", (ENV_ID,))
    )

    reward = _parse_reward(root["reward"])

    t = _mapping(
        root["training"],
        "training",
        ("total_timesteps", "seed", "n_envs", "checkpoint_freq", "device"),
    )
    training = TrainingConfig(
        total_timesteps=_int(t, "total_timesteps", "training", minimum=1),
        seed=_int(t, "seed", "training", minimum=0),
        n_envs=_int(t, "n_envs", "training", minimum=1),
        checkpoint_freq=_int(t, "checkpoint_freq", "training", minimum=0),
        device=_choice(t, "device", "training", SUPPORTED_DEVICES),
    )

    p = _mapping(
        root["ppo"],
        "ppo",
        (
            "policy",
            "net_arch",
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        ),
    )
    ppo = PPOConfig(
        policy=_choice(p, "policy", "ppo", SUPPORTED_POLICIES),
        net_arch=_net_arch(p, "ppo"),
        learning_rate=_float(p, "learning_rate", "ppo", greater_than=0.0),
        n_steps=_int(p, "n_steps", "ppo", minimum=2),
        batch_size=_int(p, "batch_size", "ppo", minimum=2),
        n_epochs=_int(p, "n_epochs", "ppo", minimum=1),
        gamma=_float(p, "gamma", "ppo", greater_than=0.0, at_most=1.0),
        gae_lambda=_float(p, "gae_lambda", "ppo", at_least=0.0, at_most=1.0),
        clip_range=_float(p, "clip_range", "ppo", greater_than=0.0),
        ent_coef=_float(p, "ent_coef", "ppo", at_least=0.0),
        vf_coef=_float(p, "vf_coef", "ppo", at_least=0.0),
        max_grad_norm=_float(p, "max_grad_norm", "ppo", greater_than=0.0),
    )

    rollout_size = ppo.n_steps * training.n_envs
    if rollout_size % ppo.batch_size != 0:
        raise ConfigError(
            f"ppo.batch_size ({ppo.batch_size}) must divide n_steps * n_envs "
            f"({ppo.n_steps} * {training.n_envs} = {rollout_size})"
        )
    if training.checkpoint_freq % training.n_envs != 0:
        raise ConfigError(
            f"training.checkpoint_freq ({training.checkpoint_freq}) must be a multiple "
            f"of training.n_envs ({training.n_envs})"
        )

    evaluation = parse_episode_set(root["evaluation"], "evaluation")
    video = parse_episode_set(root["video"], "video")
    _check_seed_ranges_disjoint(training, {"evaluation": evaluation, "video": video})

    return ExperimentConfig(
        experiment_id=experiment_id,
        environment=environment,
        reward=reward,
        training=training,
        ppo=ppo,
        evaluation=evaluation,
        video=video,
    )


def _check_seed_ranges_disjoint(
    training: TrainingConfig, episode_sets: dict[str, EpisodeSetConfig]
) -> None:
    """Reject configs whose evaluation/video seeds overlap training env seeds."""
    ranges = {"training": range(training.seed, training.seed + training.n_envs)}
    for name, settings in episode_sets.items():
        ranges[name] = range(
            settings.seed_start, settings.seed_start + settings.episodes
        )
    names = list(ranges)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = ranges[first], ranges[second]
            if a.start < b.stop and b.start < a.stop:
                raise ConfigError(
                    f"seed ranges overlap: {first} [{a.start}, {a.stop}) and "
                    f"{second} [{b.start}, {b.stop})"
                )


# --------------------------------------------------------------------------- #
# Phase 05 comparison / metrics configuration (configs/evaluation_metrics.yaml)
# --------------------------------------------------------------------------- #

_COMMIT_PREFIX_PATTERN: Final = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class RunSelectionConfig:
    """Which completed runs are compared: one per (experiment, seed)."""

    source_commit: str
    experiments: tuple[str, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class HoverConfig:
    """Hover diagnostic thresholds (scaled observation units; docs/07-metrics.md)."""

    min_height: float
    max_speed: float
    min_consecutive_steps: int


@dataclass(frozen=True)
class MetricsConfig:
    run_selection: RunSelectionConfig
    hover: HoverConfig
    training_window_start: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_metrics_config(path: Path) -> MetricsConfig:
    """Load and validate a YAML metrics/comparison configuration file."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return parse_metrics_config(data)


def _unique_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: expected a non-empty list, got {value!r}")
    if len(set(map(repr, value))) != len(value):
        raise ConfigError(f"{where}: duplicate entries in {value!r}")
    return value


def parse_metrics_config(value: Any) -> MetricsConfig:
    """Validate a metrics configuration mapping."""
    top = _mapping(value, "metrics", ("run_selection", "hover", "training_window"))
    where = "run_selection"
    sel = _mapping(
        top["run_selection"], where, ("source_commit", "experiments", "seeds")
    )
    commit = sel["source_commit"]
    if not isinstance(commit, str) or not _COMMIT_PREFIX_PATTERN.match(commit):
        raise ConfigError(
            f"{where}.source_commit: expected 7-40 lowercase hex digits, got {commit!r}"
        )
    experiments = _unique_list(sel["experiments"], f"{where}.experiments")
    for name in experiments:
        if not isinstance(name, str) or not _EXPERIMENT_ID_PATTERN.match(name):
            raise ConfigError(f"{where}.experiments: invalid experiment id {name!r}")
    seeds = _unique_list(sel["seeds"], f"{where}.seeds")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigError(
                f"{where}.seeds: expected non-negative integers, got {seed!r}"
            )
    hover = _mapping(
        top["hover"], "hover", ("min_height", "max_speed", "min_consecutive_steps")
    )
    window = _mapping(top["training_window"], "training_window", ("start_timestep",))
    return MetricsConfig(
        run_selection=RunSelectionConfig(
            source_commit=commit,
            experiments=tuple(experiments),
            seeds=tuple(seeds),
        ),
        hover=HoverConfig(
            min_height=_float(hover, "min_height", "hover", at_least=0.0),
            max_speed=_float(hover, "max_speed", "hover", greater_than=0.0),
            min_consecutive_steps=_int(
                hover, "min_consecutive_steps", "hover", minimum=1
            ),
        ),
        training_window_start=_int(
            window, "start_timestep", "training_window", minimum=0
        ),
    )


# --------------------------------------------------------------------------- #
# Phase 06 visualization configuration (configs/visualization.yaml)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VisualizationConfig:
    """Inputs and fixed choices for Phase 06 figures, traces and videos.

    Paths are as written in the YAML file (relative to the project root when
    the scripts are run from there).
    """

    phase04_results: Path
    phase05_results: Path
    component_traces: Path
    rolling_episodes: int
    trace_episode: int
    video_episode: int
    dpi: int

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        for key in ("phase04_results", "phase05_results", "component_traces"):
            data[key] = Path(data[key]).as_posix()
        return data


def load_visualization_config(path: Path) -> VisualizationConfig:
    """Load and validate a YAML visualization configuration file."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return parse_visualization_config(data)


def _relative_path(data: Mapping[str, Any], key: str, where: str) -> Path:
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}.{key}: expected a non-empty path, got {value!r}")
    path = Path(value)
    # Relative paths only: no machine-specific absolute paths in configs (CLAUDE.md).
    if path.is_absolute() or path.drive or value.startswith(("/", "\\")):
        raise ConfigError(f"{where}.{key}: must be a relative path, got {value!r}")
    return path


def parse_visualization_config(value: Any) -> VisualizationConfig:
    """Validate a visualization configuration mapping."""
    top = _mapping(
        value, "visualization", ("inputs", "training_curve", "selection", "figure")
    )
    inputs = _mapping(
        top["inputs"],
        "inputs",
        ("phase04_results", "phase05_results", "component_traces"),
    )
    curve = _mapping(top["training_curve"], "training_curve", ("rolling_episodes",))
    selection = _mapping(
        top["selection"], "selection", ("trace_episode", "video_episode")
    )
    figure = _mapping(top["figure"], "figure", ("dpi",))
    return VisualizationConfig(
        phase04_results=_relative_path(inputs, "phase04_results", "inputs"),
        phase05_results=_relative_path(inputs, "phase05_results", "inputs"),
        component_traces=_relative_path(inputs, "component_traces", "inputs"),
        rolling_episodes=_int(curve, "rolling_episodes", "training_curve", minimum=1),
        trace_episode=_int(selection, "trace_episode", "selection", minimum=0),
        video_episode=_int(selection, "video_episode", "selection", minimum=0),
        dpi=_int(figure, "dpi", "figure", minimum=50),
    )
