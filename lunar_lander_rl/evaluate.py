"""Independent policy evaluation on the original LunarLander-v3 reward.

Evaluation never uses the training environment or any reward wrapper. It
builds a fresh environment with ``make_env()``, which has only Gymnasium's
standard ``make`` wrappers, so every reported return is the sum of the
environment's own rewards (decision D-009).

Per-episode output is raw and descriptive:

- return, length, ``terminated`` / ``truncated``;
- the final step's reward;
- the final position and leg contacts.

``evaluate_run`` also saves every step (observations, actions, original
rewards) to ``trajectories.npz`` in the session directory, so diagnostics
such as hovering are derived from raw data (decision D-038).

Success, crash and hover labels are *not* assigned here. They are derived
from these raw records by ``lunar_lander_rl.metrics`` (docs/07-metrics.md).

Only the relevant ``evaluation``/``video`` section of a run's saved
``config.json`` is read (decision D-020), so runs stay evaluable when other
config sections change in later phases.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from lunar_lander_rl.artifacts import (
    CONFIG_FILENAME,
    EVALUATIONS_SUBDIR,
    MODEL_FILENAME,
    create_session_dir,
    environment_metadata,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from lunar_lander_rl.config import EpisodeSetConfig, parse_episode_set
from lunar_lander_rl.environment import (
    ENV_ID,
    OBSERVATION_LABELS,
    LunarLanderEnv,
    Observation,
    make_env,
)

PolicyFn = Callable[[Observation], int]
EnvFactory = Callable[[], LunarLanderEnv]
EpisodeSection = Literal["evaluation", "video"]

EPISODES_FILENAME: Final[str] = "episodes.csv"
SUMMARY_FILENAME: Final[str] = "summary.json"
TRAJECTORIES_FILENAME: Final[str] = "trajectories.npz"
# Registered reward_threshold for LunarLander-v3 (asserted against the spec in tests).
REWARD_THRESHOLD: Final[float] = 200.0
# LunarLander assigns exactly +/-100 on terminal steps (docs/environment-notes.md).
TERMINAL_REWARD_MAGNITUDE: Final[float] = 100.0

_X = OBSERVATION_LABELS.index("x")
_Y = OBSERVATION_LABELS.index("y")
_LEFT_LEG = OBSERVATION_LABELS.index("left_leg_contact")
_RIGHT_LEG = OBSERVATION_LABELS.index("right_leg_contact")


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    seed: int
    original_return: float
    length: int
    terminated: bool
    truncated: bool
    final_reward: float
    final_x: float
    final_y: float
    final_left_leg_contact: float
    final_right_leg_contact: float


EPISODE_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in fields(EpisodeResult))


@dataclass(frozen=True)
class EpisodeSteps:
    """Every step of one episode.

    ``observations[0]`` is the observation returned by ``reset``;
    ``observations[t]`` follows action ``actions[t - 1]``, which earned the
    original reward ``rewards[t - 1]``.
    """

    observations: np.ndarray  # (length + 1, 8) float32
    actions: np.ndarray  # (length,) int8
    rewards: np.ndarray  # (length,) float64


def run_episodes(
    policy: PolicyFn,
    episodes: int,
    seed_start: int,
    env_factory: EnvFactory = make_env,
    step_log: list[EpisodeSteps] | None = None,
) -> list[EpisodeResult]:
    """Run ``episodes`` episodes with ``policy``; episode ``i`` uses ``seed_start + i``.

    If ``step_log`` is given, one :class:`EpisodeSteps` per episode is
    appended to it. Recording does not change the rollout.

    The environment's ``TimeLimit`` guarantees termination. A hard outer
    bound of ``max_episode_steps`` raises ``RuntimeError`` if it is ever
    exceeded, instead of looping forever.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if seed_start < 0:
        raise ValueError(f"seed_start must be >= 0, got {seed_start}")
    env = env_factory()
    try:
        if env.spec is None or env.spec.max_episode_steps is None:
            raise RuntimeError(
                "evaluation environment must have a registered time limit"
            )
        step_bound = env.spec.max_episode_steps
        results: list[EpisodeResult] = []
        for index in range(episodes):
            seed = seed_start + index
            observation, _ = env.reset(seed=seed)
            total = 0.0
            reward = 0.0
            terminated = truncated = False
            length = 0
            observations = [observation]
            actions: list[int] = []
            rewards: list[float] = []
            while not (terminated or truncated):
                if length >= step_bound:
                    raise RuntimeError(
                        f"episode exceeded {step_bound} steps without ending"
                    )
                action = policy(observation)
                observation, raw_reward, terminated, truncated, _ = env.step(
                    np.int64(action)
                )
                reward = float(raw_reward)
                total += reward
                length += 1
                if step_log is not None:
                    observations.append(observation)
                    actions.append(action)
                    rewards.append(reward)
            if step_log is not None:
                step_log.append(
                    EpisodeSteps(
                        observations=np.asarray(observations, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.int8),
                        rewards=np.asarray(rewards, dtype=np.float64),
                    )
                )
            results.append(
                EpisodeResult(
                    episode=index,
                    seed=seed,
                    original_return=total,
                    length=length,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    final_reward=reward,
                    final_x=float(observation[_X]),
                    final_y=float(observation[_Y]),
                    final_left_leg_contact=float(observation[_LEFT_LEG]),
                    final_right_leg_contact=float(observation[_RIGHT_LEG]),
                )
            )
        return results
    finally:
        env.close()


def write_trajectories(
    path: Path, seeds: Sequence[int], steps: Sequence[EpisodeSteps]
) -> None:
    """Write episodes' steps to a new compressed ``.npz``; fail if it exists.

    Episodes are concatenated; ``lengths`` (steps per episode) splits them.
    """
    if len(seeds) != len(steps):
        raise ValueError(f"{len(seeds)} seeds for {len(steps)} episodes")
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            seeds=np.asarray(seeds, dtype=np.int64),
            lengths=np.asarray([len(s.actions) for s in steps], dtype=np.int64),
            observations=np.concatenate([s.observations for s in steps]),
            actions=np.concatenate([s.actions for s in steps]),
            rewards=np.concatenate([s.rewards for s in steps]),
        )


def read_trajectories(path: Path) -> tuple[list[int], list[EpisodeSteps]]:
    """Read a file written by :func:`write_trajectories`."""
    with np.load(path) as data:
        seeds = [int(s) for s in data["seeds"]]
        lengths = [int(n) for n in data["lengths"]]
        observations, actions, rewards = (
            data["observations"],
            data["actions"],
            data["rewards"],
        )
    if (
        len(observations) != sum(lengths) + len(lengths)
        or len(actions) != sum(lengths)
        or len(rewards) != sum(lengths)
    ):
        raise ValueError(f"{path}: array sizes do not match the episode lengths")
    steps: list[EpisodeSteps] = []
    obs_at = step_at = 0
    for length in lengths:
        steps.append(
            EpisodeSteps(
                observations=observations[obs_at : obs_at + length + 1],
                actions=actions[step_at : step_at + length],
                rewards=rewards[step_at : step_at + length],
            )
        )
        obs_at += length + 1
        step_at += length
    return seeds, steps


def _std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _terminal_count(results: Sequence[EpisodeResult], value: float) -> int:
    return sum(r.terminated and math.isclose(r.final_reward, value) for r in results)


def summarize(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Descriptive statistics of evaluation episodes (sample std, ddof=1)."""
    if not results:
        raise ValueError("cannot summarise zero episodes")
    returns = [r.original_return for r in results]
    lengths = [float(r.length) for r in results]
    mean_return = statistics.fmean(returns)
    return {
        "n_episodes": len(results),
        "mean_return": mean_return,
        "std_return": _std(returns),
        "median_return": statistics.median(returns),
        "min_return": min(returns),
        "max_return": max(returns),
        "mean_length": statistics.fmean(lengths),
        "std_length": _std(lengths),
        "n_terminated": sum(r.terminated for r in results),
        "n_truncated": sum(r.truncated for r in results),
        "n_final_reward_plus_100": _terminal_count(results, TERMINAL_REWARD_MAGNITUDE),
        "n_final_reward_minus_100": _terminal_count(
            results, -TERMINAL_REWARD_MAGNITUDE
        ),
        "reward_threshold": REWARD_THRESHOLD,
        "n_return_at_least_threshold": sum(v >= REWARD_THRESHOLD for v in returns),
        "mean_return_at_least_threshold": mean_return >= REWARD_THRESHOLD,
    }


def model_policy(model_path: Path, deterministic: bool) -> PolicyFn:
    """Load a saved SB3 PPO model (CPU) and return an observation → action function."""
    from stable_baselines3 import PPO  # local import keeps pure helpers SB3-free

    model = PPO.load(model_path, device="cpu")

    def policy(observation: Observation) -> int:
        action, _ = model.predict(observation, deterministic=deterministic)
        return int(action)

    return policy


@dataclass(frozen=True)
class RunPolicy:
    """A run's saved policy plus the validated episode settings to use it with."""

    settings: EpisodeSetConfig
    policy: PolicyFn
    provenance: dict[str, Any]


def load_run_policy(
    run_dir: Path,
    section: EpisodeSection,
    *,
    model_filename: str = MODEL_FILENAME,
    episodes: int | None = None,
    seed_start: int | None = None,
) -> RunPolicy:
    """Load ``section`` settings (with overrides) and the model from ``run_dir``."""
    saved = read_json(run_dir / CONFIG_FILENAME)
    section_data = dict(saved[section])
    if episodes is not None:
        section_data["episodes"] = episodes
    if seed_start is not None:
        section_data["seed_start"] = seed_start
    settings = parse_episode_set(section_data, section)
    model_path = run_dir / model_filename
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    provenance = {
        "run_dir": run_dir.name,
        "experiment_id": saved.get("experiment_id"),
        "model": model_filename,
        "model_sha256": sha256_file(model_path),
    }
    return RunPolicy(
        settings=settings,
        policy=model_policy(model_path, settings.deterministic),
        provenance=provenance,
    )


def write_session(
    session_dir: Path,
    results: Sequence[EpisodeResult],
    settings: EpisodeSetConfig,
    extra: dict[str, Any],
) -> None:
    """Write ``episodes.csv`` and ``summary.json`` into a new session directory."""
    write_csv(
        session_dir / EPISODES_FILENAME, EPISODE_FIELDS, (asdict(r) for r in results)
    )
    write_json(
        session_dir / SUMMARY_FILENAME,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "reward": "original environment reward (no wrappers)",
            "env_id": ENV_ID,
            "settings": asdict(settings),
            **extra,
            "summary": summarize(results),
            **environment_metadata(),
        },
    )


def evaluate_run(
    run_dir: Path,
    *,
    model_filename: str = MODEL_FILENAME,
    episodes: int | None = None,
    seed_start: int | None = None,
    timestamp: str | None = None,
) -> Path:
    """Evaluate a run's saved model and write results to ``run_dir/evaluations/<ts>/``."""
    loaded = load_run_policy(
        run_dir,
        "evaluation",
        model_filename=model_filename,
        episodes=episodes,
        seed_start=seed_start,
    )
    # Claim the output directory before the (long) evaluation runs.
    session_dir = create_session_dir(run_dir / EVALUATIONS_SUBDIR, timestamp)
    probe = make_env()
    try:
        max_episode_steps = probe.spec.max_episode_steps if probe.spec else None
    finally:
        probe.close()
    steps: list[EpisodeSteps] = []
    results = run_episodes(
        loaded.policy,
        loaded.settings.episodes,
        loaded.settings.seed_start,
        step_log=steps,
    )
    write_trajectories(
        session_dir / TRAJECTORIES_FILENAME, [r.seed for r in results], steps
    )
    write_session(
        session_dir,
        results,
        loaded.settings,
        {
            **loaded.provenance,
            "max_episode_steps": max_episode_steps,
            "render_mode": None,
        },
    )
    return session_dir
