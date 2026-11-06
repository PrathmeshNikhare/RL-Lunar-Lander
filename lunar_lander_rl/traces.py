"""Per-step reward traces for recorded evaluation episodes (Phase 06).

Evaluation runs on the original reward only (D-009). To show how a run's
*training* reward would have accumulated over the same evaluation episode,
the recorded observations are replayed through the unchanged
:class:`~lunar_lander_rl.reward_wrapper.RewardComponentsWrapper`. The replay
environment returns the recorded observations, original rewards and terminal
flags, so every term comes from the training code path itself; no reward
formula is re-implemented here.

Valid because each custom term depends only on the observations, the
``terminated`` flag, the reward configuration and ``ppo.gamma``
(docs/reward-wrapper.md), all of which are recorded.

The committed trace CSV is the input for the within-episode figure. Building
it needs the local, git-ignored ``trajectories.npz`` files (D-038).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import gymnasium as gym
import numpy as np

from lunar_lander_rl.artifacts import (
    CONFIG_FILENAME,
    EVALUATIONS_SUBDIR,
    environment_metadata,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from lunar_lander_rl.config import ExperimentConfig, RewardConfig, VisualizationConfig
from lunar_lander_rl.environment import OBSERVATION_LABELS, Observation, make_env
from lunar_lander_rl.evaluate import (
    TRAJECTORIES_FILENAME,
    EpisodeSteps,
    read_trajectories,
)
from lunar_lander_rl.reward_wrapper import (
    ORIGINAL_KEY,
    STEP_INFO_KEY,
    TOTAL_KEY,
    RewardComponentsWrapper,
    component_names,
)

COMPARISON_FILENAME: Final[str] = "comparison.json"
EPISODE_METRICS_FILENAME: Final[str] = "episode_metrics.csv"
RUN_CONFIG_COPY: Final[str] = "run_config.json"  # committed copy (D-018)
_N_ACTIONS: Final[int] = 4

TRACE_FIELDS: Final[tuple[str, ...]] = (
    "experiment_id",
    "training_seed",
    "step",
    "action",
    *OBSERVATION_LABELS,
    "original",
    "custom_component",
    "custom_value",
    "training_reward",
)


class TraceError(ValueError):
    """Raised when recorded data are inconsistent or a trace cannot be built."""


class ReplayEnv(gym.Env[Observation, np.int64]):
    """Replay one recorded episode: recorded observations, rewards and flags.

    ``step`` must receive the recorded actions in order; any other action is
    an error, so the replay cannot silently diverge from the recording.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        steps: EpisodeSteps,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not (terminated or truncated):
            raise TraceError("a recorded episode must end terminated or truncated")
        if len(steps.observations) != len(steps.actions) + 1 or len(
            steps.rewards
        ) != len(steps.actions):
            raise TraceError(
                "observations, actions and rewards have inconsistent lengths"
            )
        if len(steps.actions) == 0:
            raise TraceError("cannot replay an episode with no steps")
        self.observation_space = cast(gym.spaces.Space[Observation], observation_space)
        self.action_space = gym.spaces.Discrete(_N_ACTIONS)
        self._steps = steps
        self._terminated = terminated
        self._truncated = truncated
        self._t = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        observation: Observation = self._steps.observations[0]
        return observation, {}

    def step(
        self, action: np.int64
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        actions = self._steps.actions
        if self._t >= len(actions):
            raise TraceError("step() called after the recorded episode ended")
        if int(action) != int(actions[self._t]):
            raise TraceError(
                f"step {self._t + 1}: action {int(action)} differs from the "
                f"recorded action {int(actions[self._t])}"
            )
        reward = float(self._steps.rewards[self._t])
        self._t += 1
        last = self._t == len(actions)
        observation: Observation = self._steps.observations[self._t]
        return (
            observation,
            reward,
            self._terminated and last,
            self._truncated and last,
            {},
        )


def lunar_lander_observation_space() -> gym.spaces.Box:
    """The declared ``LunarLander-v3`` observation space (used for Phi clipping)."""
    env = make_env()
    try:
        space = env.observation_space
    finally:
        env.close()
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"expected a Box observation space, got {space}")
    return space


def replay_reward_terms(
    steps: EpisodeSteps,
    *,
    terminated: bool,
    truncated: bool,
    reward: RewardConfig,
    gamma: float,
    observation_space: gym.spaces.Box,
) -> list[dict[str, float]]:
    """Per-step ``info["reward_components"]`` of the wrapper over a recording."""
    env = ReplayEnv(
        observation_space, steps, terminated=terminated, truncated=truncated
    )
    wrapped = RewardComponentsWrapper(env, reward, gamma)
    wrapped.reset()
    terms: list[dict[str, float]] = []
    for action in steps.actions:
        _, _, _, _, info = wrapped.step(np.int64(action))
        terms.append(dict(info[STEP_INFO_KEY]))
    return terms


def _float32_text(value: float) -> str:
    """Shortest text that round-trips the recorded float32 value."""
    return str(np.float32(value))


def trace_rows(
    ident: Mapping[str, Any],
    steps: EpisodeSteps,
    terms: Sequence[Mapping[str, float]],
    component: str | None,
) -> list[dict[str, Any]]:
    """One row per step: the observation *after* the step and its reward terms.

    ``component`` is the run's custom reward component (``None`` for the
    original-reward baseline, written as empty cells).
    """
    if len(terms) != len(steps.actions):
        raise TraceError(f"{len(terms)} reward terms for {len(steps.actions)} steps")
    rows: list[dict[str, Any]] = []
    for step, (action, observation, term) in enumerate(
        zip(steps.actions, steps.observations[1:], terms, strict=True), start=1
    ):
        rows.append(
            {
                **ident,
                "step": step,
                "action": int(action),
                **{
                    label: _float32_text(float(observation[i]))
                    for i, label in enumerate(OBSERVATION_LABELS)
                },
                "original": term[ORIGINAL_KEY],
                "custom_component": component or "",
                "custom_value": term[component] if component else "",
                "training_reward": term[TOTAL_KEY],
            }
        )
    return rows


def _bool(value: str, where: str) -> bool:
    if value not in ("True", "False"):
        raise TraceError(f"{where}: expected True/False, got {value!r}")
    return value == "True"


def _single_component(reward: RewardConfig) -> str | None:
    names = component_names(reward)
    if len(names) > 1:
        raise TraceError(f"expected at most one custom component, got {names}")
    return names[0] if names else None


def extract_traces(
    config: VisualizationConfig, runs_root: Path, output_csv: Path
) -> dict[str, Any]:
    """Write the per-step traces of the selected evaluation episode of every run.

    Runs and evaluation sessions are exactly those listed in the Phase 05
    ``comparison.json``. Each trace is checked against Phase 05's
    ``episode_metrics.csv`` (env seed, length, flags, exact original return)
    and each run's ``config.json`` against its committed ``run_config.json``.
    Writes ``output_csv`` and a provenance file beside it (``.json``); both
    must not exist.
    """
    provenance = environment_metadata()  # before any output exists
    sidecar = output_csv.with_suffix(".json")
    for path in (output_csv, sidecar):
        if path.exists():
            raise FileExistsError(f"{path} already exists; nothing was overwritten")
    comparison_path = config.phase05_results / COMPARISON_FILENAME
    metrics_path = config.phase05_results / EPISODE_METRICS_FILENAME
    comparison = read_json(comparison_path)
    n_episodes = int(comparison["evaluation_settings"]["episodes"])
    index = config.trace_episode
    if index >= n_episodes:
        raise TraceError(f"trace_episode {index} >= evaluation episodes {n_episodes}")
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metrics = {
            (r["experiment_id"], int(r["training_seed"]), int(r["episode"])): r
            for r in csv.DictReader(handle)
        }

    space = lunar_lander_observation_space()
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for run in comparison["runs"]:
        experiment, seed = run["experiment_id"], int(run["training_seed"])
        run_dir = runs_root / experiment / run["run_dir"]
        session = run_dir / EVALUATIONS_SUBDIR / run["evaluation_session"]
        where = f"{experiment}/{run['run_dir']}/{run['evaluation_session']}"
        raw_config = read_json(run_dir / CONFIG_FILENAME)
        committed = read_json(
            config.phase04_results / experiment / f"seed{seed}" / RUN_CONFIG_COPY
        )
        if raw_config != committed:
            raise TraceError(f"{where}: config.json differs from the committed copy")
        experiment_config = ExperimentConfig.from_dict(raw_config)
        if experiment_config.training.seed != seed:
            raise TraceError(f"{where}: config training seed is not {seed}")
        trajectories = session / TRAJECTORIES_FILENAME
        env_seeds, episodes = read_trajectories(trajectories)
        steps, env_seed = episodes[index], env_seeds[index]
        record = metrics.get((experiment, seed, index))
        if record is None:
            raise TraceError(f"{where}: episode {index} missing from episode_metrics")
        terminated = _bool(record["terminated"], where)
        truncated = _bool(record["truncated"], where)
        if int(record["seed"]) != env_seed or int(record["length"]) != len(
            steps.actions
        ):
            raise TraceError(f"{where}: trajectory does not match episode_metrics")
        terms = replay_reward_terms(
            steps,
            terminated=terminated,
            truncated=truncated,
            reward=experiment_config.reward,
            gamma=experiment_config.ppo.gamma,
            observation_space=space,
        )
        original_total = 0.0
        for term, recorded in zip(terms, steps.rewards, strict=True):
            if term[ORIGINAL_KEY] != float(recorded):
                raise TraceError(f"{where}: replayed original reward differs")
            original_total += term[ORIGINAL_KEY]  # same order as evaluation
        if original_total != float(record["original_return"]):
            raise TraceError(f"{where}: original return differs from episode_metrics")
        component = _single_component(experiment_config.reward)
        ident = {
            "experiment_id": experiment,
            "training_seed": seed,
            "run_dir": run["run_dir"],
            "evaluation_session": run["evaluation_session"],
            "episode": index,
            "env_seed": env_seed,
        }
        # Rows carry only the run key; the episode identity is in the sidecar.
        key = {"experiment_id": experiment, "training_seed": seed}
        rows.extend(trace_rows(key, steps, terms, component))
        sources.append(
            {
                **ident,
                "trajectories_sha256": sha256_file(trajectories),
                "reward": experiment_config.to_dict()["reward"],
                "gamma": experiment_config.ppo.gamma,
                "length": len(steps.actions),
                "terminated": terminated,
                "truncated": truncated,
                "outcome": record["outcome"],
                "original_return": original_total,
                "custom_component": component,
                "custom_total": sum(t[component] for t in terms) if component else None,
                "training_reward_total": sum(t[TOTAL_KEY] for t in terms),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, TRACE_FIELDS, rows)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "description": (
            "Per-step original reward and training-reward terms for one fixed "
            "evaluation episode per run, replayed through RewardComponentsWrapper "
            "from the Phase 05 evaluation trajectories. Observations are the "
            "state after each step. Evaluation itself used the original reward. "
            "Rows are keyed by experiment_id and training_seed; each episode's run "
            "directory, evaluation session, index and env seed are under episodes."
        ),
        "visualization_config": config.to_dict(),
        "inputs": {
            "comparison_json_sha256": sha256_file(comparison_path),
            "episode_metrics_sha256": sha256_file(metrics_path),
        },
        "output_csv_sha256": sha256_file(output_csv),
        "n_rows": len(rows),
        "episodes": sources,
        **provenance,
    }
    write_json(sidecar, result)
    return result
