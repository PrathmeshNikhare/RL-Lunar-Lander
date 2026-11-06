"""Per-step reward traces: replay parity with the training wrapper and extraction checks."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from lunar_lander_rl.artifacts import read_json
from lunar_lander_rl.comparison import compare
from lunar_lander_rl.config import (
    HoverConfig,
    MetricsConfig,
    OriginalReward,
    PotentialShapingReward,
    RewardConfig,
    RunSelectionConfig,
    TimePenaltyReward,
    UprightSlowBonusReward,
    VisualizationConfig,
)
from lunar_lander_rl.environment import make_env
from lunar_lander_rl.evaluate import EpisodeSteps, evaluate_run
from lunar_lander_rl.reward_functions import potential
from lunar_lander_rl.reward_wrapper import STEP_INFO_KEY, RewardComponentsWrapper
from lunar_lander_rl.traces import (
    TRACE_FIELDS,
    ReplayEnv,
    TraceError,
    extract_traces,
    lunar_lander_observation_space,
    replay_reward_terms,
    trace_rows,
)
from lunar_lander_rl.train import TrainingResult

GAMMA = 0.999
REWARDS: dict[str, RewardConfig] = {
    "original": OriginalReward(),
    "time_penalty": TimePenaltyReward(coefficient=0.05),
    "upright_slow_bonus": UprightSlowBonusReward(
        bonus=1.0, max_abs_angle=0.2, max_speed=0.2
    ),
    "potential_based": PotentialShapingReward(
        alpha=100.0, distance_weight=1.0, speed_weight=1.0, angle_weight=1.0
    ),
}
# A fixed action pattern: main engine bursts with side engines, then free fall.
ACTIONS = (2, 2, 1, 0, 3, 2, 0, 0)


def _record(
    reward: RewardConfig, max_episode_steps: int | None
) -> tuple[EpisodeSteps, bool, bool, list[dict[str, float]]]:
    """Run the real wrapped env with fixed actions; keep obs, rewards and terms."""
    env = RewardComponentsWrapper(
        make_env(max_episode_steps=max_episode_steps), reward, GAMMA
    )
    observation, _ = env.reset(seed=3)
    observations, actions, rewards, terms = [observation], [], [], []
    terminated = truncated = False
    t = 0
    while not (terminated or truncated):
        action = ACTIONS[t % len(ACTIONS)]
        observation, _, terminated, truncated, info = env.step(np.int64(action))
        components = info[STEP_INFO_KEY]
        observations.append(observation)
        actions.append(action)
        rewards.append(components["original"])
        terms.append(dict(components))
        t += 1
    env.close()
    steps = EpisodeSteps(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int8),
        rewards=np.array(rewards, dtype=np.float64),
    )
    return steps, bool(terminated), bool(truncated), terms


@pytest.mark.parametrize("mode", list(REWARDS))
@pytest.mark.parametrize("limit", [None, 25], ids=["terminated", "truncated"])
def test_replay_matches_wrapper_bit_for_bit(mode: str, limit: int | None) -> None:
    steps, terminated, truncated, recorded = _record(REWARDS[mode], limit)
    assert truncated == (limit is not None)
    replayed = replay_reward_terms(
        steps,
        terminated=terminated,
        truncated=truncated,
        reward=REWARDS[mode],
        gamma=GAMMA,
        observation_space=lunar_lander_observation_space(),
    )
    assert replayed == recorded  # exact equality, not approx


def _synthetic(length: int = 2) -> EpisodeSteps:
    observations = np.zeros((length + 1, 8), dtype=np.float32)
    observations[:, 0] = np.linspace(0.3, 0.1, length + 1)  # x
    observations[:, 1] = np.linspace(0.8, 0.4, length + 1)  # y
    observations[:, 3] = -0.1  # vy
    return EpisodeSteps(
        observations=observations,
        actions=np.zeros(length, dtype=np.int8),
        rewards=np.full(length, -0.5),
    )


@pytest.mark.parametrize("terminated", [True, False])
def test_shaping_terminal_handling(terminated: bool) -> None:
    reward = REWARDS["potential_based"]
    assert isinstance(reward, PotentialShapingReward)
    steps = _synthetic()
    terms = replay_reward_terms(
        steps,
        terminated=terminated,
        truncated=not terminated,
        reward=reward,
        gamma=GAMMA,
        observation_space=lunar_lander_observation_space(),
    )
    phi = [potential(o, 1.0, 1.0, 1.0) for o in steps.observations]
    assert terms[0]["potential_shaping"] == pytest.approx(
        100.0 * (GAMMA * phi[1] - phi[0])
    )
    next_phi = 0.0 if terminated else phi[2]  # Phi(s') = 0 only when terminated
    assert terms[1]["potential_shaping"] == pytest.approx(
        100.0 * (GAMMA * next_phi - phi[1])
    )


def test_replay_env_rejects_divergence() -> None:
    space = lunar_lander_observation_space()
    steps = _synthetic()
    with pytest.raises(TraceError, match="terminated or truncated"):
        ReplayEnv(space, steps, terminated=False, truncated=False)
    bad = EpisodeSteps(steps.observations[:-1], steps.actions, steps.rewards)
    with pytest.raises(TraceError, match="inconsistent"):
        ReplayEnv(space, bad, terminated=True, truncated=False)
    env = ReplayEnv(space, steps, terminated=True, truncated=False)
    env.reset()
    with pytest.raises(TraceError, match="differs from the recorded action"):
        env.step(np.int64(2))
    env.reset()
    env.step(np.int64(0))
    _, _, terminated, truncated, _ = env.step(np.int64(0))
    assert (terminated, truncated) == (True, False)
    with pytest.raises(TraceError, match="after the recorded episode ended"):
        env.step(np.int64(0))


def test_trace_rows_format() -> None:
    steps = _synthetic()
    terms = [{"original": -0.5, "total": -0.5}] * 2
    rows = trace_rows({"experiment_id": "b"}, steps, terms, None)
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["custom_component"] == "" and rows[0]["custom_value"] == ""
    # Observation after step 1, written so that float32 round-trips exactly.
    assert np.float32(rows[0]["y"]) == steps.observations[1, 1]
    with pytest.raises(TraceError):
        trace_rows({}, steps, terms[:1], None)


# --------------------------------------------------------------------------- #
# End-to-end extraction on an evaluated copy of the shaped smoke run
# --------------------------------------------------------------------------- #


@pytest.fixture
def layout(shaped_smoke_run: TrainingResult, tmp_path: Path) -> VisualizationConfig:
    source = shaped_smoke_run.run_dir
    runs_root = tmp_path / "runs"
    run_dir = runs_root / source.parent.name / source.name
    shutil.copytree(
        source, run_dir, ignore=shutil.ignore_patterns("evaluations", "videos")
    )
    evaluate_run(run_dir, episodes=2, timestamp="20000101T000001Z")
    meta = read_json(run_dir / "metadata.json")
    seed = meta["training_seed"]
    metrics = MetricsConfig(
        run_selection=RunSelectionConfig(
            source_commit=meta["source"]["commit"][:7],
            experiments=(meta["experiment_id"],),
            seeds=(seed,),
        ),
        hover=HoverConfig(min_height=0.1, max_speed=0.2, min_consecutive_steps=5),
        training_window_start=0,
    )
    compare(metrics, runs_root, tmp_path / "phase05")
    committed = tmp_path / "phase04" / meta["experiment_id"] / f"seed{seed}"
    committed.mkdir(parents=True)
    shutil.copy(run_dir / "config.json", committed / "run_config.json")
    return VisualizationConfig(
        phase04_results=tmp_path / "phase04",
        phase05_results=tmp_path / "phase05",
        component_traces=tmp_path / "phase06" / "component_traces.csv",
        rolling_episodes=1,
        trace_episode=1,
        video_episode=0,
        dpi=50,
    )


def test_extract_traces_writes_checked_rows(
    layout: VisualizationConfig, tmp_path: Path
) -> None:
    out = layout.component_traces
    result = extract_traces(layout, tmp_path / "runs", out)
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    (episode,) = result["episodes"]
    assert tuple(rows[0]) == TRACE_FIELDS
    assert len(rows) == episode["length"] == result["n_rows"]
    assert episode["episode"] == 1 and episode["env_seed"] == 10001
    assert {r["custom_component"] for r in rows} == {"potential_shaping"}
    with (layout.phase05_results / "episode_metrics.csv").open(encoding="utf-8") as f:
        metric = next(r for r in csv.DictReader(f) if r["episode"] == "1")
    assert sum(float(r["original"]) for r in rows) == pytest.approx(
        float(metric["original_return"])
    )
    for row in rows:
        assert float(row["training_reward"]) == pytest.approx(
            float(row["original"]) + float(row["custom_value"])
        )
    sidecar = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["n_rows"] == len(rows)
    with pytest.raises(FileExistsError):
        extract_traces(layout, tmp_path / "runs", out)


def test_extract_traces_rejects_changed_config(
    layout: VisualizationConfig, tmp_path: Path
) -> None:
    (committed,) = layout.phase04_results.glob("*/*/run_config.json")
    data = read_json(committed)
    data["ppo"]["gamma"] = 0.5
    committed.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TraceError, match="differs from the committed copy"):
        extract_traces(layout, tmp_path / "runs", layout.component_traces)
    assert not layout.component_traces.exists()


def test_extract_traces_rejects_out_of_range_episode(
    layout: VisualizationConfig, tmp_path: Path
) -> None:
    config = VisualizationConfig(**{**layout.__dict__, "trace_episode": 2})
    with pytest.raises(TraceError, match="trace_episode"):
        extract_traces(config, tmp_path / "runs", layout.component_traces)
