"""Comparison pipeline: run selection, raw-data validation and write-once tables.

Uses a copy of the shaped smoke run (potential shaping, 512 steps), so the
training table has shaped components that must stay out of evaluation.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from lunar_lander_rl.artifacts import read_json
from lunar_lander_rl.comparison import (
    CONFIG_METRICS,
    EPISODE_METRIC_FIELDS,
    ComparisonError,
    compare,
    select_runs,
)
from lunar_lander_rl.config import (
    HoverConfig,
    MetricsConfig,
    RunSelectionConfig,
)
from lunar_lander_rl.evaluate import TRAJECTORIES_FILENAME, evaluate_run
from lunar_lander_rl.metrics import OUTCOMES
from lunar_lander_rl.train import TrainingResult

HOVER = HoverConfig(min_height=0.1, max_speed=0.2, min_consecutive_steps=5)
MISSING_SEED = 999


@pytest.fixture
def runs_root(shaped_smoke_run: TrainingResult, tmp_path: Path) -> Path:
    """A runs root holding one evaluated copy of the shaped smoke run."""
    source = shaped_smoke_run.run_dir
    root = tmp_path / "runs"
    target = root / source.parent.name / source.name
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("evaluations", "videos")
    )
    evaluate_run(target, episodes=2, timestamp="20000101T000001Z")
    return root


def _config(run_dir: Path) -> MetricsConfig:
    meta = read_json(run_dir / "metadata.json")
    return MetricsConfig(
        run_selection=RunSelectionConfig(
            source_commit=meta["source"]["commit"][:7],
            experiments=(meta["experiment_id"],),
            seeds=(meta["training_seed"], MISSING_SEED),
        ),
        hover=HOVER,
        training_window_start=0,
    )


def _only_run(root: Path) -> Path:
    (run_dir,) = [p.parent for p in root.glob("*/*/metadata.json")]
    return run_dir


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_compare_writes_validated_tables(runs_root: Path, tmp_path: Path) -> None:
    run_dir = _only_run(runs_root)
    config = _config(run_dir)
    out = tmp_path / "out"
    result = compare(config, runs_root, out)

    episodes = _csv(out / "episode_metrics.csv")
    assert tuple(episodes[0]) == EPISODE_METRIC_FIELDS
    assert len(episodes) == 2
    assert all(row["outcome"] in OUTCOMES for row in episodes)
    evaluation = _csv(run_dir / "evaluations" / "20000101T000001Z" / "episodes.csv")
    assert [r["original_return"] for r in episodes] == [
        r["original_return"] for r in evaluation
    ]

    (run_row,) = _csv(out / "run_summary.csv")
    assert int(run_row["n_episodes"]) == 2
    assert sum(int(run_row[f"n_{o}"]) for o in OUTCOMES) == 2

    config_rows = _csv(out / "config_summary.csv")
    assert [r["metric"] for r in config_rows] == list(CONFIG_METRICS)
    assert all(r["n_seeds"] in ("0", "1") for r in config_rows)

    (training,) = _csv(out / "training_vs_evaluation.csv")
    components = _csv(run_dir / "reward_components.csv")
    shaped = [float(r["total"]) for r in components]
    assert float(training["training_mean_shaped_total"]) == pytest.approx(
        sum(shaped) / len(shaped)
    )
    assert float(training["evaluation_mean_original"]) == pytest.approx(
        float(run_row["mean_return"])
    )

    assert result["missing_runs"] == [
        {
            "experiment_id": config.run_selection.experiments[0],
            "training_seed": MISSING_SEED,
        }
    ]
    saved = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert saved["evaluation_reward"] == "original environment reward (no wrappers)"
    assert saved["metrics_config"]["hover"]["min_consecutive_steps"] == 5
    assert (
        saved["pooled_counts"][config.run_selection.experiments[0]]["n_episodes"] == 2
    )

    with pytest.raises(FileExistsError):
        compare(config, runs_root, out)


def test_tampered_trajectory_is_rejected(runs_root: Path, tmp_path: Path) -> None:
    run_dir = _only_run(runs_root)
    good = run_dir / "evaluations" / "20000101T000001Z"
    bad = run_dir / "evaluations" / "20000101T000002Z"  # newer, so selected
    shutil.copytree(good, bad)
    with np.load(bad / TRAJECTORIES_FILENAME) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["rewards"][0] += 1.0
    (bad / TRAJECTORIES_FILENAME).unlink()
    np.savez_compressed(bad / TRAJECTORIES_FILENAME, **arrays)
    with pytest.raises(ComparisonError, match="does not match"):
        compare(_config(run_dir), runs_root, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_non_original_evaluation_is_rejected(runs_root: Path, tmp_path: Path) -> None:
    run_dir = _only_run(runs_root)
    summary_path = run_dir / "evaluations" / "20000101T000001Z" / "summary.json"
    summary = read_json(summary_path)
    summary["reward"] = "shaped"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ComparisonError, match="original reward"):
        compare(_config(run_dir), runs_root, tmp_path / "out")


def test_duplicate_runs_are_rejected(runs_root: Path) -> None:
    run_dir = _only_run(runs_root)
    shutil.copytree(run_dir, run_dir.with_name(run_dir.name + "-copy"))
    with pytest.raises(ComparisonError, match="two completed runs"):
        select_runs(runs_root, _config(run_dir).run_selection)


def test_runs_are_selected_by_metadata_not_folder(runs_root: Path) -> None:
    run_dir = _only_run(runs_root)
    config = _config(run_dir)
    # An interrupted run: same folder pattern, no metadata.json.
    partial = run_dir.with_name(
        "20990101T000000Z_seed" + str(config.run_selection.seeds[0])
    )
    partial.mkdir()
    selected, _ = select_runs(runs_root, config.run_selection)
    assert [r.run_dir for r in selected] == [run_dir]
    # Another commit is not selected.
    other = RunSelectionConfig("0000000", config.run_selection.experiments, (1,))
    assert select_runs(runs_root, other)[0] == []


def test_run_without_trajectories_is_rejected(runs_root: Path) -> None:
    run_dir = _only_run(runs_root)
    shutil.rmtree(run_dir / "evaluations")
    with pytest.raises(ComparisonError, match="no evaluation"):
        select_runs(runs_root, _config(run_dir).run_selection)
