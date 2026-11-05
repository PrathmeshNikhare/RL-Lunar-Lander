"""Controlled comparison of evaluated runs on the original reward (Phase 05).

Pipeline:

1. select one completed run per (experiment, seed) by ``metadata.json``
   (status, source commit, training seed), never by folder name;
2. take each run's newest evaluation session that has ``trajectories.npz``;
3. validate the raw data: ``episodes.csv`` header, original-reward summary,
   identical evaluation settings across runs, and trajectories that agree
   with ``episodes.csv`` (seed, length, reward sum, final observation);
4. derive per-episode outcomes and diagnostics (``lunar_lander_rl.metrics``);
5. write new, write-once tables to an output directory that must not exist.

Training-time returns (``reward_components.csv``, which include custom
reward components) are reported in their own table and never mixed with
evaluation returns.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from lunar_lander_rl.artifacts import (
    EVALUATIONS_SUBDIR,
    METADATA_FILENAME,
    REWARD_COMPONENTS_FILENAME,
    environment_metadata,
    read_json,
    write_csv,
    write_json,
)
from lunar_lander_rl.config import HoverConfig, MetricsConfig, RunSelectionConfig
from lunar_lander_rl.environment import OBSERVATION_LABELS
from lunar_lander_rl.evaluate import (
    EPISODE_FIELDS,
    EPISODES_FILENAME,
    SUMMARY_FILENAME,
    TRAJECTORIES_FILENAME,
    read_trajectories,
)
from lunar_lander_rl.metrics import (
    OUTCOMES,
    across_seeds,
    classify_outcome,
    episode_diagnostics,
    run_summary,
)

_X = OBSERVATION_LABELS.index("x")
_Y = OBSERVATION_LABELS.index("y")
ORIGINAL_REWARD_LABEL: Final[str] = "original environment reward (no wrappers)"
EPISODE_METRICS_FILENAME: Final[str] = "episode_metrics.csv"
RUN_SUMMARY_FILENAME: Final[str] = "run_summary.csv"
CONFIG_SUMMARY_FILENAME: Final[str] = "config_summary.csv"
TRAINING_FILENAME: Final[str] = "training_vs_evaluation.csv"
COMPARISON_FILENAME: Final[str] = "comparison.json"

EPISODE_METRIC_FIELDS: Final[tuple[str, ...]] = (
    "experiment_id",
    "training_seed",
    "run_dir",
    "episode",
    "seed",
    "original_return",
    "length",
    "terminated",
    "truncated",
    "final_reward",
    "final_x",
    "final_y",
    "outcome",
    "first_contact_step",
    "max_hover_steps",
    "hover",
    "main_engine_steps",
    "side_engine_steps",
)
# Per-run metrics aggregated across seeds in config_summary.csv.
CONFIG_METRICS: Final[tuple[str, ...]] = (
    "mean_return",
    "mean_length",
    *(f"{o}_rate" for o in OUTCOMES),
    "hover_rate",
    "mean_first_contact_step",
    "mean_main_engine_steps",
    "mean_side_engine_steps",
)
TRAINING_FIELDS: Final[tuple[str, ...]] = (
    "experiment_id",
    "training_seed",
    "run_dir",
    "window_start_timestep",
    "n_training_episodes",
    "training_mean_original",
    "training_mean_shaped_total",
    "evaluation_n_episodes",
    "evaluation_mean_original",
)


class ComparisonError(ValueError):
    """Raised when runs or their evaluation data fail selection or validation."""


@dataclass(frozen=True)
class SelectedRun:
    experiment_id: str
    training_seed: int
    run_dir: Path
    evaluation_dir: Path


def select_runs(
    runs_root: Path, selection: RunSelectionConfig
) -> tuple[list[SelectedRun], list[dict[str, Any]]]:
    """Return the selected runs and the (experiment, seed) pairs with no run.

    A run qualifies if its ``metadata.json`` says ``status: completed``, its
    source commit starts with ``selection.source_commit`` and its training
    seed is selected. Two qualifying runs for one pair raise
    :class:`ComparisonError`.
    """
    found: dict[tuple[str, int], Path] = {}
    for experiment in selection.experiments:
        for meta_path in sorted(
            (runs_root / experiment).glob(f"*/{METADATA_FILENAME}")
        ):
            meta = read_json(meta_path)
            commit = (meta.get("source") or {}).get("commit") or ""
            seed = meta.get("training_seed")
            if (
                meta.get("status") != "completed"
                or not commit.startswith(selection.source_commit)
                or seed not in selection.seeds
            ):
                continue
            key = (experiment, int(seed))
            if key in found:
                raise ComparisonError(
                    f"two completed runs for {key}: {found[key].name}, {meta_path.parent.name}"
                )
            found[key] = meta_path.parent
    selected: list[SelectedRun] = []
    missing: list[dict[str, Any]] = []
    for experiment in selection.experiments:
        for seed in selection.seeds:
            run_dir = found.get((experiment, seed))
            if run_dir is None:
                missing.append({"experiment_id": experiment, "training_seed": seed})
            else:
                selected.append(
                    SelectedRun(experiment, seed, run_dir, latest_evaluation(run_dir))
                )
    return selected, missing


def latest_evaluation(run_dir: Path) -> Path:
    """The newest evaluation session of ``run_dir`` that has trajectories."""
    sessions = sorted(
        p.parent
        for p in (run_dir / EVALUATIONS_SUBDIR).glob(f"*/{TRAJECTORIES_FILENAME}")
    )
    if not sessions:
        raise ComparisonError(
            f"{run_dir.name}: no evaluation with {TRAJECTORIES_FILENAME}"
        )
    return sessions[-1]


def _bool(value: str, where: str) -> bool:
    if value not in ("True", "False"):
        raise ComparisonError(f"{where}: expected True/False, got {value!r}")
    return value == "True"


def load_episode_metrics(run: SelectedRun, hover: HoverConfig) -> list[dict[str, Any]]:
    """Validate one evaluation session and return its per-episode metric rows."""
    where = f"{run.run_dir.name}/{run.evaluation_dir.name}"
    summary = read_json(run.evaluation_dir / SUMMARY_FILENAME)
    if summary.get("reward") != ORIGINAL_REWARD_LABEL:
        raise ComparisonError(f"{where}: evaluation is not on the original reward")
    with (run.evaluation_dir / EPISODES_FILENAME).open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
            raise ComparisonError(
                f"{where}: unexpected episodes.csv columns {reader.fieldnames}"
            )
        records = list(reader)
    seeds, steps = read_trajectories(run.evaluation_dir / TRAJECTORIES_FILENAME)
    if len(records) != len(steps) or len(records) != summary["settings"]["episodes"]:
        raise ComparisonError(f"{where}: episode counts disagree")
    rows: list[dict[str, Any]] = []
    for record, seed, episode in zip(records, seeds, steps, strict=True):
        tag = f"{where} episode {record['episode']}"
        original_return = float(record["original_return"])
        length = int(record["length"])
        total = 0.0
        for reward in episode.rewards:  # same left-to-right order as evaluation
            total += float(reward)
        final = episode.observations[-1]
        if (
            int(record["seed"]) != seed
            or len(episode.actions) != length
            or total != original_return
            or float(record["final_reward"]) != float(episode.rewards[-1])
            or float(record["final_x"]) != float(final[_X])
            or float(record["final_y"]) != float(final[_Y])
        ):
            raise ComparisonError(f"{tag}: trajectory does not match episodes.csv")
        terminated = _bool(record["terminated"], tag)
        truncated = _bool(record["truncated"], tag)
        diagnostics = episode_diagnostics(episode.observations, episode.actions, hover)
        rows.append(
            {
                "experiment_id": run.experiment_id,
                "training_seed": run.training_seed,
                "run_dir": run.run_dir.name,
                "episode": int(record["episode"]),
                "seed": seed,
                "original_return": original_return,
                "length": length,
                "terminated": terminated,
                "truncated": truncated,
                "final_reward": float(record["final_reward"]),
                "final_x": float(record["final_x"]),
                "final_y": float(record["final_y"]),
                "outcome": classify_outcome(
                    terminated,
                    truncated,
                    float(record["final_reward"]),
                    float(record["final_x"]),
                ),
                **asdict(diagnostics),
            }
        )
    return rows


def training_summary(run: SelectedRun, window_start: int) -> dict[str, Any]:
    """Late-training original and shaped returns from ``reward_components.csv``."""
    path = run.run_dir / REWARD_COMPONENTS_FILENAME
    with path.open(encoding="utf-8", newline="") as handle:
        late = [
            r for r in csv.DictReader(handle) if int(r["timesteps"]) >= window_start
        ]
    if not late:
        raise ComparisonError(
            f"{run.run_dir.name}: no training episodes after {window_start}"
        )
    return {
        "window_start_timestep": window_start,
        "n_training_episodes": len(late),
        "training_mean_original": statistics.fmean(float(r["original"]) for r in late),
        "training_mean_shaped_total": statistics.fmean(float(r["total"]) for r in late),
    }


def compare(config: MetricsConfig, runs_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run the comparison and write all tables to the new ``output_dir``."""
    runs, missing = select_runs(runs_root, config.run_selection)
    if not runs:
        raise ComparisonError("no runs selected")
    settings = {
        r.run_dir.name: read_json(r.evaluation_dir / SUMMARY_FILENAME)["settings"]
        for r in runs
    }
    if len({repr(s) for s in settings.values()}) != 1:
        raise ComparisonError(f"evaluation settings differ between runs: {settings}")

    episode_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for run in runs:
        rows = load_episode_metrics(run, config.hover)
        episode_rows.extend(rows)
        summary = run_summary(rows)
        ident = {
            "experiment_id": run.experiment_id,
            "training_seed": run.training_seed,
            "run_dir": run.run_dir.name,
        }
        run_rows.append(
            {**ident, "evaluation_session": run.evaluation_dir.name, **summary}
        )
        training_rows.append(
            {
                **ident,
                **training_summary(run, config.training_window_start),
                "evaluation_n_episodes": summary["n_episodes"],
                "evaluation_mean_original": summary["mean_return"],
            }
        )

    config_rows: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, int]] = {}
    for experiment in config.run_selection.experiments:
        mine = [r for r in run_rows if r["experiment_id"] == experiment]
        if not mine:
            continue
        for metric in CONFIG_METRICS:
            config_rows.append(
                {
                    "experiment_id": experiment,
                    "metric": metric,
                    **across_seeds([r[metric] for r in mine]),
                }
            )
        pooled[experiment] = {
            "n_runs": len(mine),
            "n_episodes": sum(r["n_episodes"] for r in mine),
            **{f"n_{o}": sum(r[f"n_{o}"] for r in mine) for o in OUTCOMES},
            "n_hover": sum(r["n_hover"] for r in mine),
        }

    # Validate and compute everything first, so a failure leaves no output.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(
        output_dir / EPISODE_METRICS_FILENAME, EPISODE_METRIC_FIELDS, episode_rows
    )
    write_csv(output_dir / RUN_SUMMARY_FILENAME, list(run_rows[0]), run_rows)
    write_csv(
        output_dir / CONFIG_SUMMARY_FILENAME,
        ("experiment_id", "metric", "n_seeds", "mean", "std", "min", "max"),
        config_rows,
    )
    write_csv(output_dir / TRAINING_FILENAME, TRAINING_FIELDS, training_rows)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_reward": ORIGINAL_REWARD_LABEL,
        "metrics_config": config.to_dict(),
        "evaluation_settings": next(iter(settings.values())),
        "runs": [
            {
                "experiment_id": r.experiment_id,
                "training_seed": r.training_seed,
                "run_dir": r.run_dir.name,
                "evaluation_session": r.evaluation_dir.name,
            }
            for r in runs
        ],
        "missing_runs": missing,
        "pooled_counts": pooled,
        **environment_metadata(),
    }
    write_json(output_dir / COMPARISON_FILENAME, result)
    return result
