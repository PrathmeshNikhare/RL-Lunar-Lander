"""Figures: built only from committed-style inputs, write-once, honest axes.

Uses a small synthetic results tree (2 configurations x 2 seeds x 4
evaluation episodes); no training or environment runs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.collections import PathCollection

from lunar_lander_rl.artifacts import sha256_file
from lunar_lander_rl.comparison import EPISODE_METRIC_FIELDS, TRAINING_FIELDS
from lunar_lander_rl.config import VisualizationConfig
from lunar_lander_rl.figures import (
    MANIFEST_FILENAME,
    FigureDataError,
    build_figures,
    cumulative,
    describe_reward,
    make_figures,
    rolling_mean,
)
from lunar_lander_rl.plotting import config_color, seed_style
from lunar_lander_rl.traces import TRACE_FIELDS

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline": {"mode": "original"},
    "time_penalty": {"mode": "time_penalty", "coefficient": 0.05},
}
SEEDS = (1, 2)
N_EVAL = 4
N_TRAIN = 12
SHAPED_SENTINEL = 9999.0  # a shaped training mean that must never be plotted


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _results(root: Path) -> VisualizationConfig:
    runs, episodes, tve, traces, trace_meta, summary = [], [], [], [], [], []
    for experiment, reward in EXPERIMENTS.items():
        component = None if reward["mode"] == "original" else reward["mode"]
        for seed in SEEDS:
            ident = {
                "experiment_id": experiment,
                "training_seed": seed,
                "run_dir": f"r{seed}",
            }
            runs.append({**ident, "evaluation_session": "s"})
            for e in range(N_EVAL):
                episodes.append(
                    {
                        **ident,
                        "episode": e,
                        "seed": 10000 + e,
                        "original_return": 100.0 + 10 * e + seed,
                        "length": 300 + e,
                        "terminated": True,
                        "truncated": False,
                        "final_reward": 100.0,
                        "final_x": 0.0,
                        "final_y": 0.0,
                        "outcome": "success" if e else "crash",
                        "first_contact_step": 250,
                        "max_hover_steps": 20 * e + seed,
                        "hover": False,
                        "main_engine_steps": 100,
                        "side_engine_steps": 10,
                    }
                )
            tve.append(
                {
                    **ident,
                    "window_start_timestep": 900,
                    "n_training_episodes": 3,
                    "training_mean_original": 50.0 + seed,
                    "training_mean_shaped_total": SHAPED_SENTINEL,
                    "evaluation_n_episodes": N_EVAL,
                    "evaluation_mean_original": 115.0,
                }
            )
            folder = root / "phase04" / experiment / f"seed{seed}"
            folder.mkdir(parents=True)
            (folder / "run_config.json").write_text(
                json.dumps({"reward": reward, "ppo": {"gamma": 0.999}}),
                encoding="utf-8",
            )
            fields = ("timesteps", "env_rank", "length", "original")
            fields += (component,) if component else ()
            _write_csv(
                folder / "reward_components.csv",
                (*fields, "total"),
                [
                    {
                        "timesteps": 100 * (i + 1),
                        "env_rank": 0,
                        "length": 50,
                        "original": float(i),
                        **({component: -2.5} if component else {}),
                        "total": float(i) - (2.5 if component else 0.0),
                    }
                    for i in range(N_TRAIN)
                ],
            )
            length = 3
            for step in range(1, length + 1):
                traces.append(
                    {
                        "experiment_id": experiment,
                        "training_seed": seed,
                        "step": step,
                        "action": 0,
                        "x": 0.0,
                        "y": 1.0 - 0.3 * step,
                        "vx": 0.0,
                        "vy": 0.0,
                        "angle": 0.0,
                        "angular_velocity": 0.0,
                        "left_leg_contact": 0.0,
                        "right_leg_contact": 0.0,
                        "original": 1.0,
                        "custom_component": component or "",
                        "custom_value": -0.05 if component else "",
                        "training_reward": 0.95 if component else 1.0,
                    }
                )
            trace_meta.append(
                {
                    **ident,
                    "length": length,
                    "outcome": "success",
                    "original_return": 3.0,
                }
            )
        summary.append(
            {
                "experiment_id": experiment,
                "metric": "mean_return",
                "n_seeds": 2,
                "mean": 116.5,
                "std": 0.7,
                "min": 116.0,
                "max": 117.0,
            }
        )
    p05 = root / "phase05"
    _write_csv(p05 / "episode_metrics.csv", EPISODE_METRIC_FIELDS, episodes)
    _write_csv(p05 / "training_vs_evaluation.csv", TRAINING_FIELDS, tve)
    _write_csv(
        p05 / "config_summary.csv",
        ("experiment_id", "metric", "n_seeds", "mean", "std", "min", "max"),
        summary,
    )
    (p05 / "comparison.json").write_text(
        json.dumps(
            {
                "metrics_config": {
                    "run_selection": {
                        "experiments": list(EXPERIMENTS),
                        "seeds": list(SEEDS),
                    },
                    "hover": {
                        "min_height": 0.1,
                        "max_speed": 0.2,
                        "min_consecutive_steps": 30,
                    },
                },
                "evaluation_settings": {"episodes": N_EVAL, "seed_start": 10000},
                "runs": runs,
                "missing_runs": [],
            }
        ),
        encoding="utf-8",
    )
    traces_csv = root / "phase06" / "component_traces.csv"
    _write_csv(traces_csv, TRACE_FIELDS, traces)
    traces_csv.with_suffix(".json").write_text(
        json.dumps(
            {
                "output_csv_sha256": sha256_file(traces_csv),
                "visualization_config": {"trace_episode": 0},
                "episodes": trace_meta,
            }
        ),
        encoding="utf-8",
    )
    return VisualizationConfig(
        phase04_results=root / "phase04",
        phase05_results=p05,
        component_traces=traces_csv,
        rolling_episodes=5,
        trace_episode=0,
        video_episode=0,
        dpi=50,
    )


def test_make_figures_writes_pngs_and_manifest(tmp_path: Path) -> None:
    config = _results(tmp_path)
    out = tmp_path / "figures"
    result = make_figures(config, out)
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 8
    manifest = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    for name, figure in manifest["figures"].items():
        assert figure["file"] == f"{name}.png"
        assert figure["sha256"] == sha256_file(out / figure["file"])
    for item in manifest["inputs"].values():
        assert item["sha256"] == sha256_file(Path(item["path"]))
    assert result["figures"]["06_hover_diagnostic"]["post_hoc_line"] == 3 * 20 + 2
    assert (
        result["figures"]["06_hover_diagnostic"]["post_hoc_reference_experiment"]
        == "baseline"
    )
    with pytest.raises(FileExistsError):
        make_figures(config, out)


def test_rerun_reproduces_identical_pngs(tmp_path: Path) -> None:
    config = _results(tmp_path)
    first = make_figures(config, tmp_path / "a")
    second = make_figures(config, tmp_path / "b")
    assert {k: v["sha256"] for k, v in first["figures"].items()} == {
        k: v["sha256"] for k, v in second["figures"].items()
    }


def _scatter_y(figure: Any) -> set[float]:
    values: set[float] = set()
    for ax in figure.axes:
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):  # scatter points only
                continue
            offsets = np.asarray(collection.get_offsets())
            if offsets.size:
                values.update(float(v) for v in offsets[:, 1])
    return values


def test_evaluation_axes_hold_only_original_reward(tmp_path: Path) -> None:
    figures, _ = build_figures(_results(tmp_path))
    returns = _scatter_y(figures["03_evaluation_returns"])
    assert returns == {100.0 + 10 * e + s for e in range(N_EVAL) for s in SEEDS}
    tve = _scatter_y(figures["07_training_vs_evaluation"])
    assert SHAPED_SENTINEL not in tve
    assert {51.0, 52.0, 115.0} <= tve
    assert "original" in figures["03_evaluation_returns"].axes[0].get_ylabel()
    for ax in figures["02_training_custom_terms"].axes:
        assert "training" in ax.get_ylabel()


def test_inconsistent_inputs_are_rejected(tmp_path: Path) -> None:
    config = _results(tmp_path)
    with config.component_traces.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(FigureDataError, match="does not match"):
        build_figures(config)


def test_missing_evaluation_episodes_are_rejected(tmp_path: Path) -> None:
    config = _results(tmp_path)
    path = config.phase05_results / "episode_metrics.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(FigureDataError, match="evaluation episodes"):
        build_figures(config)
    assert not (tmp_path / "figures").exists()


def test_rolling_mean_orders_by_timestep_and_uses_full_windows() -> None:
    timesteps = np.array([30, 10, 20, 40])
    values = np.array([3.0, 1.0, 2.0, 4.0])
    x, y = rolling_mean(timesteps, values, 2)
    assert x.tolist() == [20, 30, 40]
    assert y.tolist() == [1.5, 2.5, 3.5]
    with pytest.raises(FigureDataError):
        rolling_mean(timesteps, values, 5)


def test_cumulative_sums_left_to_right() -> None:
    values = [0.1, 0.2, 0.3]
    expected = 0.0
    for value in values:
        expected += value
    assert cumulative(values)[-1] == expected
    assert math.isclose(cumulative(values)[0], 0.1)


def test_describe_reward_states_every_parameter() -> None:
    assert describe_reward({"mode": "original"}, 0.999) == "original reward only"
    assert "0.05" in describe_reward(
        {"mode": "time_penalty", "coefficient": 0.05}, 0.999
    )
    bonus = describe_reward(
        {
            "mode": "upright_slow_bonus",
            "bonus": 1.0,
            "max_abs_angle": 0.2,
            "max_speed": 0.2,
        },
        0.999,
    )
    assert "+1/step" in bonus and "0.2 rad" in bonus
    shaping = describe_reward(
        {
            "mode": "potential_based",
            "alpha": 100.0,
            "distance_weight": 1.0,
            "speed_weight": 1.0,
            "angle_weight": 1.0,
        },
        0.999,
    )
    assert "alpha=100" in shaping and "gamma=0.999" in shaping
    with pytest.raises(FigureDataError):
        describe_reward({"mode": "other"}, 0.999)


def test_palette_slots_are_fixed_and_bounded() -> None:
    assert config_color(0) != config_color(1)
    assert seed_style(0)[1] != seed_style(1)[1]  # line style as a second cue
    with pytest.raises(ValueError):
        config_color(4)
    with pytest.raises(ValueError):
        seed_style(3)


def test_outcome_groups_are_neutral_and_paired_markers_do_not_overlap(
    tmp_path: Path,
) -> None:
    figures, _ = build_figures(_results(tmp_path))
    (outcome_ax,) = figures["04_evaluation_outcomes"].axes
    group_labels = [t for t in outcome_ax.texts if t.get_text() in EXPERIMENTS]
    assert group_labels
    # Outcome segments use the configuration colour slots, so names stay neutral.
    assert {t.get_color() for t in group_labels} == {"#0b0b0b"}
    (tve_ax,) = figures["07_training_vs_evaluation"].axes
    xs = [
        float(np.asarray(c.get_offsets())[0, 0])
        for c in tve_ax.collections
        if isinstance(c, PathCollection)
    ]
    assert len(xs) == len(set(xs))  # training and evaluation markers side by side
