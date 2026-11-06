"""Build the Phase 06 figures from committed result files only.

Inputs (paths from ``configs/visualization.yaml``):

- ``results/phase05/``: ``comparison.json`` (run list, evaluation settings,
  hover thresholds), ``episode_metrics.csv``, ``config_summary.csv``,
  ``training_vs_evaluation.csv``;
- ``results/phase04/<experiment>/seed<N>/``: ``reward_components.csv``
  (training episodes) and ``run_config.json`` (reward definitions);
- ``results/phase06/component_traces.csv`` and its ``.json``.

Outputs go to a new directory: one PNG per figure and ``figures_manifest.json``
with input hashes, sample sizes and the source revision. Nothing is
overwritten.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import gymnasium as gym
import matplotlib
import numpy as np
from matplotlib.figure import Figure

from lunar_lander_rl.artifacts import (
    REWARD_COMPONENTS_FILENAME,
    environment_metadata,
    read_json,
    sha256_file,
    write_json,
)
from lunar_lander_rl.config import VisualizationConfig
from lunar_lander_rl.environment import ENV_ID
from lunar_lander_rl.evaluate import REWARD_THRESHOLD
from lunar_lander_rl.metrics import OUTCOMES
from lunar_lander_rl.plotting import (
    PLOT_STYLE,
    Curve,
    EpisodeTrace,
    ReferenceLine,
    outcome_figure,
    strip_figure,
    trace_figure,
    training_component_figure,
    training_original_figure,
    training_vs_evaluation_figure,
)

MANIFEST_FILENAME: Final[str] = "figures_manifest.json"
RUN_CONFIG_COPY: Final[str] = "run_config.json"
_TRAINING_BASE_FIELDS: Final[tuple[str, ...]] = (
    "timesteps",
    "env_rank",
    "length",
    "original",
    "total",
)
_PNG_METADATA: Final[dict[str, str | None]] = {"Software": None}


class FigureDataError(ValueError):
    """Raised when committed result files are missing or inconsistent."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def describe_reward(reward: Mapping[str, Any], gamma: float) -> str:
    """Short, exact description of a reward configuration for figure titles."""
    mode = reward["mode"]
    if mode == "original":
        return "original reward only"
    if mode == "time_penalty":
        return f"time penalty: -{reward['coefficient']:g} per step"
    if mode == "upright_slow_bonus":
        return (
            f"bonus +{reward['bonus']:g}/step if |angle| <= {reward['max_abs_angle']:g} rad"
            f" and speed <= {reward['max_speed']:g}"
        )
    if mode == "potential_based":
        return (
            f"potential shaping: alpha={reward['alpha']:g}, gamma={gamma:g}, weights "
            f"d={reward['distance_weight']:g} v={reward['speed_weight']:g} "
            f"a={reward['angle_weight']:g}"
        )
    raise FigureDataError(f"unknown reward mode {mode!r}")


def rolling_mean(
    timesteps: np.ndarray, values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean over ``window`` consecutive episodes; full windows only.

    Episodes are ordered by the timestep at which they ended (stable sort);
    each mean is plotted at the last episode's timestep.
    """
    if len(values) < window:
        raise FigureDataError(f"{len(values)} episodes is fewer than window {window}")
    order = np.argsort(timesteps, kind="stable")
    ordered = values[order]
    cumulative = np.concatenate(([0.0], np.cumsum(ordered, dtype=np.float64)))
    means = (cumulative[window:] - cumulative[:-window]) / window
    return timesteps[order][window - 1 :], means


def cumulative(values: Sequence[float]) -> np.ndarray:
    """Running sum, accumulated left to right like the episode return."""
    out = np.empty(len(values))
    total = 0.0
    for i, value in enumerate(values):
        total += value
        out[i] = total
    return out


class _Inputs:
    """Committed inputs, loaded and cross-checked once."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config
        p05 = config.phase05_results
        self.paths: dict[str, Path] = {
            "comparison": p05 / "comparison.json",
            "episode_metrics": p05 / "episode_metrics.csv",
            "config_summary": p05 / "config_summary.csv",
            "training_vs_evaluation": p05 / "training_vs_evaluation.csv",
            "component_traces": config.component_traces,
            "component_traces_json": config.component_traces.with_suffix(".json"),
        }
        self.comparison = read_json(self.paths["comparison"])
        selection = self.comparison["metrics_config"]["run_selection"]
        self.experiments: list[str] = list(selection["experiments"])
        self.seeds: list[int] = [int(s) for s in selection["seeds"]]
        listed = {
            (r["experiment_id"], int(r["training_seed"]))
            for r in self.comparison["runs"]
        }
        expected = {(e, s) for e in self.experiments for s in self.seeds}
        if listed != expected or self.comparison.get("missing_runs"):
            raise FigureDataError("comparison.json does not hold every selected run")
        self.episodes = _read_csv(self.paths["episode_metrics"])
        self.config_summary = _read_csv(self.paths["config_summary"])
        self.training_vs_evaluation = _read_csv(self.paths["training_vs_evaluation"])
        self.traces = _read_csv(self.paths["component_traces"])
        self.traces_meta = read_json(self.paths["component_traces_json"])
        if self.traces_meta["output_csv_sha256"] != sha256_file(
            self.paths["component_traces"]
        ):
            raise FigureDataError(
                "component_traces.csv does not match its .json record"
            )
        self.n_evaluation = int(self.comparison["evaluation_settings"]["episodes"])
        self.run_configs: dict[tuple[str, int], dict[str, Any]] = {}
        self.training: dict[tuple[str, int], list[dict[str, str]]] = {}
        for experiment, seed in sorted(expected):
            folder = config.phase04_results / experiment / f"seed{seed}"
            self.paths[f"run_config:{experiment}:{seed}"] = folder / RUN_CONFIG_COPY
            self.paths[f"training:{experiment}:{seed}"] = (
                folder / REWARD_COMPONENTS_FILENAME
            )
            self.run_configs[(experiment, seed)] = read_json(folder / RUN_CONFIG_COPY)
            self.training[(experiment, seed)] = _read_csv(
                folder / REWARD_COMPONENTS_FILENAME
            )
        self._check_episode_counts()

    def _check_episode_counts(self) -> None:
        for experiment in self.experiments:
            for seed in self.seeds:
                n = sum(
                    1
                    for r in self.episodes
                    if r["experiment_id"] == experiment
                    and int(r["training_seed"]) == seed
                )
                if n != self.n_evaluation:
                    raise FigureDataError(
                        f"{experiment} seed {seed}: {n} evaluation episodes, "
                        f"expected {self.n_evaluation}"
                    )

    def reward(self, experiment: str) -> tuple[dict[str, Any], float]:
        """The experiment's reward definition and gamma (identical across seeds)."""
        found = {
            repr((c["reward"], c["ppo"]["gamma"]))
            for (e, _), c in self.run_configs.items()
            if e == experiment
        }
        if len(found) != 1:
            raise FigureDataError(f"{experiment}: reward differs between seeds")
        config = self.run_configs[(experiment, self.seeds[0])]
        return config["reward"], float(config["ppo"]["gamma"])

    def description(self, experiment: str) -> str:
        reward, gamma = self.reward(experiment)
        return describe_reward(reward, gamma)

    def episode_values(self, field: str) -> dict[str, dict[int, np.ndarray]]:
        out: dict[str, dict[int, np.ndarray]] = {}
        for experiment in self.experiments:
            out[experiment] = {}
            for seed in self.seeds:
                out[experiment][seed] = np.array(
                    [
                        float(r[field])
                        for r in self.episodes
                        if r["experiment_id"] == experiment
                        and int(r["training_seed"]) == seed
                    ]
                )
        return out

    def config_stat(self, experiment: str, metric: str) -> dict[str, float]:
        for row in self.config_summary:
            if row["experiment_id"] == experiment and row["metric"] == metric:
                return {
                    k: float(row[k]) for k in ("n_seeds", "mean", "std", "min", "max")
                }
        raise FigureDataError(f"config_summary.csv has no {experiment}/{metric}")


def _training_curves(
    inputs: _Inputs, window: int
) -> tuple[dict[str, dict[int, Curve]], dict[str, dict[int, Curve]], dict[str, str]]:
    original: dict[str, dict[int, Curve]] = {}
    custom: dict[str, dict[int, Curve]] = {}
    component_of: dict[str, str] = {}
    for experiment in inputs.experiments:
        original[experiment] = {}
        for seed in inputs.seeds:
            rows = inputs.training[(experiment, seed)]
            extra = [k for k in rows[0] if k not in _TRAINING_BASE_FIELDS]
            if len(extra) > 1:
                raise FigureDataError(
                    f"{experiment}: several custom components {extra}"
                )
            timesteps = np.array([int(r["timesteps"]) for r in rows])
            values = np.array([float(r["original"]) for r in rows])
            x, y = rolling_mean(timesteps, values, window)
            original[experiment][seed] = Curve(x, y, len(rows))
            if extra:
                name = extra[0]
                if component_of.setdefault(experiment, name) != name:
                    raise FigureDataError(
                        f"{experiment}: component differs between seeds"
                    )
                x, y = rolling_mean(
                    timesteps, np.array([float(r[name]) for r in rows]), window
                )
                custom.setdefault(experiment, {})[seed] = Curve(x, y, len(rows))
    return original, custom, component_of


def _traces(inputs: _Inputs) -> dict[str, dict[int, EpisodeTrace]]:
    out: dict[str, dict[int, EpisodeTrace]] = {}
    meta = {
        (e["experiment_id"], int(e["training_seed"])): e
        for e in inputs.traces_meta["episodes"]
    }
    for experiment in inputs.experiments:
        out[experiment] = {}
        for seed in inputs.seeds:
            rows = [
                r
                for r in inputs.traces
                if r["experiment_id"] == experiment and int(r["training_seed"]) == seed
            ]
            info = meta.get((experiment, seed))
            if not rows or info is None or len(rows) != int(info["length"]):
                raise FigureDataError(f"{experiment} seed {seed}: trace rows missing")
            steps = np.array([int(r["step"]) for r in rows])
            if not np.array_equal(steps, np.arange(1, len(rows) + 1)):
                raise FigureDataError(f"{experiment} seed {seed}: trace steps not 1..n")
            custom_cells = [r["custom_value"] for r in rows]
            out[experiment][seed] = EpisodeTrace(
                steps=steps,
                cumulative_original=cumulative([float(r["original"]) for r in rows]),
                cumulative_custom=None
                if all(c == "" for c in custom_cells)
                else cumulative([float(c) for c in custom_cells]),
                height=np.array([float(r["y"]) for r in rows]),
                label=(
                    f"seed {seed}: {info['outcome']}, {info['length']} steps, "
                    f"return {float(info['original_return']):.1f}"
                ),
            )
    return out


def _reference_experiment(inputs: _Inputs) -> str:
    """The single configuration trained on the original reward."""
    originals = [
        e for e in inputs.experiments if inputs.reward(e)[0]["mode"] == "original"
    ]
    if len(originals) != 1:
        raise FigureDataError(
            f"expected one original-reward configuration, got {originals}"
        )
    return originals[0]


def build_figures(
    config: VisualizationConfig,
) -> tuple[dict[str, Figure], dict[str, Any]]:
    """Load the committed inputs and build every figure (nothing is written)."""
    inputs = _Inputs(config)
    window = config.rolling_episodes
    n_eval = inputs.n_evaluation
    settings = inputs.comparison["evaluation_settings"]
    seed_range = f"{settings['seed_start']}-{settings['seed_start'] + n_eval - 1}"
    eval_note = (
        f"n = {n_eval} evaluation episodes per run (env seeds {seed_range}), "
        "deterministic policy, original environment reward, no wrappers."
    )
    time_limit = gym.spec(ENV_ID).max_episode_steps
    if time_limit is None:
        raise FigureDataError(f"{ENV_ID} has no registered time limit")
    hover = inputs.comparison["metrics_config"]["hover"]
    reference = _reference_experiment(inputs)
    hover_values = inputs.episode_values("max_hover_steps")
    posthoc = int(max(hover_values[reference][s].max() for s in inputs.seeds))

    figures: dict[str, Figure] = {}
    meta: dict[str, Any] = {}
    original, custom, component_of = _training_curves(inputs, window)
    n_training = {
        f"{e}:{s}": c.n_episodes for e, g in original.items() for s, c in g.items()
    }
    figures["01_training_original_reward"] = training_original_figure(original, window)
    meta["01_training_original_reward"] = {
        "shows": "rolling mean of the original reward per training episode",
        "rolling_episodes": window,
        "n_training_episodes": n_training,
    }
    figures["02_training_custom_terms"] = training_component_figure(
        custom, {e: inputs.description(e) for e in custom}, window
    )
    meta["02_training_custom_terms"] = {
        "shows": "rolling mean of each custom term per training episode",
        "components": component_of,
        "rolling_episodes": window,
    }

    notes = {}
    for experiment in inputs.experiments:
        stat = inputs.config_stat(experiment, "mean_return")
        notes[experiment] = (
            f"mean of {int(stat['n_seeds'])} run means: {stat['mean']:.1f} "
            f"(sample std {stat['std']:.1f})"
        )
    figures["03_evaluation_returns"] = strip_figure(
        inputs.episode_values("original_return"),
        title="Evaluation return per episode (original environment reward)",
        ylabel="Episode return (original reward)",
        caption=eval_note + " Source: results/phase05/episode_metrics.csv.",
        reference_lines=[
            ReferenceLine(
                REWARD_THRESHOLD, f"{ENV_ID} reward threshold ({REWARD_THRESHOLD:g})"
            )
        ],
        group_notes=notes,
    )
    meta["03_evaluation_returns"] = {"n_per_run": n_eval, "group_notes": notes}

    counts: dict[str, dict[int, dict[str, int]]] = {
        e: {
            s: {
                o: sum(
                    1
                    for r in inputs.episodes
                    if r["experiment_id"] == e
                    and int(r["training_seed"]) == s
                    and r["outcome"] == o
                )
                for o in OUTCOMES
            }
            for s in inputs.seeds
        }
        for e in inputs.experiments
    }
    figures["04_evaluation_outcomes"] = outcome_figure(
        counts,
        OUTCOMES,
        caption=eval_note
        + " success = at rest (+100) with |x| <= 0.2; rest_off_pad = at rest elsewhere; "
        "crash / out_of_bounds = -100; timeout = truncated at the time limit "
        "(docs/07-metrics.md). Source: results/phase05/episode_metrics.csv.",
    )
    meta["04_evaluation_outcomes"] = {"n_per_run": n_eval, "counts": counts}

    figures["05_episode_lengths"] = strip_figure(
        inputs.episode_values("length"),
        title="Evaluation episode length",
        ylabel="Episode length (environment steps, 50 per simulated second)",
        caption=eval_note + " Source: results/phase05/episode_metrics.csv.",
        reference_lines=[
            ReferenceLine(
                float(time_limit), f"time limit: truncated at {time_limit} steps"
            )
        ],
    )
    meta["05_episode_lengths"] = {"n_per_run": n_eval, "time_limit": time_limit}

    figures["06_hover_diagnostic"] = strip_figure(
        hover_values,
        title="Longest slow, airborne stretch per evaluation episode (hover diagnostic)",
        ylabel="Longest run of consecutive hover steps",
        caption=eval_note
        + f" A hover step has both legs off the ground, y >= {hover['min_height']:g} and "
        f"speed <= {hover['max_speed']:g} (D-037). The pre-registered rule flags "
        f">= {hover['min_consecutive_steps']} steps and also flags slow final descents "
        f"(D-039). The {posthoc}-step line is the largest {reference} value, chosen after "
        "seeing the data: descriptive only. Source: results/phase05/episode_metrics.csv.",
        reference_lines=[
            ReferenceLine(
                float(hover["min_consecutive_steps"]),
                f"pre-registered hover threshold ({hover['min_consecutive_steps']} steps)",
            ),
            ReferenceLine(
                float(posthoc), f"post-hoc: largest {reference} value ({posthoc})", ":"
            ),
        ],
    )
    meta["06_hover_diagnostic"] = {
        "n_per_run": n_eval,
        "pre_registered_threshold": hover["min_consecutive_steps"],
        "post_hoc_line": posthoc,
        "post_hoc_reference_experiment": reference,
    }

    tve: dict[str, dict[int, tuple[float, float, int]]] = {}
    window_start = None
    for row in inputs.training_vs_evaluation:
        window_start = int(row["window_start_timestep"])
        tve.setdefault(row["experiment_id"], {})[int(row["training_seed"])] = (
            float(row["training_mean_original"]),
            float(row["evaluation_mean_original"]),
            int(row["n_training_episodes"]),
        )
    if window_start is None or set(tve) != set(inputs.experiments):
        raise FigureDataError("training_vs_evaluation.csv is incomplete")
    tve = {e: {s: tve[e][s] for s in inputs.seeds} for e in inputs.experiments}
    figures["07_training_vs_evaluation"] = training_vs_evaluation_figure(
        tve,
        window_start=window_start,
        n_evaluation=n_eval,
        caption="Both markers use the original reward; the shaped training totals are "
        "deliberately not shown here. Source: results/phase05/training_vs_evaluation.csv.",
    )
    meta["07_training_vs_evaluation"] = {"window_start_timestep": window_start}

    trace_episode = int(inputs.traces_meta["visualization_config"]["trace_episode"])
    env_seed = int(settings["seed_start"]) + trace_episode
    figures["08_episode_traces"] = trace_figure(
        _traces(inputs),
        {e: inputs.description(e) for e in inputs.experiments},
        reference_height=ReferenceLine(
            float(hover["min_height"]),
            f"hover min height ({hover['min_height']:g})",
            ":",
        ),
        caption=(
            f"Evaluation episode {trace_episode} (env seed {env_seed}) of every run, "
            "chosen before looking at any trace. The custom term is what the run's training "
            "reward would have added in this episode (recorded observations replayed "
            "through RewardComponentsWrapper); evaluation itself used the original reward. "
            "Source: results/phase06/component_traces.csv."
        ),
    )
    meta["08_episode_traces"] = {
        "evaluation_episode": trace_episode,
        "env_seed": env_seed,
        "episodes": inputs.traces_meta["episodes"],
    }
    manifest = {
        "inputs": {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in sorted(inputs.paths.items())
        },
        "figures": meta,
    }
    return figures, manifest


def make_figures(config: VisualizationConfig, output_dir: Path) -> dict[str, Any]:
    """Build every figure and write PNGs plus a manifest to the new ``output_dir``."""
    provenance = environment_metadata()  # before any output exists
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; nothing was overwritten")
    with matplotlib.rc_context(PLOT_STYLE):
        figures, manifest = build_figures(config)
        output_dir.mkdir(parents=True, exist_ok=False)
        files: dict[str, str] = {}
        for name, figure in figures.items():
            path = output_dir / f"{name}.png"
            with path.open("xb") as handle:
                figure.savefig(
                    handle, format="png", dpi=config.dpi, metadata=_PNG_METADATA
                )
            files[name] = sha256_file(path)
    for name, digest in files.items():
        manifest["figures"][name]["file"] = f"{name}.png"
        manifest["figures"][name]["sha256"] = digest
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "visualization_config": config.to_dict(),
        **manifest,
        **provenance,
    }
    write_json(output_dir / MANIFEST_FILENAME, result)
    return result
