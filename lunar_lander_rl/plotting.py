"""Figure builders for the Phase 06 analysis.

Every function takes already-loaded data and returns a new
``matplotlib.figure.Figure``. Nothing here reads or writes files or touches
pyplot's global state; ``lunar_lander_rl.figures`` loads the committed
results and saves the figures (inside ``matplotlib.rc_context(PLOT_STYLE)``).

Colours: the dataviz skill's reference categorical palette (light mode),
validated with its ``validate_palette.js``. Configurations use slots 1-4,
outcomes slots 1-5 (fixed order, never cycled), training seeds slots 5-7
with a different line style each as a second cue. Identity is never colour
alone: every figure has a legend or direct group labels, and the committed
CSV files are the table view.

Training returns that include custom reward terms are never drawn on an axis
that shows evaluation returns.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
from matplotlib.typing import RcKeyType

SURFACE: Final[str] = "#fcfcfb"
TEXT: Final[str] = "#0b0b0b"
TEXT_SECONDARY: Final[str] = "#52514e"
GRID: Final[str] = "#e4e3df"
REFERENCE: Final[str] = "#52514e"
CATEGORICAL: Final[tuple[str, ...]] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
)
SEED_STYLES: Final[tuple[tuple[str, str], ...]] = (
    ("#e87ba4", "-"),
    ("#008300", "--"),
    ("#6250d6", ":"),
)
CAPTION_FONT_SIZE: Final[float] = 7.5
PLOT_STYLE: Final[dict[RcKeyType, Any]] = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "text.color": TEXT,
    "figure.facecolor": SURFACE,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
}
_GROUP_GAP: Final[float] = 1.0  # empty slots between configuration groups
_JITTER_WIDTH: Final[float] = 0.55
_PAIR_OFFSET: Final[float] = 0.14  # training/evaluation markers either side of a run
_GOLDEN: Final[float] = 0.6180339887498949  # deterministic, evenly spread jitter
_CHARS_PER_INCH: Final[float] = 17.0  # caption wrap width at CAPTION_FONT_SIZE
_LINE_INCHES: Final[float] = 0.135  # caption line height at CAPTION_FONT_SIZE
_TITLE_WRAP: Final[int] = 34  # characters per line in panel titles


@dataclass(frozen=True)
class Curve:
    """A rolling-mean curve of one training run."""

    timesteps: np.ndarray
    values: np.ndarray
    n_episodes: int


@dataclass(frozen=True)
class ReferenceLine:
    value: float
    label: str
    style: str = "--"


@dataclass(frozen=True)
class EpisodeTrace:
    """One evaluation episode's per-step series (step 1..length)."""

    steps: np.ndarray
    cumulative_original: np.ndarray
    cumulative_custom: np.ndarray | None
    height: np.ndarray
    label: str


def config_color(index: int) -> str:
    """Colour of the ``index``-th configuration (fixed order)."""
    if not 0 <= index < 4:
        raise ValueError(f"configuration colour slots are 0-3, got {index}")
    return CATEGORICAL[index]


def seed_style(index: int) -> tuple[str, str]:
    """Colour and line style of the ``index``-th training seed (fixed order)."""
    if not 0 <= index < len(SEED_STYLES):
        raise ValueError(f"seed style slots are 0-{len(SEED_STYLES) - 1}, got {index}")
    return SEED_STYLES[index]


def _figure(width: float, height: float, title: str, caption: str) -> Figure:
    """A figure with a left-aligned title and a wrapped caption with reserved space."""
    text = textwrap.fill(caption, width=int(width * _CHARS_PER_INCH))
    reserved = (text.count("\n") + 1) * _LINE_INCHES + 0.12
    fig = Figure(
        figsize=(width, height),
        layout=ConstrainedLayoutEngine(
            rect=(0.0, reserved / height, 1.0, 1.0 - reserved / height)
        ),
    )
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.text(
        0.01,
        0.01,
        text,
        ha="left",
        va="bottom",
        fontsize=CAPTION_FONT_SIZE,
        color=TEXT_SECONDARY,
    )
    return fig


def _description_lines(description: str) -> int:
    return textwrap.fill(description, width=_TITLE_WRAP).count("\n") + 1


def _panel_title(
    ax: Axes, name: str, description: str, colour: str, lines: int = 0
) -> None:
    """Name plus wrapped description, padded to ``lines`` description lines."""
    text = textwrap.fill(description, width=_TITLE_WRAP)
    text += "\n" * max(0, lines - (text.count("\n") + 1))
    ax.set_title(
        name + "\n" + text,
        loc="left",
        fontsize=8.5,
        color=colour,
    )


def _millions(ax: Axes) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}"))


def _reference_lines(ax: Axes, lines: Sequence[ReferenceLine]) -> list[Artist]:
    """Draw horizontal reference lines; return legend handles (no in-plot text)."""
    return [
        ax.axhline(
            line.value,
            color=REFERENCE,
            linewidth=1.0,
            linestyle=line.style,
            label=line.label,
        )
        for line in lines
    ]


def _positions(
    experiments: Sequence[str], seeds: Sequence[int]
) -> dict[tuple[str, int], float]:
    positions: dict[tuple[str, int], float] = {}
    x = 0.0
    for experiment in experiments:
        for seed in seeds:
            positions[(experiment, seed)] = x
            x += 1.0
        x += _GROUP_GAP
    return positions


def _run_axis(
    ax: Axes,
    experiments: Sequence[str],
    seeds: Sequence[int],
    *,
    coloured_labels: bool = True,
) -> dict[tuple[str, int], float]:
    """Seed ticks within configuration groups, group names below the ticks.

    ``coloured_labels=False`` writes the group names in neutral text, for
    figures whose marks use the same colour slots for something else.
    """
    positions = _positions(experiments, seeds)
    ax.set_xticks(
        [positions[k] for k in positions],
        [f"seed {s}" for _, s in positions],
        fontsize=7.5,
    )
    ax.grid(axis="x", visible=False)
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for i, experiment in enumerate(experiments):
        centre = float(np.mean([positions[(experiment, s)] for s in seeds]))
        ax.text(
            centre,
            -0.09,
            experiment,
            transform=transform,
            ha="center",
            va="top",
            fontsize=9,
            color=config_color(i) if coloured_labels else TEXT,
            fontweight="bold",
        )
    ax.set_xlim(-0.7, max(positions.values()) + 0.7)
    return positions


def _bottom_legend(ax: Axes, handles: Sequence[Artist], ncols: int) -> None:
    ax.legend(
        handles=list(handles),
        loc="upper left",
        bbox_to_anchor=(0.0, -0.14),
        ncols=ncols,
    )


def _jitter(n: int) -> np.ndarray:
    return ((np.arange(n) * _GOLDEN) % 1.0 - 0.5) * _JITTER_WIDTH


def training_original_figure(
    curves: Mapping[str, Mapping[int, Curve]], window: int
) -> Figure:
    """Original-reward training curves: one panel per configuration, one line per seed."""
    experiments = list(curves)
    fig = _figure(
        11.0,
        6.6,
        "Training curves on the original environment reward (custom terms excluded)",
        "Training episodes use the stochastic policy being trained (16 parallel "
        "environments), so they are not comparable to deterministic evaluation episodes. "
        "Source: results/phase04/<experiment>/seed<N>/reward_components.csv, column "
        "'original'.",
    )
    axes = fig.subplots(2, 2, sharex=True, sharey=True).ravel()
    for i, (ax, experiment) in enumerate(zip(axes, experiments, strict=False)):
        for j, (seed, curve) in enumerate(curves[experiment].items()):
            colour, style = seed_style(j)
            ax.plot(
                curve.timesteps,
                curve.values,
                color=colour,
                linestyle=style,
                label=f"seed {seed} (n = {curve.n_episodes:,} episodes)",
            )
        ax.set_title(experiment, color=config_color(i), fontweight="bold", loc="left")
        ax.legend(loc="lower right")
        _millions(ax)
    for ax in axes[len(experiments) :]:
        ax.set_visible(False)
    for ax in axes[max(0, len(experiments) - 2) : len(experiments)]:
        ax.set_xlabel("Environment steps (millions, summed over all environments)")
    fig.supylabel(
        f"Original reward per training episode\n(trailing mean of {window} episodes)",
        fontsize=9,
    )
    return fig


def training_component_figure(
    curves: Mapping[str, Mapping[int, Curve]],
    descriptions: Mapping[str, str],
    window: int,
) -> Figure:
    """Per-episode totals of each custom training term; one panel (own y-axis) each."""
    experiments = list(curves)
    fig = _figure(
        11.0,
        4.4,
        "Custom training-reward terms, summed per training episode (separate y-axis per panel)",
        "Training reward only: these terms were never part of evaluation. Source: "
        "results/phase04/<experiment>/seed<N>/reward_components.csv.",
    )
    axes = fig.subplots(1, len(experiments), sharex=True, squeeze=False)[0]
    lines = max(_description_lines(descriptions[e]) for e in experiments)
    for ax, experiment in zip(axes, experiments, strict=True):
        for j, (seed, curve) in enumerate(curves[experiment].items()):
            colour, style = seed_style(j)
            ax.plot(
                curve.timesteps,
                curve.values,
                color=colour,
                linestyle=style,
                label=f"seed {seed} (n = {curve.n_episodes:,})",
            )
        _panel_title(ax, experiment, descriptions[experiment], TEXT, lines)
        ax.set_ylabel(f"Custom term per training episode\n(trailing mean of {window})")
        ax.set_xlabel("Environment steps (millions)")
        ax.legend(loc="best")
        _millions(ax)
    return fig


def strip_figure(
    values: Mapping[str, Mapping[int, np.ndarray]],
    *,
    title: str,
    ylabel: str,
    caption: str,
    reference_lines: Sequence[ReferenceLine] = (),
    group_notes: Mapping[str, str] | None = None,
) -> Figure:
    """Every episode as a dot per run, with the run mean as a black bar."""
    experiments = list(values)
    seeds = list(next(iter(values.values())))
    fig = _figure(11.0, 5.6, title, caption)
    ax = fig.subplots()
    positions = _run_axis(ax, experiments, seeds)
    for i, experiment in enumerate(experiments):
        for seed in seeds:
            data = values[experiment][seed]
            x = positions[(experiment, seed)]
            ax.scatter(
                x + _jitter(len(data)),
                data,
                s=9,
                color=config_color(i),
                alpha=0.5,
                linewidths=0,
            )
            ax.hlines(float(np.mean(data)), x - 0.32, x + 0.32, color=TEXT, linewidth=2)
    references = _reference_lines(ax, reference_lines)
    if group_notes:
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        for experiment, note in group_notes.items():
            centre = float(np.mean([positions[(experiment, s)] for s in seeds]))
            ax.text(
                centre,
                1.01,
                note,
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=TEXT_SECONDARY,
            )
    ax.set_ylabel(ylabel)
    handles: list[Artist] = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=TEXT_SECONDARY,
            markersize=4,
            label="one evaluation episode",
        ),
        Line2D([], [], color=TEXT, linewidth=2, label="run mean"),
        *references,
    ]
    _bottom_legend(ax, handles, ncols=len(handles))
    return fig


def outcome_figure(
    counts: Mapping[str, Mapping[int, Mapping[str, int]]],
    outcomes: Sequence[str],
    caption: str,
) -> Figure:
    """Stacked outcome counts per run (each bar is one run's evaluation episodes)."""
    experiments = list(counts)
    seeds = list(next(iter(counts.values())))
    fig = _figure(
        11.0,
        5.6,
        "Evaluation outcomes per run (original reward, D-036 outcome rule)",
        caption,
    )
    ax = fig.subplots()
    positions = _run_axis(ax, experiments, seeds, coloured_labels=False)
    for k, outcome in enumerate(outcomes):
        colour = CATEGORICAL[k]
        for experiment in experiments:
            for seed in seeds:
                run = counts[experiment][seed]
                bottom = sum(run[o] for o in outcomes[:k])
                height = run[outcome]
                if height == 0:
                    continue
                x = positions[(experiment, seed)]
                ax.bar(
                    x,
                    height,
                    bottom=bottom,
                    width=0.72,
                    color=colour,
                    edgecolor=SURFACE,
                    linewidth=1.0,
                )
                if height >= 6:
                    ax.text(
                        x,
                        bottom + height / 2,
                        str(height),
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=TEXT,
                    )
    totals = {sum(run.values()) for group in counts.values() for run in group.values()}
    ax.set_ylabel(
        "Evaluation episodes per run"
        + (f" (n = {totals.pop()})" if len(totals) == 1 else "")
    )
    _bottom_legend(
        ax,
        [Patch(color=CATEGORICAL[k], label=o) for k, o in enumerate(outcomes)],
        ncols=len(outcomes),
    )
    return fig


def training_vs_evaluation_figure(
    rows: Mapping[str, Mapping[int, tuple[float, float, int]]],
    *,
    window_start: int,
    n_evaluation: int,
    caption: str,
) -> Figure:
    """Late-training versus evaluation mean original return per run.

    ``rows[experiment][seed] = (training_mean_original, evaluation_mean_original,
    n_training_episodes)``. Both values are on the original reward.
    """
    experiments = list(rows)
    seeds = list(next(iter(rows.values())))
    fig = _figure(
        11.0,
        5.2,
        "Late training versus evaluation, both on the original reward",
        caption,
    )
    ax = fig.subplots()
    positions = _run_axis(ax, experiments, seeds)
    for i, experiment in enumerate(experiments):
        colour = config_color(i)
        for seed in seeds:
            training, evaluation, n_training = rows[experiment][seed]
            x = positions[(experiment, seed)]
            xt, xe = x - _PAIR_OFFSET, x + _PAIR_OFFSET  # side by side, never hidden
            ax.plot([xt, xe], [training, evaluation], color=GRID, linewidth=2, zorder=1)
            ax.scatter(
                [xt],
                [training],
                s=46,
                facecolors=SURFACE,
                edgecolors=colour,
                linewidths=1.8,
                zorder=2,
            )
            ax.scatter([xe], [evaluation], s=46, color=colour, zorder=3)
            ax.annotate(
                f"n={n_training}",
                (xt, training),
                xytext=(-7, 0),
                ha="right",
                textcoords="offset points",
                fontsize=6.5,
                color=TEXT_SECONDARY,
                va="center",
            )
    ax.set_ylabel("Mean original reward per episode")
    handles: list[Artist] = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=SURFACE,
            markeredgecolor=TEXT_SECONDARY,
            markersize=7,
            label=f"training: episodes ending at step >= {window_start:,} "
            "(window chosen post hoc, D-035; stochastic policy; n per run shown)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=TEXT_SECONDARY,
            markersize=7,
            label=f"evaluation: {n_evaluation} episodes (deterministic policy)",
        ),
    ]
    _bottom_legend(ax, handles, ncols=2)
    return fig


def trace_figure(
    traces: Mapping[str, Mapping[int, EpisodeTrace]],
    descriptions: Mapping[str, str],
    *,
    reference_height: ReferenceLine,
    caption: str,
) -> Figure:
    """Cumulative original reward, cumulative custom term and height per step."""
    experiments = list(traces)
    fig = _figure(
        12.0,
        9.6,
        "One fixed evaluation episode per run, step by step",
        caption + f" Dotted line in the bottom row: {reference_height.label}.",
    )
    axes = fig.subplots(3, len(experiments), sharex="col", squeeze=False)
    n_lines = max(_description_lines(descriptions[e]) for e in experiments)
    for col, experiment in enumerate(experiments):
        top, middle, bottom = axes[0][col], axes[1][col], axes[2][col]
        _panel_title(
            top, experiment, descriptions[experiment], config_color(col), n_lines
        )
        has_custom = False
        handles: list[Artist] = []
        for j, (_, trace) in enumerate(traces[experiment].items()):
            colour, style = seed_style(j)
            kwargs: dict[str, Any] = {"color": colour, "linestyle": style}
            (line,) = top.plot(
                trace.steps, trace.cumulative_original, label=trace.label, **kwargs
            )
            handles.append(line)
            if trace.cumulative_custom is not None:
                has_custom = True
                middle.plot(trace.steps, trace.cumulative_custom, **kwargs)
            bottom.plot(trace.steps, trace.height, **kwargs)
        if not has_custom:
            middle.text(
                0.5,
                0.5,
                "no custom term\n(original reward only)",
                transform=middle.transAxes,
                ha="center",
                va="center",
                color=TEXT_SECONDARY,
            )
            middle.set_yticks([])
        _reference_lines(bottom, [reference_height])
        bottom.set_xlabel("Step in episode")
        bottom.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(-0.05, -0.2),
            fontsize=6.5,
        )
    axes[0][0].set_ylabel("Cumulative original reward\n(evaluation reward)")
    axes[1][0].set_ylabel("Cumulative custom term\n(training reward only)")
    axes[2][0].set_ylabel("Height y after the step\n(scaled observation units)")
    for row in (0, 2):
        lows = [a.get_ylim()[0] for a in axes[row]]
        highs = [a.get_ylim()[1] for a in axes[row]]
        for ax in axes[row]:
            ax.set_ylim(min(lows), max(highs))
    return fig
