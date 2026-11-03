"""Run bounded random-policy episodes in LunarLander-v3 and print what happens.

Exploration tool for Phase 01. Uses the original environment reward only.

Usage (PowerShell, from the project root)::

    .\\.venv\\Scripts\\python.exe scripts\\random_rollout.py --episodes 3 --seed 0
    .\\.venv\\Scripts\\python.exe scripts\\random_rollout.py --episodes 2 --render human

Episode ``i`` (0-based) is seeded with ``seed + i``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# The project package is not installed (decision D-015); make the repository
# root importable when this file is run directly as ``python scripts\...``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunar_lander_rl.cli import non_negative_int, positive_int
from lunar_lander_rl.environment import (
    ACTION_MEANINGS,
    OBSERVATION_LABELS,
    LunarLanderEnv,
    RenderMode,
    make_env,
)
from lunar_lander_rl.rollout import EpisodeSummary, StepRecord, run_random_episode

RENDER_CHOICES: tuple[str, ...] = ("none", "human")
DEFAULT_EPISODES = 3
DEFAULT_SEED = 0
DEFAULT_PRINT_STEPS = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--episodes", type=positive_int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=non_negative_int, default=DEFAULT_SEED)
    parser.add_argument("--render", choices=RENDER_CHOICES, default="none")
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Outer bound on steps per episode (default: the env's registered time limit).",
    )
    parser.add_argument(
        "--print-steps",
        type=non_negative_int,
        default=DEFAULT_PRINT_STEPS,
        help="Number of initial transitions to print per episode.",
    )
    return parser.parse_args(argv)


def _format_observation(values: Sequence[float]) -> str:
    return ", ".join(
        f"{label}={v:+.3f}" for label, v in zip(OBSERVATION_LABELS, values, strict=True)
    )


def _print_spaces(env: LunarLanderEnv) -> None:
    print(f"Environment: {env.spec.id if env.spec else 'unknown'}")
    print(
        f"Time limit (max_episode_steps): {env.spec.max_episode_steps if env.spec else 'unknown'}"
    )
    print(f"Observation space: {env.observation_space}")
    print(f"  components ({len(OBSERVATION_LABELS)}): {', '.join(OBSERVATION_LABELS)}")
    print(f"Action space: {env.action_space}")
    for action, meaning in ACTION_MEANINGS.items():
        print(f"  {action}: {meaning}")


def _print_step(record: StepRecord) -> None:
    print(
        f"  step {record.step:4d} action={record.action} reward={record.reward:+8.3f} "
        f"terminated={record.terminated} truncated={record.truncated}"
    )
    print(f"             obs: {_format_observation(record.observation)}")


def _print_summary(index: int, summary: EpisodeSummary) -> None:
    if summary.terminated:
        ended_by = "terminated"
    elif summary.truncated:
        ended_by = "truncated (time limit)"
    else:
        ended_by = "step cap reached"
    print(
        f"Episode {index} (seed {summary.seed}): steps={summary.steps} "
        f"original_return={summary.original_return:+.2f} ended_by={ended_by} "
        f"terminated={summary.terminated} truncated={summary.truncated}"
    )
    print(
        f"  final obs: {_format_observation([float(v) for v in summary.final_observation])}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested episodes and return a process exit code."""
    args = parse_args(argv)
    render_mode: RenderMode | None = "human" if args.render == "human" else None
    env = make_env(render_mode=render_mode)
    try:
        _print_spaces(env)
        max_steps: int | None = args.max_steps
        if max_steps is None:
            if env.spec is None or env.spec.max_episode_steps is None:
                raise RuntimeError(
                    "Environment has no registered time limit; pass --max-steps"
                )
            max_steps = env.spec.max_episode_steps
        print(f"Per-episode step bound: {max_steps}\n")
        for index in range(args.episodes):
            summary = run_random_episode(
                env,
                seed=args.seed + index,
                max_steps=max_steps,
                record_steps=args.print_steps,
            )
            print(f"Episode {index} first {len(summary.recorded_steps)} steps:")
            for record in summary.recorded_steps:
                _print_step(record)
            _print_summary(index, summary)
            print()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
