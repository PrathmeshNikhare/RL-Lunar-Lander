"""Evaluate a trained run on the original LunarLander-v3 reward.

Usage (PowerShell, from the project root)::

    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py --run-dir runs\\baseline\\<run>
    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py --run-dir runs\\baseline\\<run> --episodes 10

Episode count, determinism and seeds default to the run's saved config.
Results go to a new ``<run>\\evaluations\\<timestamp>\\`` directory.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# The project package is not installed (decision D-015).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunar_lander_rl.artifacts import MODEL_FILENAME, read_json
from lunar_lander_rl.cli import non_negative_int, positive_int
from lunar_lander_rl.evaluate import SUMMARY_FILENAME, evaluate_run


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model", default=MODEL_FILENAME, help="Model file inside the run dir."
    )
    parser.add_argument("--episodes", type=positive_int, default=None)
    parser.add_argument("--seed-start", type=non_negative_int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run evaluation and return a process exit code."""
    args = parse_args(argv)
    session_dir = evaluate_run(
        args.run_dir,
        model_filename=args.model,
        episodes=args.episodes,
        seed_start=args.seed_start,
    )
    summary = read_json(session_dir / SUMMARY_FILENAME)["summary"]
    print(f"Evaluation written to: {session_dir}")
    print(
        f"Episodes: {summary['n_episodes']}  "
        f"mean original return: {summary['mean_return']:.2f} "
        f"(std {summary['std_return']:.2f}, min {summary['min_return']:.2f}, "
        f"max {summary['max_return']:.2f})"
    )
    print(
        f"Mean length: {summary['mean_length']:.1f}  terminated: {summary['n_terminated']}  "
        f"truncated: {summary['n_truncated']}  final reward +100: "
        f"{summary['n_final_reward_plus_100']}  -100: {summary['n_final_reward_minus_100']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
