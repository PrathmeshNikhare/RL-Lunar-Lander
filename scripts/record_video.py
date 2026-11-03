"""Record MP4 videos of a trained run's evaluation episodes (original reward).

Usage (PowerShell, from the project root)::

    .\\.venv\\Scripts\\python.exe scripts\\record_video.py --run-dir runs\\baseline\\<run>
    .\\.venv\\Scripts\\python.exe scripts\\record_video.py --run-dir runs\\baseline\\<run> --episodes 1

Episode count, determinism and seeds default to the run's saved ``video``
config. Output goes to a new ``<run>\\videos\\<timestamp>\\`` directory, which
is git-ignored.
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

from lunar_lander_rl.artifacts import MODEL_FILENAME
from lunar_lander_rl.cli import non_negative_int, positive_int
from lunar_lander_rl.record_video import record_run_videos


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Record videos of a trained PPO run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model", default=MODEL_FILENAME, help="Model file inside the run dir."
    )
    parser.add_argument("--episodes", type=positive_int, default=None)
    parser.add_argument("--seed-start", type=non_negative_int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Record videos and return a process exit code."""
    args = parse_args(argv)
    videos = record_run_videos(
        args.run_dir,
        model_filename=args.model,
        episodes=args.episodes,
        seed_start=args.seed_start,
    )
    for video in videos:
        print(video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
