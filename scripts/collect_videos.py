r"""Copy one recorded clip per compared run, with a data-derived manifest (Phase 06).

Usage (PowerShell, from the project root)::

    .\.venv\Scripts\python.exe scripts\collect_videos.py --config configs\visualization.yaml --output results\phase06\videos

Needs the local, git-ignored ``runs/`` video sessions. The output directory
must not exist.
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

from lunar_lander_rl.artifacts import DEFAULT_RUNS_ROOT
from lunar_lander_rl.config import ConfigError, load_visualization_config
from lunar_lander_rl.video_selection import VideoSelectionError, collect_videos


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Collect one clip per compared run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Copy the clips and return a process exit code."""
    args = parse_args(argv)
    try:
        result = collect_videos(
            load_visualization_config(args.config), args.runs_root, args.output
        )
    except (
        ConfigError,
        VideoSelectionError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"Video selection error: {exc}", file=sys.stderr)
        return 2
    for clip in result["clips"]:
        print(
            f"{clip['file']}: {clip['outcome']}, {clip['length']} steps, "
            f"{clip['bytes']} bytes, identical in {clip['identical_clip_sessions']}"
            f"/{clip['matching_sessions']} matching sessions"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
