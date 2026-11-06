r"""Write per-step reward traces of one fixed evaluation episode per run (Phase 06).

Usage (PowerShell, from the project root)::

    .\.venv\Scripts\python.exe scripts\extract_traces.py --config configs\visualization.yaml --output results\phase06\component_traces.csv

Needs the local, git-ignored evaluation ``trajectories.npz`` files of the
runs listed in the Phase 05 ``comparison.json``. Writes the CSV and a
``.json`` provenance record beside it; neither may already exist.
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
from lunar_lander_rl.traces import TraceError, extract_traces


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract per-step reward traces.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Extract the traces and return a process exit code."""
    args = parse_args(argv)
    try:
        result = extract_traces(
            load_visualization_config(args.config), args.runs_root, args.output
        )
    except (ConfigError, TraceError, FileExistsError, FileNotFoundError) as exc:
        print(f"Trace extraction error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {result['n_rows']} steps from {len(result['episodes'])} episodes to {args.output}"
    )
    for episode in result["episodes"]:
        print(
            f"{episode['experiment_id']} seed {episode['training_seed']}: "
            f"{episode['outcome']}, {episode['length']} steps, "
            f"original {episode['original_return']:.2f}, "
            f"training reward {episode['training_reward_total']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
