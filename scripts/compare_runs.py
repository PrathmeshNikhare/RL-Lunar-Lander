r"""Compare evaluated runs on the original LunarLander-v3 reward (Phase 05).

Usage (PowerShell, from the project root)::

    .\.venv\Scripts\python.exe scripts\compare_runs.py --config configs\evaluation_metrics.yaml --output results\phase05

Each selected run must already have an evaluation session with
``trajectories.npz`` (``scripts\evaluate.py``). The output directory must
not exist; nothing is overwritten.
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
from lunar_lander_rl.comparison import compare
from lunar_lander_rl.config import load_metrics_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare evaluated runs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and return a process exit code."""
    args = parse_args(argv)
    result = compare(load_metrics_config(args.config), args.runs_root, args.output)
    print(f"Compared {len(result['runs'])} runs; written to: {args.output}")
    for missing in result["missing_runs"]:
        print(f"MISSING: {missing['experiment_id']} seed {missing['training_seed']}")
    for experiment, counts in result["pooled_counts"].items():
        print(f"{experiment}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
