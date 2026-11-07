r"""Copy PPO training diagnostics from local TensorBoard logs into a CSV.

Usage (PowerShell, from the project root)::

    .\.venv\Scripts\python.exe scripts\extract_ppo_diagnostics.py --comparison results\phase05\comparison.json --output results\phase07\ppo_diagnostics.csv

Needs the local, git-ignored ``runs/`` folders of the runs listed in
``comparison.json``. Writes the CSV and a ``.json`` provenance record beside
it; neither may already exist.
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
from lunar_lander_rl.ppo_diagnostics import DiagnosticsError, extract_ppo_diagnostics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract PPO training diagnostics.")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Extract the diagnostics and return a process exit code."""
    args = parse_args(argv)
    try:
        record = extract_ppo_diagnostics(args.comparison, args.runs_root, args.output)
    except (DiagnosticsError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"Diagnostics extraction error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {record['n_rows']} rows from {len(record['runs'])} runs to {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
