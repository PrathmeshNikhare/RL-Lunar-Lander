r"""Build the Phase 06 figures from committed result files.

Usage (PowerShell, from the project root)::

    .\.venv\Scripts\python.exe scripts\make_figures.py --config configs\visualization.yaml --output results\phase06\figures

Reads only committed files (``results/phase04``, ``results/phase05``,
``results/phase06/component_traces.*``). The output directory must not exist.
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

from lunar_lander_rl.config import ConfigError, load_visualization_config
from lunar_lander_rl.figures import FigureDataError, make_figures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the Phase 06 figures.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the figures and return a process exit code."""
    args = parse_args(argv)
    try:
        result = make_figures(load_visualization_config(args.config), args.output)
    except (ConfigError, FigureDataError, FileExistsError, FileNotFoundError) as exc:
        print(f"Figure error: {exc}", file=sys.stderr)
        return 2
    for figure in result["figures"].values():
        print(f"{figure['file']}  sha256 {figure['sha256'][:12]}")
    print(f"Manifest: {args.output / 'figures_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
