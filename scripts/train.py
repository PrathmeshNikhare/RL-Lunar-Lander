"""Train PPO on LunarLander-v3 from a YAML experiment configuration.

Usage (PowerShell, from the project root)::

    .\\.venv\\Scripts\\python.exe scripts\\train.py --config configs\\baseline.yaml
    .\\.venv\\Scripts\\python.exe scripts\\train.py --config configs\\baseline.yaml --total-timesteps 20000

Full runs (1M timesteps) are long; launch them explicitly. Each invocation
creates a new run directory under ``--runs-root`` and never overwrites one.
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
from lunar_lander_rl.cli import non_negative_int, positive_int
from lunar_lander_rl.config import ConfigError, load_config
from lunar_lander_rl.train import train


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=positive_int, default=None)
    parser.add_argument("--seed", type=non_negative_int, default=None)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress SB3 progress tables."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run training and return a process exit code."""
    args = parse_args(argv)
    try:
        config = load_config(args.config).with_overrides(
            total_timesteps=args.total_timesteps, seed=args.seed
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Training {config.experiment_id}: {config.training.total_timesteps} timesteps, "
        f"seed {config.training.seed}, {config.training.n_envs} envs"
    )
    result = train(
        config,
        args.runs_root,
        argv=sys.argv if argv is None else ["scripts/train.py", *argv],
        verbose=0 if args.quiet else 1,
    )
    print(f"Run directory: {result.run_dir}")
    print(f"Timesteps trained: {result.num_timesteps}")
    print(f"Duration: {result.duration_seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
