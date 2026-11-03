"""Small argparse helpers shared by the command-line scripts in ``scripts/``."""

from __future__ import annotations

import argparse


def positive_int(text: str) -> int:
    """argparse ``type=`` for integers > 0."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def non_negative_int(text: str) -> int:
    """argparse ``type=`` for integers >= 0."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value}")
    return value
