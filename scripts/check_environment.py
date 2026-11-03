"""Report interpreter and dependency status for the local development environment.

Read-only diagnostic: it imports the core packages, prints their versions and
constructs ``LunarLander-v3`` once to confirm Box2D works. It does not train,
render, write files or explore the environment (that is Phase 01).

Usage (PowerShell, from the project root)::

    .\\.venv\\Scripts\\python.exe scripts\\check_environment.py

Exit code is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import importlib
import platform
import struct
import sys
from importlib import metadata

REQUIRED_PYTHON: tuple[int, int] = (3, 12)
REQUIRED_POINTER_BITS: int = 64  # Box2D and torch Windows wheels are 64-bit only
BITS_PER_BYTE: int = 8
ENV_ID: str = "LunarLander-v3"

# (import name, distribution name used for the version lookup)
CORE_PACKAGES: tuple[tuple[str, str], ...] = (
    ("gymnasium", "gymnasium"),
    ("Box2D", "Box2D"),
    ("pygame", "pygame-ce"),
    ("stable_baselines3", "stable-baselines3"),
    ("torch", "torch"),
    ("tensorboard", "tensorboard"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
)


def check_python() -> bool:
    """Return True if the interpreter matches the required major.minor and is 64-bit."""
    actual = sys.version_info[:2]
    ok = actual == REQUIRED_PYTHON
    status = "OK" if ok else "FAIL"
    required = ".".join(map(str, REQUIRED_PYTHON))
    print(f"[{status}] Python {platform.python_version()} (required {required}.x)")
    print(f"       executable: {sys.executable}")
    bits = struct.calcsize("P") * BITS_PER_BYTE
    is_64_bit = bits == REQUIRED_POINTER_BITS
    status = "OK" if is_64_bit else "FAIL"
    print(f"[{status}] {bits}-bit interpreter (required {REQUIRED_POINTER_BITS}-bit)")
    return ok and is_64_bit


def check_imports() -> bool:
    """Import each core package and print its installed distribution version."""
    all_ok = True
    for module_name, dist_name in CORE_PACKAGES:
        try:
            importlib.import_module(module_name)
            version = metadata.version(dist_name)
        except Exception as exc:  # noqa: BLE001 - diagnostic must report any failure
            all_ok = False
            print(f"[FAIL] {module_name}: {type(exc).__name__}: {exc}")
            continue
        print(f"[OK]   {module_name} {version}")
    return all_ok


def check_environment_construction() -> bool:
    """Construct, reset and close the target environment once (no rendering)."""
    try:
        import gymnasium as gym

        env = gym.make(ENV_ID)
        try:
            env.reset(seed=0)
        finally:
            env.close()
    except Exception as exc:  # noqa: BLE001 - diagnostic must report any failure
        print(f"[FAIL] gym.make({ENV_ID!r}): {type(exc).__name__}: {exc}")
        return False
    print(f"[OK]   gym.make({ENV_ID!r}) + reset + close")
    return True


def report_accelerator() -> None:
    """Print CUDA availability for information only; a GPU is not required."""
    try:
        import torch

        available = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001 - informational only
        print(f"[INFO] CUDA check skipped: {type(exc).__name__}: {exc}")
        return
    print(f"[INFO] torch.cuda.is_available() = {available} (GPU not required)")


def main() -> int:
    """Run all checks and return a process exit code."""
    results = [check_python(), check_imports(), check_environment_construction()]
    report_accelerator()
    passed = all(results)
    print("All checks passed." if passed else "One or more checks FAILED.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
