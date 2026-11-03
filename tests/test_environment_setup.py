"""Fast setup checks: interpreter version, core imports and environment construction.

These tests do not train, render or assert stochastic outcomes.
"""

from __future__ import annotations

import importlib
import struct
import sys

import pytest

REQUIRED_PYTHON = (3, 12)
REQUIRED_POINTER_BITS = 64
BITS_PER_BYTE = 8
ENV_ID = "LunarLander-v3"
CORE_MODULES = (
    "gymnasium",
    "Box2D",
    "pygame",
    "stable_baselines3",
    "torch",
    "tensorboard",
    "matplotlib",
)


def test_python_version_is_required_minor() -> None:
    assert sys.version_info[:2] == REQUIRED_PYTHON, (
        f"Expected Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x, "
        f"got {sys.version.split()[0]} at {sys.executable}"
    )


def test_interpreter_is_64_bit() -> None:
    assert struct.calcsize("P") * BITS_PER_BYTE == REQUIRED_POINTER_BITS, (
        "Box2D and torch Windows wheels require a 64-bit Python interpreter"
    )


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_lunar_lander_constructs_and_resets() -> None:
    import gymnasium as gym

    env = gym.make(ENV_ID)
    try:
        observation, info = env.reset(seed=0)
    finally:
        env.close()
    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)
