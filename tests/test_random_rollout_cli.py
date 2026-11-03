"""In-process smoke tests for ``scripts/random_rollout.py`` (no rendering)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "random_rollout.py"


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    # scripts/ is not a package, so load the file directly.
    spec = importlib.util.spec_from_file_location("random_rollout", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_runs_short_headless_episode(
    cli: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["--episodes", "1", "--max-steps", "3", "--print-steps", "2"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "LunarLander-v3" in output
    assert "Discrete(4)" in output
    assert "step cap reached" in output
    assert output.count("  step ") == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--episodes", "0"],
        ["--max-steps", "-1"],
        ["--print-steps", "-1"],
        ["--render", "x"],
    ],
)
def test_invalid_arguments_exit(cli: ModuleType, argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(argv)
