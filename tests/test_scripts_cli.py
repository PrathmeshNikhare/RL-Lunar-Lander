"""In-process tests for the train / evaluate / record_video command-line scripts."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lunar_lander_rl.artifacts import read_json
from lunar_lander_rl.train import TrainingResult

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    # scripts/ is not a package, so load each file directly.
    spec = importlib.util.spec_from_file_location(
        f"script_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_cli() -> ModuleType:
    return _load("train")


@pytest.fixture(scope="module")
def evaluate_cli() -> ModuleType:
    return _load("evaluate")


@pytest.fixture(scope="module")
def video_cli() -> ModuleType:
    return _load("record_video")


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--config", "x.yaml", "--total-timesteps", "0"],
        ["--config", "x.yaml", "--seed", "-1"],
    ],
)
def test_train_rejects_invalid_arguments(
    train_cli: ModuleType, argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        train_cli.parse_args(argv)
    assert excinfo.value.code == 2


def test_train_reports_config_error_and_creates_nothing(
    train_cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_root = tmp_path / "runs"
    code = train_cli.main(
        ["--config", str(tmp_path / "absent.yaml"), "--runs-root", str(runs_root)]
    )
    assert code == 2
    assert "Configuration error" in capsys.readouterr().err
    assert not runs_root.exists()


def test_train_main_launches_a_short_run(
    train_cli: ModuleType, baseline_dict: dict[str, Any], tmp_path: Path
) -> None:
    data = copy.deepcopy(baseline_dict)
    data["experiment_id"] = "cli"
    data["training"].update(n_envs=2, checkpoint_freq=0)
    data["ppo"].update(n_steps=64, batch_size=32, n_epochs=1)
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    runs_root = tmp_path / "runs"
    code = train_cli.main(
        [
            "--config",
            str(config_path),
            "--total-timesteps",
            "128",
            "--seed",
            "3",
            "--quiet",
            "--runs-root",
            str(runs_root),
        ]
    )
    assert code == 0
    (run_dir,) = (runs_root / "cli").iterdir()
    assert run_dir.name.endswith("_seed3")
    meta = read_json(run_dir / "metadata.json")
    assert meta["status"] == "completed"
    assert meta["argv"][0] == "scripts/train.py"
    saved = read_json(run_dir / "config.json")
    assert saved["training"]["total_timesteps"] == 128
    assert saved["training"]["seed"] == 3


def test_evaluate_main_prints_summary(
    evaluate_cli: ModuleType,
    smoke_run: TrainingResult,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # One call only: CLI sessions use 1-second timestamps.
    code = evaluate_cli.main(["--run-dir", str(smoke_run.run_dir), "--episodes", "1"])
    assert code == 0
    output = capsys.readouterr().out
    assert "mean original return" in output
    assert "Episodes: 1" in output


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--run-dir", "r", "--episodes", "0"],
        ["--run-dir", "r", "--seed-start", "-1"],
    ],
)
def test_evaluate_and_video_reject_invalid_arguments(
    evaluate_cli: ModuleType, video_cli: ModuleType, argv: list[str]
) -> None:
    for cli in (evaluate_cli, video_cli):
        with pytest.raises(SystemExit):
            cli.parse_args(argv)
