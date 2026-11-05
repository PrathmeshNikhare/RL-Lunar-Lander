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


def test_compare_runs_main_and_config_error(
    shaped_smoke_run: TrainingResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    from lunar_lander_rl.evaluate import evaluate_run

    compare_cli = _load("compare_runs")
    source = shaped_smoke_run.run_dir
    runs_root = tmp_path / "runs"
    target = runs_root / source.parent.name / source.name
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("evaluations", "videos")
    )
    evaluate_run(target, episodes=1, timestamp="20000101T000001Z")
    meta = read_json(target / "metadata.json")
    config: dict[str, Any] = {
        "run_selection": {
            "source_commit": meta["source"]["commit"][:7],
            "experiments": [meta["experiment_id"]],
            "seeds": [meta["training_seed"]],
        },
        "hover": {"min_height": 0.1, "max_speed": 0.2, "min_consecutive_steps": 5},
        "training_window": {"start_timestep": 0},
    }
    config_path = tmp_path / "metrics.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "out"
    argv = [
        "--config",
        str(config_path),
        "--output",
        str(out),
        "--runs-root",
        str(runs_root),
    ]
    assert compare_cli.main(argv) == 0
    assert (out / "comparison.json").is_file()
    assert "Compared 1 runs" in capsys.readouterr().out
    # Existing output and invalid config: clean exit code 2, no traceback.
    assert compare_cli.main(argv) == 2
    config["hover"]["max_speed"] = 0
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    argv[3] = str(tmp_path / "out2")
    assert compare_cli.main(argv) == 2
    assert "Comparison error" in capsys.readouterr().err
    assert not (tmp_path / "out2").exists()
