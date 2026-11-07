"""Tests for copying PPO diagnostics out of TensorBoard event files."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from lunar_lander_rl.ppo_diagnostics import (
    DIAGNOSTIC_TAGS,
    DiagnosticsError,
    extract_ppo_diagnostics,
)


def _write_run(
    runs_root: Path, experiment: str, run_dir: str, tags: tuple[str, ...]
) -> None:
    writer = SummaryWriter(
        str(runs_root / experiment / run_dir / "tensorboard" / "ppo_1")
    )
    for i, tag in enumerate(tags):
        for step in (16384, 32768):
            writer.add_scalar(tag, i + step / 1e5, step)
    writer.close()


def _comparison(tmp_path: Path) -> Path:
    path = tmp_path / "comparison.json"
    runs = [
        {"experiment_id": "baseline", "training_seed": 42, "run_dir": "A_seed42"},
        {"experiment_id": "time_penalty", "training_seed": 142, "run_dir": "B_seed142"},
    ]
    path.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return path


def test_extracts_every_tag_of_every_listed_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "baseline", "A_seed42", DIAGNOSTIC_TAGS)
    _write_run(runs_root, "time_penalty", "B_seed142", DIAGNOSTIC_TAGS)
    output = tmp_path / "out" / "ppo_diagnostics.csv"

    record = extract_ppo_diagnostics(_comparison(tmp_path), runs_root, output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == record["n_rows"] == 2 * len(DIAGNOSTIC_TAGS) * 2
    kl = [
        r
        for r in rows
        if r["experiment_id"] == "baseline" and r["tag"] == "train/approx_kl"
    ]
    index = DIAGNOSTIC_TAGS.index("train/approx_kl")
    assert [int(r["timesteps"]) for r in kl] == [16384, 32768]
    assert [float(r["value"]) for r in kl] == pytest.approx(
        [index + 0.16384, index + 0.32768]
    )
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert [r["training_seed"] for r in sidecar["runs"]] == [42, 142]
    assert "tfevents" not in output.with_suffix(".json").read_text(encoding="utf-8")


def test_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "ppo_diagnostics.csv"
    output.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        extract_ppo_diagnostics(_comparison(tmp_path), tmp_path / "runs", output)
    assert output.read_text(encoding="utf-8") == "old"


def test_missing_tag_or_log_is_an_error(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "baseline", "A_seed42", DIAGNOSTIC_TAGS[:-1])
    with pytest.raises(DiagnosticsError, match="missing tags"):
        extract_ppo_diagnostics(_comparison(tmp_path), runs_root, tmp_path / "a.csv")
    # A second event file in the same run makes the source ambiguous.
    _write_run(runs_root, "baseline", "A_seed42", DIAGNOSTIC_TAGS)
    with pytest.raises(DiagnosticsError, match="expected 1 event file"):
        extract_ppo_diagnostics(_comparison(tmp_path), runs_root, tmp_path / "b.csv")


def test_cli_returns_2_on_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # scripts/ is not a package, so load the file directly.
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "extract_ppo_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "script_extract_ppo_diagnostics", path
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    code = cli.main(
        [
            "--comparison",
            str(_comparison(tmp_path)),
            "--runs-root",
            str(tmp_path / "runs"),
            "--output",
            str(tmp_path / "x.csv"),
        ]
    )
    assert code == 2
    assert "expected 1 event file" in capsys.readouterr().err
