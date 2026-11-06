"""Visualization config validation and the Phase 06 command-line error paths."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lunar_lander_rl.config import (
    ConfigError,
    load_visualization_config,
    parse_visualization_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "visualization.yaml"


@pytest.fixture
def raw() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return copy.deepcopy(data)


def test_repository_config_is_valid() -> None:
    config = load_visualization_config(CONFIG_PATH)
    assert config.phase05_results == Path("results/phase05")
    assert config.trace_episode == 0 and config.video_episode == 0
    assert (
        config.to_dict()["component_traces"] == "results/phase06/component_traces.csv"
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("training_curve", "rolling_episodes", 0, ">= 1"),
        ("selection", "trace_episode", -1, ">= 0"),
        ("figure", "dpi", 10, ">= 50"),
        ("inputs", "phase05_results", "C:/abs/results", "relative"),
        ("inputs", "phase05_results", "/abs/results", "relative"),
        ("inputs", "component_traces", "", "non-empty"),
        ("selection", "video_episode", True, "integer"),
    ],
)
def test_invalid_values_are_rejected(
    raw: dict[str, Any], section: str, key: str, value: Any, message: str
) -> None:
    raw[section][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_visualization_config(raw)


def test_unknown_and_missing_keys_are_rejected(raw: dict[str, Any]) -> None:
    raw["figure"]["extra"] = 1
    with pytest.raises(ConfigError, match="unknown"):
        parse_visualization_config(raw)
    del raw["figure"]["extra"]
    del raw["selection"]
    with pytest.raises(ConfigError, match="missing"):
        parse_visualization_config(raw)


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["make_figures", "extract_traces", "collect_videos"])
def test_cli_reports_missing_inputs_without_traceback(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)  # relative input paths resolve to nothing here
    config = tmp_path / "visualization.yaml"
    config.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    code = _load_script(name).main(
        ["--config", str(config), "--output", str(tmp_path / "out" / "x.csv")]
    )
    assert code == 2
    assert "error" in capsys.readouterr().err.lower()
    assert not (tmp_path / "out").exists()


def test_cli_rejects_invalid_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("inputs: {}\n", encoding="utf-8")
    code = _load_script("make_figures").main(
        ["--config", str(config), "--output", str(tmp_path / "out")]
    )
    assert code == 2
    assert "Figure error" in capsys.readouterr().err
