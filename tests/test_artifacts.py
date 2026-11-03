"""Run-directory, write-once file and metadata tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lunar_lander_rl.artifacts import (
    create_run_dir,
    create_session_dir,
    environment_metadata,
    read_json,
    sha256_file,
    utc_timestamp,
    write_csv,
    write_json,
)


def test_utc_timestamp_format() -> None:
    moment = datetime(2026, 11, 3, 17, 5, 9, tzinfo=UTC)
    assert utc_timestamp(moment) == "20261103T170509Z"


def test_run_dir_layout_and_refuses_existing(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path, "baseline", 42, timestamp="20261103T170509Z")
    assert run_dir == tmp_path / "baseline" / "20261103T170509Z_seed42"
    assert run_dir.is_dir()
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "baseline", 42, timestamp="20261103T170509Z")


def test_session_dir_refuses_existing(tmp_path: Path) -> None:
    session = create_session_dir(tmp_path / "evaluations", timestamp="t1")
    assert session.is_dir()
    with pytest.raises(FileExistsError):
        create_session_dir(tmp_path / "evaluations", timestamp="t1")


def test_write_json_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(path, {"a": 2})
    assert read_json(path) == {"a": 1}


def test_write_csv_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    write_csv(path, ["x", "y"], [{"x": 1, "y": 2}])
    with pytest.raises(FileExistsError):
        write_csv(path, ["x", "y"], [{"x": 3, "y": 4}])
    assert path.read_text(encoding="utf-8").splitlines() == ["x,y", "1,2"]


def test_environment_metadata_schema() -> None:
    meta = environment_metadata(["prog", "--flag"])
    assert set(meta) == {"python", "platform", "packages", "source", "argv"}
    assert meta["argv"] == ["prog", "--flag"]
    for package in ("gymnasium", "stable-baselines3", "torch", "tensorboard", "pillow"):
        assert isinstance(meta["packages"][package], str)
    assert set(meta["source"]) == {"commit", "dirty", "diff_sha256", "untracked"}
    source = meta["source"]
    assert source["commit"] is None or len(source["commit"]) == 40
    assert source["diff_sha256"] is None or len(source["diff_sha256"]) == 64
    assert source["untracked"] is None or isinstance(source["untracked"], list)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 5000  # > one read chunk
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
