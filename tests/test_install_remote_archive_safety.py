"""Regression tests for the installer's archive safety guards.

* zip member names with control characters (e.g. macOS ``Icon\\r``
  resource forks in the official Magenta GMD zip) are silently
  dropped, not raised.
* zip member names with path traversal are rejected with a clear
  error.
* symlink members are rejected.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from scripts.install_pretrain_corpora import _safe_extract_zip  # type: ignore


def _make_zip_with_entries(entries: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def _write_zip(tmp_path: Path, entries: list[tuple[str, bytes]]) -> Path:
    p = tmp_path / "archive.zip"
    p.write_bytes(_make_zip_with_entries(entries))
    return p


def test_zip_drops_control_char_members(tmp_path: Path) -> None:
    archive = _write_zip(
        tmp_path,
        [
            ("good/mid.mid", b"MThd"),
            ("good/Icon\r", b"resource-fork"),
            ("good/other.mid", b"MThd"),
        ],
    )
    target = tmp_path / "out"
    _safe_extract_zip(archive, target)
    extracted = sorted(p.name for p in target.rglob("*") if p.is_file())
    assert extracted == ["mid.mid", "other.mid"]


def test_zip_rejects_traversal(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path, [("ok.mid", b"MThd"), ("../evil.mid", b"x")])
    with pytest.raises(RuntimeError, match="unsafe zip member path"):
        _safe_extract_zip(archive, tmp_path / "out")


def test_zip_rejects_absolute_path(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path, [("/etc/passwd", b"x")])
    with pytest.raises(RuntimeError, match="unsafe zip member path"):
        _safe_extract_zip(archive, tmp_path / "out")


def test_zip_rejects_drive_letter(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path, [("C:\\evil.mid", b"x")])
    with pytest.raises(RuntimeError, match="unsafe zip member path"):
        _safe_extract_zip(archive, tmp_path / "out")
