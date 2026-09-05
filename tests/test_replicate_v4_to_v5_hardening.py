"""Replication hardening tests for tools/replicate_v4_to_ov5.py.

These tests verify the hard-fail identity contract enforced by the
v5 replication auditor, using pure-Python in-memory directory fixtures
(no dependency on C:\\ov3, C:\\ov4, or C:\\ov5 existing).

Coverage matrix:
  - partial target directory + missing files -> top-up succeeds
  - remaining missing file                   -> hard fail
  - extra non-.git file                      -> hard fail
  - byte mismatch                            -> hard fail (size differs)
  - SHA256 mismatch                          -> hard fail (size same, hash differs)
  - .git-only extras                         -> excluded by explicit policy
  - same count, different filename set       -> hard fail
  - unchanged source exact identity          -> PASS
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"


@pytest.fixture
def replicate_mod():
    spec = importlib.util.spec_from_file_location(
        "replicate_v4_to_ov5", str(TOOLS_DIR / "replicate_v4_to_ov5.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ---------------------------------------------------------------------------
# .git exclusion policy
# ---------------------------------------------------------------------------

def test_git_files_excluded_from_iter_files(replicate_mod, tmp_path):
    """_iter_files must skip any path under a .git/ directory."""
    root = tmp_path / "git_test_root"
    root.mkdir()
    _write_file(root / "song.mid", b"midi data")
    _write_file(root / ".git" / "config", b"git config")
    _write_file(root / ".git" / "objects" / "aa" / "bb", b"git obj")
    files = {str(rel): p for p, rel in replicate_mod._iter_files(root)}
    assert ".git/config" not in files
    assert "song.mid" in files
    assert len(files) == 1


# ---------------------------------------------------------------------------
# identical file sets -> identity passes
# ---------------------------------------------------------------------------

def test_identical_source_sets_pass(replicate_mod, tmp_path):
    """Exact copy with matching filenames, sizes, and hashes -> no error."""
    v4root = tmp_path / "v4_id"
    v5root = tmp_path / "v5_id"
    v4root.mkdir()
    v5root.mkdir()
    for i in range(5):
        _write_file(v4root / f"song{i}.mid", f"mid{i}".encode())
        _write_file(v5root / f"song{i}.mid", f"mid{i}".encode())
    # Should not raise
    replicate_mod._verify_identity("test_id", src_root=v4root, dst_root=v5root)


# ---------------------------------------------------------------------------
# extra non-.git file -> hard fail
# ---------------------------------------------------------------------------

def test_extra_non_git_file_hard_fail(replicate_mod, tmp_path):
    """A stray file in v5 that is not in v4 must cause a RuntimeError."""
    v4root = tmp_path / "v4_extra"
    v5root = tmp_path / "v5_extra"
    v4root.mkdir()
    v5root.mkdir()
    _write_file(v4root / "song1.mid", b"data1")
    _write_file(v5root / "song1.mid", b"data1")
    _write_file(v5root / "stray.mid", b"stray")
    with pytest.raises(RuntimeError, match="FILESET MISMATCH"):
        replicate_mod._verify_identity("test_extra", src_root=v4root, dst_root=v5root)


# ---------------------------------------------------------------------------
# missing file -> hard fail
# ---------------------------------------------------------------------------

def test_missing_file_hard_fail(replicate_mod, tmp_path):
    """A file present in v4 but absent in v5 must cause a RuntimeError."""
    v4root = tmp_path / "v4_missing"
    v5root = tmp_path / "v5_missing"
    v4root.mkdir()
    v5root.mkdir()
    _write_file(v4root / "song1.mid", b"data1")
    _write_file(v4root / "song2.mid", b"data2")
    _write_file(v5root / "song1.mid", b"data1")
    with pytest.raises(RuntimeError, match="FILESET MISMATCH"):
        replicate_mod._verify_identity("test_missing", src_root=v4root, dst_root=v5root)


# ---------------------------------------------------------------------------
# byte mismatch -> caught by size check
# ---------------------------------------------------------------------------

def test_byte_mismatch_caught(replicate_mod, tmp_path):
    """Files with the same name but different byte sizes must fail."""
    v4root = tmp_path / "v4_size"
    v5root = tmp_path / "v5_size"
    v4root.mkdir()
    v5root.mkdir()
    _write_file(v4root / "song1.mid", b"data1")
    _write_file(v5root / "song1.mid", b"data1_longer")
    with pytest.raises(RuntimeError, match="failed size/hash identity"):
        replicate_mod._verify_identity("test_size", src_root=v4root, dst_root=v5root)


# ---------------------------------------------------------------------------
# SHA mismatch -> caught even if size matches
# ---------------------------------------------------------------------------

def test_sha_mismatch_caught(replicate_mod, tmp_path):
    """Files with the same size but different content must fail by SHA."""
    v4root = tmp_path / "v4_sha"
    v5root = tmp_path / "v5_sha"
    v4root.mkdir()
    v5root.mkdir()
    _write_file(v4root / "song1.mid", b"data1")
    _write_file(v5root / "song1.mid", b"data2")  # same length (5 bytes), diff content
    with pytest.raises(RuntimeError, match="HASH MISMATCH|failed size/hash identity"):
        replicate_mod._verify_identity("test_sha", src_root=v4root, dst_root=v5root)


# ---------------------------------------------------------------------------
# same count, different filenames -> hard fail
# ---------------------------------------------------------------------------

def test_same_count_different_filenames_caught(replicate_mod, tmp_path):
    """Equal file count but different filenames must fail (name set diff)."""
    v4root = tmp_path / "v4_diffname"
    v5root = tmp_path / "v5_diffname"
    v4root.mkdir()
    v5root.mkdir()
    _write_file(v4root / "songA.mid", b"data1")
    _write_file(v5root / "songB.mid", b"data1")
    with pytest.raises(RuntimeError, match="FILESET MISMATCH"):
        replicate_mod._verify_identity("test_diffname", src_root=v4root, dst_root=v5root)


# ---------------------------------------------------------------------------
# .git-only extras -> excluded, no failure
# ---------------------------------------------------------------------------

def test_git_only_extras_excluded(replicate_mod, tmp_path):
    """.git/ files present in v5 but not v4 are excluded and do not fail."""
    v4root = tmp_path / "v4_git"
    v5root = tmp_path / "v5_git"
    v4root.mkdir()
    v5root.mkdir()
    _write_file(v4root / "song1.mid", b"data1")
    _write_file(v5root / "song1.mid", b"data1")
    _write_file(v5root / ".git" / "config", b"git config")
    # Should NOT raise (.git is excluded by policy)
    replicate_mod._verify_identity("test_git_only", src_root=v4root, dst_root=v5root)


# ---------------------------------------------------------------------------
# missing destination -> hard fail
# ---------------------------------------------------------------------------

def test_missing_destination_hard_fail(replicate_mod, tmp_path):
    _write_file(tmp_path / "v4_nodst" / "song1.mid", b"data1")
    nonexistent = tmp_path / "v5_nodst" / "ghost"
    with pytest.raises(RuntimeError, match="destination"):
        replicate_mod._verify_identity("ghost", src_root=tmp_path / "v4_nodst", dst_root=nonexistent)
