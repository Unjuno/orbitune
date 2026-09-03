from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from orbitune.pretrain_corpus import commercial_safe_sources, load_registry


PDMX_FILES = {
    "PDMX.csv": (225399738, "30392ccf38bb63ce70e7afae70f9c88c"),
    "mid.tar.gz": (214395208, "d920a21b2fcd99a56d9c381b39debbb2"),
    "subset_paths.tar.gz": (29258714, "092eee416ece8060f77d08575b94a43d"),
}
PDMX_RECORD = 15571083
_HF_SOURCE_LOCK = ".orbitune_source_lock.json"
_FULL_HEX_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream integrity checksum, not security use
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()  # noqa: S324 - upstream integrity checksum, not security use
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, target: Path, *, expected_size: int | None = None, md5: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size) and (md5 is None or _md5(target) == md5):
        return
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Orbitune-Corpus-Installer/1"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:  # noqa: S310 - registry-controlled HTTPS URL
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if expected_size is not None and partial.stat().st_size != expected_size:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch for {target.name}")
    if md5 is not None and _md5(partial) != md5:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {target.name}")
    partial.replace(target)


def _safe_extract_tar(archive: Path, target: Path) -> None:
    """Extract only ordinary files/directories without relying on 3.12 filters."""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            resolved = (target / member.name).resolve()
            if root != resolved and root not in resolved.parents:
                raise RuntimeError(f"unsafe tar member path: {member.name}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise RuntimeError(f"unsupported tar member type: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsupported tar member type: {member.name}")
        for member in tar.getmembers():
            tar.extract(member, target)  # noqa: S202 - path/type checks above make this portable and bounded


def _safe_extract_zip(archive: Path, target: Path) -> None:
    """Extract a zip archive, rejecting path traversal, symlink members, and
    member names with control characters (e.g. macOS ``Icon\\r`` resource
    forks that would otherwise be silently stored as invalid Windows paths).
    """
    import zipfile

    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            # Reject absolute paths, drive letters, and traversal.
            if name.startswith("/") or name.startswith("\\") or re.match(r"^[A-Za-z]:[\\/]", name):
                raise RuntimeError(f"unsafe zip member path: {name!r}")
            if ".." in Path(name).parts:
                raise RuntimeError(f"unsafe zip member path: {name!r}")
            # Skip any member name with control characters. The Magenta GMD
            # zip carries macOS ``Icon\r`` resource forks that would raise
            # OSError [Errno 22] on Windows extract. We drop them silently:
            # they are not part of the dataset and have no MIDI content.
            if any(ord(c) < 0x20 or ord(c) == 0x7F for c in name):
                continue
            resolved = (target / name).resolve()
            if root != resolved and root not in resolved.parents:
                raise RuntimeError(f"unsafe zip member path: {name!r}")
            # Reject symlink members: zip can carry Unix symlink attributes.
            mode = (info.external_attr >> 16) & 0xFFFF
            if (mode & 0o170000) == 0o120000:
                raise RuntimeError(f"unsupported zip member type: {name!r}")
            zf.extract(info, target)  # noqa: S202 - path/type checks above make this bounded


def install_remote_archive(raw: Mapping[str, object], target: Path) -> dict[str, object]:
    """Download a remote archive, verify its checksum, and safe-extract it.

    The caller (``main()``) does not pass an expected size: only a
    checksum is required. Extraction refuses absolute, traversal, or
    symlink members. The function fails closed if the post-extract
    ``expected_file_globs`` match zero files.
    """
    url = str(raw["url"])
    archive_type = str(raw["archive_type"])
    if archive_type not in {"zip", "tar.gz"}:
        raise SystemExit(f"unsupported archive_type: {archive_type}")
    checksum_algorithm = str(raw.get("checksum_algorithm", "")).lower()
    if checksum_algorithm not in {"md5", "sha256"}:
        raise SystemExit(f"unsupported checksum_algorithm: {checksum_algorithm!r}")
    expected_checksum = str(raw["checksum"]).lower()
    expected_globs = [str(g) for g in raw.get("expected_file_globs", [])]

    target.mkdir(parents=True, exist_ok=True)
    ext = ".zip" if archive_type == "zip" else ".tar.gz"
    archive_path = target / f"archive{ext}"
    _download(url, archive_path, expected_size=None, md5=None)

    actual_checksum = _md5(archive_path) if checksum_algorithm == "md5" else _sha256(archive_path)
    if actual_checksum.lower() != expected_checksum:
        archive_path.unlink(missing_ok=True)
        raise SystemExit(
            f"checksum mismatch for {archive_path.name}: expected {expected_checksum}, got {actual_checksum}"
        )

    if archive_type == "zip":
        _safe_extract_zip(archive_path, target)
    else:
        _safe_extract_tar(archive_path, target)

    if expected_globs:
        matches: list[str] = []
        for pattern in expected_globs:
            matches.extend([str(p.relative_to(target)).replace("\\", "/") for p in target.rglob(pattern)])
        if not matches:
            raise SystemExit(
                f"post-extract check failed: no files matched expected_file_globs {expected_globs!r} in {target}"
            )
    else:
        matches = []

    return {
        "url": url,
        "archive_type": archive_type,
        "checksum_algorithm": checksum_algorithm,
        "checksum": expected_checksum,
        "expected_file_globs": expected_globs,
        "post_extract_matches": matches,
        "path": str(target),
    }


def install_pdmx(target: Path) -> dict[str, object]:
    target.mkdir(parents=True, exist_ok=True)
    for name, (size, checksum) in PDMX_FILES.items():
        url = f"https://zenodo.org/api/records/{PDMX_RECORD}/files/{name}/content"
        _download(url, target / name, expected_size=size, md5=checksum)
    for name in ("mid.tar.gz", "subset_paths.tar.gz"):
        marker = target / f".{name}.extracted"
        if not marker.exists():
            _safe_extract_tar(target / name, target)
            marker.write_text("ok\n", encoding="utf-8")
    return {"record_id": PDMX_RECORD, "files": list(PDMX_FILES), "path": str(target)}


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def install_git_source(url: str, target: Path, *, ref: str | None = None) -> dict[str, object]:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--filter=blob:none", url, str(target)])
    elif not (target / ".git").exists():
        raise RuntimeError(f"{target} exists but is not a git checkout")
    _git("fetch", "--tags", "origin", cwd=target)
    if ref:
        _git("checkout", "--detach", ref, cwd=target)
    else:
        default_ref = _git("symbolic-ref", "refs/remotes/origin/HEAD", cwd=target).split("/")[-1]
        _git("checkout", default_ref, cwd=target)
        _git("pull", "--ff-only", "origin", default_ref, cwd=target)
    return {"git_url": url, "commit": _git("rev-parse", "HEAD", cwd=target), "path": str(target)}


def _require_full_hf_revision(repo_id: str, revision: object) -> str:
    value = str(revision or "").strip()
    if not _FULL_HEX_REVISION.fullmatch(value):
        raise RuntimeError(f"{repo_id}: Hugging Face revision must be an exact 40-character hexadecimal SHA")
    return value.lower()


def install_hf_score_snapshot(source: dict[str, object], target: Path) -> dict[str, object]:
    """Resolve once, lock, and materialize only score files from a HF dataset.

    Dynamic resolution is allowed only as a local *pin creation* step. The first
    successful install stores the exact Hub SHA in ``.orbitune_source_lock.json``.
    Subsequent installs reuse that SHA and never follow a moving default branch.
    A pre-existing non-empty target without a lock fails closed to prevent a
    mixed or unprovenanced snapshot.
    """
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError('Hugging Face score source requires: pip install -e ".[corpus]"') from exc

    repo_id = str(source["repo_id"])
    policy = str(source.get("revision_policy", "")).strip()
    allow_patterns_raw = source.get("allow_patterns", [])
    score_globs_raw = source.get("score_globs", [])
    if not isinstance(allow_patterns_raw, list) or not allow_patterns_raw:
        raise RuntimeError(f"{repo_id}: allow_patterns must be a non-empty list")
    if not isinstance(score_globs_raw, list) or not score_globs_raw:
        raise RuntimeError(f"{repo_id}: score_globs must be a non-empty list")
    allow_patterns = [str(item) for item in allow_patterns_raw]
    score_globs = [str(item) for item in score_globs_raw]
    if any(".pdf" in pattern.lower() for pattern in allow_patterns):
        raise RuntimeError(f"{repo_id}: Base score snapshot must not download PDF payloads")

    lock_path = target / _HF_SOURCE_LOCK
    locked: dict[str, object] | None = None
    if lock_path.exists():
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        if locked.get("repo_id") != repo_id:
            raise RuntimeError(f"{repo_id}: source lock belongs to {locked.get('repo_id')!r}")
        revision = _require_full_hf_revision(repo_id, locked.get("revision"))
        if locked.get("allow_patterns") != allow_patterns:
            raise RuntimeError(f"{repo_id}: source lock allow_patterns differ from registry")
    else:
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"{repo_id}: non-empty target has no {_HF_SOURCE_LOCK}; refusing unprovenanced reuse")
        explicit_revision = source.get("revision")
        if explicit_revision:
            revision = _require_full_hf_revision(repo_id, explicit_revision)
        elif policy == "resolve-exact-at-install":
            resolved = HfApi().dataset_info(repo_id).sha
            revision = _require_full_hf_revision(repo_id, resolved)
        else:
            raise RuntimeError(
                f"{repo_id}: registry must provide an exact revision or revision_policy=resolve-exact-at-install"
            )

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=str(target),
    )

    score_files: set[Path] = set()
    for pattern in score_globs:
        score_files.update(path for path in target.glob(pattern) if path.is_file())
    if not score_files:
        raise RuntimeError(f"{repo_id}@{revision}: snapshot produced zero configured score files")

    lock = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "allow_patterns": allow_patterns,
        "score_globs": score_globs,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "revision_policy": policy or "explicit",
        "score_files": len(score_files),
        "allow_patterns": allow_patterns,
        "lock_file": str(lock_path),
        "path": str(target),
    }


def install_hf_midi(source: dict[str, object], target: Path) -> dict[str, object]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Hugging Face source requires: pip install -e ".[corpus]"') from exc

    repo_id = str(source["repo_id"])
    revision = _require_full_hf_revision(repo_id, source.get("revision"))
    midi_column = str(source.get("midi_column", "midi"))
    target.mkdir(parents=True, exist_ok=True)

    # Remove the pre-v1 unprefixed materialization scheme if a developer ran an
    # earlier branch revision. Otherwise the builder would see those rows twice.
    for legacy in target.glob("[0-9][0-9][0-9][0-9][0-9][0-9].mid"):
        legacy.unlink(missing_ok=True)
        legacy.with_suffix(".json").unlink(missing_ok=True)

    loaded = load_dataset(repo_id, revision=revision)
    if isinstance(loaded, Mapping):
        split_items = sorted(loaded.items(), key=lambda item: str(item[0]))
    else:
        split_items = [("train", loaded)]

    written = 0
    split_counts: dict[str, int] = {}
    for split_name_raw, dataset in split_items:
        split_name = str(split_name_raw)
        count = 0
        for index, row in enumerate(dataset):
            raw = row[midi_column]
            if isinstance(raw, dict) and "bytes" in raw:
                raw = raw["bytes"]
            if not isinstance(raw, (bytes, bytearray)):
                raise RuntimeError(f"{repo_id}:{split_name}:{index}: MIDI column is not bytes")
            stem = f"{split_name}-{index:06d}"
            (target / f"{stem}.mid").write_bytes(bytes(raw))
            meta = {
                key: value
                for key, value in row.items()
                if key != midi_column and isinstance(value, (str, int, float, bool, type(None)))
            }
            meta["upstream_split"] = split_name
            (target / f"{stem}.json").write_text(
                json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            written += 1
            count += 1
        split_counts[split_name] = count

    return {
        "repo_id": repo_id,
        "revision": revision,
        "rows": written,
        "splits": split_counts,
        "path": str(target),
    }


def _merge_install_manifest(
    manifest_path: Path,
    existing: dict[str, object] | None,
    installed: dict[str, object],
    *,
    registry_name: str,
) -> dict[str, object]:
    """Fail-closed merge of an existing install_manifest.json payload with a
    new partial install result.

    The merge preserves prior source entries not touched by the current
    invocation and refuses to silently overwrite any prior source entry
    whose provenance payload differs from the new install. A non-empty
    existing manifest with a different ``registry_name`` is rejected.

    Returns the merged manifest dict that the caller writes to disk.
    """

    existing_sources: dict[str, object] = {}
    if existing is not None:
        existing_registry_name = str(existing.get("registry_name", ""))
        if existing_registry_name and existing_registry_name != registry_name:
            raise SystemExit(
                f"{manifest_path}: refusing to merge sources across registries "
                f"({existing_registry_name!r} vs {registry_name!r}); "
                "remove the file or pass --root for a fresh root"
            )
        sources_obj = existing.get("sources")
        if not isinstance(sources_obj, dict):
            raise SystemExit(
                f"{manifest_path}: refusing to merge: 'sources' is not a dict; "
                "remove the file or pass --root for a fresh root"
            )
        existing_sources = dict(sources_obj)

    for sid, payload in installed.items():
        prior = existing_sources.get(sid)
        if isinstance(prior, dict) and prior != payload:
            raise SystemExit(
                f"{manifest_path}: refusing to overwrite existing source {sid!r} with "
                "a different provenance payload; remove the file or pass a fresh --root"
            )

    merged_sources = {**existing_sources, **installed}
    return {"registry_name": registry_name, "sources": merged_sources}


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Orbitune commercial-safe Base pretraining corpora.")
    parser.add_argument("--config", default="configs/pretrain_corpus_commercial_v1.json")
    parser.add_argument("--root", default="data/corpora/commercial_v1")
    parser.add_argument("--sources", default="all", help="Comma-separated source IDs or 'all'.")
    args = parser.parse_args()

    registry = load_registry(args.config)
    sources = commercial_safe_sources(registry)
    wanted = {item.strip() for item in args.sources.split(",") if item.strip()}
    if wanted != {"all"}:
        unknown = wanted - {source.id for source in sources}
        if unknown:
            raise SystemExit(f"unknown sources: {sorted(unknown)}")
        sources = [source for source in sources if source.id in wanted]

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "install_manifest.json"
    existing: dict[str, object] | None = None
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"{manifest_path}: refusing to overwrite an unreadable install manifest ({exc}); "
                "remove the file or pass --root for a fresh root"
            ) from exc
        if not isinstance(existing, dict):
            raise SystemExit(
                f"{manifest_path}: refusing to overwrite a non-dict install manifest; "
                "remove the file or pass --root for a fresh root"
            )

    installed: dict[str, object] = {}
    for source in sources:
        target = root / source.id
        print(f"[install] {source.id}", flush=True)
        if source.kind == "zenodo_pdmx":
            installed[source.id] = install_pdmx(target)
        elif source.kind == "git_scores":
            installed[source.id] = install_git_source(
                str(source.raw["git_url"]), target, ref=str(source.raw["ref"]) if source.raw.get("ref") else None
            )
        elif source.kind == "huggingface_midi_bytes":
            installed[source.id] = install_hf_midi(source.raw, target)
        elif source.kind == "huggingface_score_snapshot":
            installed[source.id] = install_hf_score_snapshot(source.raw, target)
        elif source.kind == "remote_archive":
            installed[source.id] = install_remote_archive(source.raw, target)
        else:
            raise SystemExit(f"unsupported source kind: {source.kind}")

    manifest = _merge_install_manifest(
        manifest_path,
        existing,
        installed,
        registry_name=str(registry.get("name", "")),
    )
    manifest["registry"] = str(args.config)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
