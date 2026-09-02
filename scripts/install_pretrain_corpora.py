from __future__ import annotations

import argparse
import hashlib
import json
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


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream integrity checksum, not security use
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


def install_hf_midi(source: dict[str, object], target: Path) -> dict[str, object]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Hugging Face source requires: pip install -e ".[corpus]"') from exc

    repo_id = str(source["repo_id"])
    revision = str(source.get("revision", "")).strip()
    if len(revision) != 40:
        raise RuntimeError(f"{repo_id}: registry must pin a full 40-character revision")
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
        else:
            raise SystemExit(f"unsupported source kind: {source.kind}")

    manifest = {
        "registry": str(args.config),
        "registry_name": registry.get("name"),
        "sources": installed,
    }
    (root / "install_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
