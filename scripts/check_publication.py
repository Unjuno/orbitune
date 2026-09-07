#!/usr/bin/env python3
"""Read-only checks for source/documentation publication, not weight certification.

Scans only the tracked working tree, not Git history. Never prints matched
credential values. Does not deserialize weights, access the network, or train.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

MODEL_DIR = Path("models/research_nc_aria_gigamidi_v1")
PUBLIC_DOCS = (
    "README.md", "CONTRIBUTING.md", "SECURITY.md", "docs/README.md",
    "docs/PUBLICATION.md", "models/README.md",
    *(f"{MODEL_DIR.as_posix()}/{name}" for name in (
        "README.md", "training_data.md", "usage.md", "reproducibility.md",
    )),
)
BLOCKED_ROOTS = {"raw", "converted", "indexes", "runs", "checkpoints"}
BLOCKED_SUFFIXES = (".i32", ".u8", ".parquet", ".tar", ".tar.gz", ".zip", ".log", ".log.err")
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin")
MAX_TRACKED_BYTES = 95 * 1024 * 1024
TOKEN_PATTERNS = (
    re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
)
LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


def tracked_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return sorted(part.decode("utf-8") for part in result.stdout.split(b"\0") if part)


def scan_paths(root: Path, paths: list[str]) -> list[str]:
    """Return redacted path/rule findings; keep large-file reads bounded."""
    root = root.resolve()
    errors: list[str] = []
    for rel in paths:
        item = PurePosixPath(rel)
        name = item.name.lower()
        path = root / rel
        if item.is_absolute() or ".." in item.parts or not path.resolve().is_relative_to(root):
            errors.append(f"{rel}: path escapes checkout")
            continue
        if not path.is_file():
            errors.append(f"{rel}: tracked file unavailable")
            continue
        parts = {part.lower() for part in item.parts}
        secret_name = (
            name in {".env", "hf_token", ".hf_token", "cookies.txt", "id_rsa", "id_ed25519"}
            or (name.startswith(".env.") and name != ".env.example")
            or name.endswith((".pem", ".key"))
            or bool(parts & {".huggingface", ".aws", ".ssh"})
        )
        if secret_name:
            errors.append(f"{rel}: credential/private-key path must not be tracked")
        legacy_artifact = item.parts[0] in {"bases", "adapters"}
        if (item.parts[0] in BLOCKED_ROOTS and name != ".gitkeep") or name.endswith(BLOCKED_SUFFIXES):
            errors.append(f"{rel}: local data/build output must not be tracked")
        if name.endswith(WEIGHT_SUFFIXES) and not legacy_artifact:
            errors.append(f"{rel}: weight artifact outside reviewed Base/Adapter path")
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            errors.append(f"{rel}: exceeds existing 95 MiB artifact ceiling")
            continue
        if name.endswith(WEIGHT_SUFFIXES):
            continue  # Binary validation belongs to the existing asset pipeline.
        # Overlap catches ordinary tokens that straddle a block boundary.
        tail = b""
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                data = tail + block
                if any(pattern.search(data) for pattern in TOKEN_PATTERNS):
                    errors.append(f"{rel}: possible credential material (value redacted)")
                    break
                tail = data[-512:]
    return errors


def check_doc_links(root: Path, docs: tuple[str, ...] = PUBLIC_DOCS) -> list[str]:
    errors: list[str] = []
    root = root.resolve()
    for rel in docs:
        path = root / rel
        if not path.is_file():
            errors.append(f"{rel}: public documentation missing")
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"[A-Za-z]:[\\/]Users[\\/]", text):
            errors.append(f"{rel}: personal home path in public instructions")
        for target in LINK_RE.findall(text):
            target = target.strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            linked = (path.parent / unquote(parsed.path)).resolve()
            if not linked.is_relative_to(root) or not linked.exists():
                errors.append(f"{rel}: broken local link: {target}")
    return errors


def validate_publication(root: Path) -> list[str]:
    from jsonschema import Draft202012Validator

    schema = json.loads((root / "schemas/research_publication.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    model_dir = root / MODEL_DIR
    record = json.loads((model_dir / "publication.json").read_text(encoding="utf-8"))
    failures = list(Draft202012Validator(schema).iter_errors(record))
    if failures:
        return [f"publication.json: schema violation at {list(error.absolute_path)}" for error in failures]
    errors: list[str] = []
    raw = (model_dir / record["historical_manifest"]).read_bytes()
    blob_sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob_sha != record["historical_manifest_git_blob_sha1"]:
        errors.append("historical manifest: Git blob identity changed")
    historical = json.loads(raw)
    if record["id"] != historical["id"]:
        errors.append("publication model ID differs from historical report")
    for key in ("sha256", "bytes"):
        if record["checkpoint"][key] != historical["checkpoint"][key]:
            errors.append(f"publication checkpoint {key} differs from historical report")
    parent = historical["lineage"]["parent_checkpoint"]
    if (record["lineage"]["parent_id"], record["lineage"]["parent_sha256"]) != (parent["id"], parent["sha256"]):
        errors.append("publication parent identity differs from historical report")
    if not {"aria_midi", "gigamidi"}.issubset(record["lineage"]["restricted_source_ids"]):
        errors.append("both research data sources must remain restricted")
    corpus = record["reported_corpus"]
    if corpus["train_records"] - corpus["train_songs"] != corpus["available_active_next_event_pairs"]:
        errors.append("reported corpus arithmetic does not balance")
    training = record["reported_training"]
    if training["global_step"] - training["parent_step"] != training["new_steps"]:
        errors.append("reported training step arithmetic does not balance")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        paths = tracked_paths(root)
        errors = scan_paths(root, paths) + check_doc_links(root) + validate_publication(root)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, ImportError) as exc:
        # Do not echo arbitrary file contents or credential-bearing subprocess output.
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "hint": "Use a source checkout with dev dependencies installed."}))
        return 2
    print(json.dumps({
        "status": "FAIL" if errors else "PASS", "tracked_files_checked": len(paths),
        "findings": errors, "scope": "tracked_working_tree_and_public_docs_only",
        "checkpoint_bytes_verified": False, "git_history_scanned": False,
    }, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
