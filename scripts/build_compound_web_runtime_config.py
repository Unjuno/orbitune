from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_graph(path: Path, spec: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"{path}: byte-size mismatch {path.stat().st_size} != {spec['bytes']}")
    actual = sha256_file(path)
    expected = str(spec["sha256"]).lower()
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 mismatch {actual} != {expected}")


def _validate_release(release: dict, graph_dir: Path) -> tuple[dict, dict, dict, dict]:
    if release.get("status") != "approved_for_publication":
        raise ValueError("Web release is not approved_for_publication")
    rights = release["rights"]
    if rights.get("redistribution_review") != "completed":
        raise ValueError("Web release redistribution review is not completed")
    if rights.get("commercial_eligible") is not False:
        raise ValueError("A2 Web release must remain noncommercial")
    if rights.get("distribution_scope") != "research-noncommercial":
        raise ValueError("A2 Web release must preserve research-noncommercial scope")
    publication = release["publication"]
    if publication.get("runtime_publication_status") != "runtime_model_published":
        raise ValueError("Web release publication status is not runtime_model_published")

    base = release["base"]
    runtime = release["runtime"]
    graphs = release["graphs"]
    verify_graph(graph_dir / graphs["stream"]["filename"], graphs["stream"])
    verify_graph(graph_dir / graphs["decoder"]["filename"], graphs["decoder"])
    return base, runtime, graphs, rights


def _variant_from_release(
    release: dict,
    graph_dir: Path,
    *,
    canonical_base: dict,
    canonical_runtime_abi: str,
) -> dict:
    base, runtime, graphs, _rights = _validate_release(release, graph_dir)
    for field in ("model_id", "checkpoint_sha256", "architecture", "tokenizer"):
        if base.get(field) != canonical_base.get(field):
            raise ValueError(f"Web variant Base {field} does not match the canonical published Base")
    if runtime.get("abi") != canonical_runtime_abi:
        raise ValueError("Web variant runtime ABI does not match the canonical published Base")

    kind = str(release.get("kind", "base"))
    if kind not in {"base", "lora-premerged"}:
        raise ValueError("Web release kind must be base or lora-premerged")
    variant = {
        "id": release["id"],
        "display_name": release["display_name"],
        "kind": kind,
        "available": True,
        "architecture": base["architecture"],
        "tokenizer": base["tokenizer"],
        "runtime_abi": runtime["abi"],
        "base_checkpoint_sha256": base["checkpoint_sha256"],
        "stream": {"url": graphs["stream"]["url"], "sha256": graphs["stream"]["sha256"]},
        "decoder": {"url": graphs["decoder"]["url"], "sha256": graphs["decoder"]["sha256"]},
        "execution_providers": list(runtime["execution_providers"]),
    }
    if kind == "lora-premerged":
        adapter = release.get("adapter")
        if not isinstance(adapter, dict) or not str(adapter.get("id", "")).strip():
            raise ValueError("lora-premerged Web release requires adapter.id")
        variant["adapter_id"] = str(adapter["id"])
    elif release.get("adapter") is not None:
        raise ValueError("Base Web release must not declare adapter metadata")
    return variant


def build_runtime_config(
    release: dict,
    graph_dir: Path,
    *,
    additional_releases: list[tuple[dict, Path]] | None = None,
) -> dict:
    base, runtime, _graphs, rights = _validate_release(release, graph_dir)
    if release.get("kind", "base") != "base":
        raise ValueError("primary Web release must be the Base variant")
    variants = [
        _variant_from_release(
            release,
            graph_dir,
            canonical_base=base,
            canonical_runtime_abi=runtime["abi"],
        )
    ]
    for extra_release, extra_graph_dir in additional_releases or []:
        if extra_release.get("kind") != "lora-premerged":
            raise ValueError("additional Web releases must be lora-premerged variants")
        variants.append(
            _variant_from_release(
                extra_release,
                extra_graph_dir,
                canonical_base=base,
                canonical_runtime_abi=runtime["abi"],
            )
        )
    ids = [variant["id"] for variant in variants]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate Web release variant id")
    return {
        "schema_version": "0.2.0",
        "runtime_abi": runtime["abi"],
        "publication_status": "runtime_model_published",
        "model_id": base["model_id"],
        "checkpoint_sha256": base["checkpoint_sha256"],
        "architecture": base["architecture"],
        "tokenizer": base["tokenizer"],
        "commercial_eligible": False,
        "distribution_scope": rights["distribution_scope"],
        "license_policy": "research-nc",
        "redistribution_review": "completed",
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reviewed Compound runtime config for a Pages artifact")
    parser.add_argument("--release", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--additional-release", action="append", default=[])
    parser.add_argument("--additional-graph-dir", action="append", default=[])
    args = parser.parse_args()
    if len(args.additional_release) != len(args.additional_graph_dir):
        raise SystemExit("--additional-release and --additional-graph-dir must be supplied in matching pairs")
    release = json.loads(Path(args.release).read_text(encoding="utf-8"))
    extras = [
        (json.loads(Path(path).read_text(encoding="utf-8")), Path(graph_dir))
        for path, graph_dir in zip(args.additional_release, args.additional_graph_dir, strict=True)
    ]
    config = build_runtime_config(release, Path(args.graph_dir), additional_releases=extras)
    Path(args.output).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(config, sort_keys=True))


if __name__ == "__main__":
    main()
