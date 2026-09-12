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


def build_runtime_config(release: dict, graph_dir: Path) -> dict:
    if release.get("status") != "approved_for_publication":
        raise ValueError("Web release is not approved_for_publication")
    rights = release["rights"]
    if rights.get("redistribution_review") != "completed":
        raise ValueError("Web release redistribution review is not completed")
    if rights.get("commercial_eligible") is not False:
        raise ValueError("A2 Web release must remain noncommercial")
    base = release["base"]
    runtime = release["runtime"]
    graphs = release["graphs"]
    stream_path = graph_dir / graphs["stream"]["filename"]
    decoder_path = graph_dir / graphs["decoder"]["filename"]
    verify_graph(stream_path, graphs["stream"])
    verify_graph(decoder_path, graphs["decoder"])
    publication = release["publication"]
    if publication.get("runtime_publication_status") != "runtime_model_published":
        raise ValueError("Web release publication status is not runtime_model_published")
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
        "variants": [
            {
                "id": release["id"],
                "display_name": release["display_name"],
                "kind": "base",
                "available": True,
                "architecture": base["architecture"],
                "tokenizer": base["tokenizer"],
                "runtime_abi": runtime["abi"],
                "base_checkpoint_sha256": base["checkpoint_sha256"],
                "stream": {"url": graphs["stream"]["url"], "sha256": graphs["stream"]["sha256"]},
                "decoder": {"url": graphs["decoder"]["url"], "sha256": graphs["decoder"]["sha256"]},
                "execution_providers": list(runtime["execution_providers"]),
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reviewed Compound runtime config for a Pages artifact")
    parser.add_argument("--release", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    release = json.loads(Path(args.release).read_text(encoding="utf-8"))
    config = build_runtime_config(release, Path(args.graph_dir))
    Path(args.output).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(config, sort_keys=True))


if __name__ == "__main__":
    main()
