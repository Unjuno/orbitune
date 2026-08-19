from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from orbitune.model import OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab

BASE_MODEL_ID = "orbitune-tiny-v0"
BASE_ARCHITECTURE = "orbitune-midi-gpt-v0"
BASE_TOKENIZER = "theory-remi-v0"
BASE_PARAMETER_COUNT = 2_945_760
BASE_RELEASE_ASSETS = {
    "checkpoint": "orbitune-tiny-v0.pt",
    "web_onnx": "orbitune-tiny-v0-web.onnx",
    "manifest": "orbitune-tiny-v0-manifest.json",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_base_checkpoint(path: str | Path) -> OrbituneGPT:
    model = OrbituneGPT.load_checkpoint(path, map_location="cpu").eval()
    cfg = model.config
    vocab = TheoryRemiVocab()
    errors: list[str] = []
    if model.architecture != BASE_ARCHITECTURE:
        errors.append(f"architecture={model.architecture!r}")
    if model.parameter_count() != BASE_PARAMETER_COUNT:
        errors.append(f"parameters={model.parameter_count()}")
    if cfg.vocab_size != len(vocab):
        errors.append(f"vocab_size={cfg.vocab_size}")
    if cfg.max_seq_len != 512:
        errors.append(f"max_seq_len={cfg.max_seq_len}")
    if (cfg.n_layer, cfg.n_embd, cfg.n_head) != (4, 240, 4):
        errors.append(f"transformer={(cfg.n_layer, cfg.n_embd, cfg.n_head)}")
    if errors:
        raise ValueError("checkpoint is not the fixed Orbitune v0 base: " + ", ".join(errors))
    return model


def _artifact(path: Path, *, url: str) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "url": url,
    }


def package_base_release(
    base: str | Path,
    web_onnx: str | Path,
    out_dir: str | Path,
    *,
    repository: str = "Unjuno/orbitune",
    release_tag: str = BASE_MODEL_ID,
) -> dict[str, object]:
    base = Path(base)
    web_onnx = Path(web_onnx)
    if not base.is_file():
        raise FileNotFoundError(base)
    if not web_onnx.is_file():
        raise FileNotFoundError(web_onnx)
    model = validate_base_checkpoint(base)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = out_dir / BASE_RELEASE_ASSETS["checkpoint"]
    onnx_out = out_dir / BASE_RELEASE_ASSETS["web_onnx"]
    shutil.copy2(base, checkpoint_out)
    shutil.copy2(web_onnx, onnx_out)

    release_root = f"https://github.com/{repository}/releases/download/{release_tag}"
    manifest: dict[str, object] = {
        "schema_version": "0.1.0",
        "model_id": BASE_MODEL_ID,
        "architecture": BASE_ARCHITECTURE,
        "tokenizer": BASE_TOKENIZER,
        "parameters": BASE_PARAMETER_COUNT,
        "config": asdict(model.config),
        "release": {"repository": repository, "tag": release_tag},
        "artifacts": {
            "checkpoint": _artifact(
                checkpoint_out,
                url=f"{release_root}/{BASE_RELEASE_ASSETS['checkpoint']}",
            ),
            "web_onnx": _artifact(
                onnx_out,
                url=f"{release_root}/{BASE_RELEASE_ASSETS['web_onnx']}",
            ),
        },
    }
    manifest_path = out_dir / BASE_RELEASE_ASSETS["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    runtime_config = {
        "model_url": manifest["artifacts"]["web_onnx"]["url"],
        "model_sha256": manifest["artifacts"]["web_onnx"]["sha256"],
        "execution_providers": ["wasm"],
    }
    (out_dir / "runtime-config.json").write_text(
        json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
