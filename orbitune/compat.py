from __future__ import annotations

import hashlib
from pathlib import Path

# Default official Base id used by scaffolding and legacy helpers only.
# It is NOT a global compatibility singleton: contributed Bases may coexist.
DEFAULT_BASE_MODEL_ID = "orbitune-base"
BASE_MODEL_ID = DEFAULT_BASE_MODEL_ID  # backward-compatible alias
BASE_PARAMETER_COUNT = 2_945_760

# Protocol / ABI identifiers. A Base declares which ABI it implements.
ARCHITECTURE_ABI = "orbitune-midi-gpt-v0"
TOKENIZER_ABI = "theory-remi-v0"
ADAPTER_FORMAT_ABI = "orbitune-lora-v0"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str) -> bool:
    value = value.lower()
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
