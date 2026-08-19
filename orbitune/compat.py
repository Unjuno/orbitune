from __future__ import annotations

import hashlib
from pathlib import Path

# Public Base identity. This name is not a rolling model version.
# Once the first official checkpoint is published, its bytes are immutable.
BASE_MODEL_ID = "orbitune-base"
BASE_PARAMETER_COUNT = 2_945_760

# These are protocol / ABI identifiers. They may be versioned independently
# of the immutable Base checkpoint because they describe file/runtime formats.
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
