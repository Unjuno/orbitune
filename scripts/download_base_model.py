#!/usr/bin/env python3
"""Download Orbitune base model release assets.

This is a placeholder until `orbitune-tiny-v0` release assets are published.
The script intentionally keeps base weights out of the Git repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download an Orbitune base model release asset.")
    parser.add_argument("--model", default="orbitune-tiny-v0", help="Base model id to download.")
    parser.add_argument("--out", default="models", help="Output directory.")
    args = parser.parse_args()

    target = Path(args.out) / args.model
    target.mkdir(parents=True, exist_ok=True)

    readme = target / "README.md"
    readme.write_text(
        "# " + args.model + "\n\n"
        "Release download is not configured yet. "
        "Place local base model files here once available.\n",
        encoding="utf-8",
    )

    print(f"Created placeholder directory: {target}")
    print("No model weights were downloaded because no release asset URL is configured yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
