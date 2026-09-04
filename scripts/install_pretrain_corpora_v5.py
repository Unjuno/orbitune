"""Install the v5 commercial corpus.

v5 is the v4 commercial source set re-frozen with a single
OpenScore/StringQuartets pin update:

    8fd2169855a1b27586454b183739573962f9d8ca
    -> a3b3813477b06c2c48887f1055f8567dae0c2232

All other v4 source pins are unchanged. The v5 installer passes
the v5 registry explicitly to the existing installer; it does not
introduce a new installation path, just a registry pointer.

Use a fresh --root to materialise a new C:\\ov5 corpus (or any other
target). This script does not mutate C:\\ov3 or C:\\ov4.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_V5_REGISTRY = "configs/pretrain_corpus_commercial_v5.json"
DEFAULT_V4_REGISTRY = "configs/pretrain_corpus_commercial_v4.json"
INSTALLER = "scripts/install_pretrain_corpora.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_V5_REGISTRY)
    parser.add_argument("--root", required=True,
                        help="Fresh install root (e.g. C:\\ov5 or "
                             "data/corpora/commercial_v5). Do not point at "
                             "C:\\ov3 or C:\\ov4; those are immutable.")
    parser.add_argument("--sources", default="all")
    args = parser.parse_args()

    if "ov3" in args.root.lower() or "ov4" in args.root.lower():
        print(
            f"ERROR: refusing to mutate immutable root {args.root!r}. "
            "C:\\ov3 and C:\\ov4 are frozen.",
            file=sys.stderr,
        )
        return 2

    registry_path = Path(args.config)
    if not registry_path.exists():
        print(f"ERROR: registry not found: {registry_path}", file=sys.stderr)
        return 2

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    name = str(payload.get("name", ""))
    if "v5" not in name:
        print(
            f"WARNING: registry name {name!r} does not look like v5; "
            "double-check before continuing.",
            file=sys.stderr,
        )

    cmd = [
        sys.executable,
        INSTALLER,
        "--config", str(registry_path),
        "--root", str(args.root),
        "--sources", args.sources,
    ]
    print(f"[install_v5] running: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
