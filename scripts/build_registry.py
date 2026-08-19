from __future__ import annotations

import argparse
import json

from orbitune.registry import build_web_adapter_assets, write_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Orbitune bundled adapter registry")
    parser.add_argument("--adapters", default="adapters")
    parser.add_argument("--out", default="registry/adapters.json")
    parser.add_argument("--web-root", help="also copy browser adapter assets into this web root")
    args = parser.parse_args()

    if args.web_root:
        registry = build_web_adapter_assets(args.web_root, args.adapters)
    else:
        registry = write_registry(args.out, args.adapters)
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
