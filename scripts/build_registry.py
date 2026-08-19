from __future__ import annotations

import argparse
import json

from orbitune.registry import build_web_assets, write_registries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Orbitune Base and Adapter registries")
    parser.add_argument("--adapters", default="adapters")
    parser.add_argument("--bases", default="bases")
    parser.add_argument("--adapter-out", default="registry/adapters.json")
    parser.add_argument("--base-out", default="registry/bases.json")
    parser.add_argument("--web-root", help="also copy browser Base/Adapter assets into this web root")
    args = parser.parse_args()

    if args.web_root:
        bases, adapters = build_web_assets(args.web_root, args.adapters, args.bases)
    else:
        bases, adapters = write_registries(args.adapter_out, args.base_out, args.adapters, args.bases)
    print(json.dumps({"bases": bases, "adapters": adapters}, indent=2))


if __name__ == "__main__":
    main()
