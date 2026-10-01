from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "v4_source_census.py"
spec = importlib.util.spec_from_file_location("v4_source_census", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# CPDL intermittently rejects generic cloud-runner user agents / hostnames.
# Keep this disposable audit fail-closed, but try the public mirrors with a
# normal browser UA before declaring the metadata endpoint unavailable.
mod.USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

ENDPOINTS = [
    "https://www.cpdl.org/wiki/api.php",
    "https://www2.cpdl.org/wiki/api.php",
    "https://test.cpdl.org/wiki/api.php",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    failures: list[str] = []
    namespace = None
    for endpoint in ENDPOINTS:
        mod.CPDL_API = endpoint
        try:
            namespace = mod._cpdl_edition_namespace()
        except Exception as exc:  # audit endpoint probe
            failures.append(f"{endpoint}: {type(exc).__name__}: {exc}")
            print(f"CPDL_ENDPOINT_FAIL {failures[-1]}", flush=True)
            continue
        print(f"CPDL_ENDPOINT_OK {endpoint} namespace={namespace}", flush=True)
        break

    if namespace is None:
        raise SystemExit("No CPDL MediaWiki API endpoint was reachable: " + " | ".join(failures))

    # run_cpdl will resolve the namespace again against the selected endpoint.
    result = mod.run_cpdl(args)
    result["api_endpoint"] = mod.CPDL_API
    mod._write_result(result, args.output)


if __name__ == "__main__":
    main()
