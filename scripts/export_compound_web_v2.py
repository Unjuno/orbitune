from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_web_export import (
    export_compound_web_v2,
    export_report,
    sha256_file,
    verify_native_decoder_parity,
    verify_native_stream_parity,
    verify_onnxruntime_parity,
    write_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Orbitune Compound native-stream V2 Web graphs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--native-parity-steps", type=int, default=96)
    parser.add_argument("--ort-parity-steps", type=int, default=24)
    parser.add_argument("--skip-onnxruntime", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = Path(args.checkpoint)
    actual_sha = sha256_file(checkpoint)
    if args.checkpoint_sha256 and actual_sha != args.checkpoint_sha256.lower():
        raise SystemExit(f"checkpoint SHA-256 mismatch: {actual_sha} != {args.checkpoint_sha256.lower()}")
    model, payload = CompoundHierarchicalGPT.load_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    native_stream = verify_native_stream_parity(model, steps=args.native_parity_steps)
    native_decoder = verify_native_decoder_parity(model)
    stream_path, decoder_path = export_compound_web_v2(model, args.out_dir, opset=args.opset)
    ort_parity = None if args.skip_onnxruntime else verify_onnxruntime_parity(
        model, stream_path, decoder_path, steps=args.ort_parity_steps
    )
    report = export_report(
        model, stream_path, decoder_path,
        checkpoint_sha256=actual_sha,
        checkpoint_source_commit=payload.get("source_commit"),
        native_stream=native_stream,
        native_decoder=native_decoder,
        ort_parity=ort_parity,
    )
    report_path = Path(args.out_dir) / "export-report.json"
    write_report(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
