from __future__ import annotations

import argparse
import json
import time

import torch

from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Orbitune full-context autoregressive inference")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()

    if args.tokens <= 0:
        raise SystemExit("--tokens must be positive")
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    if args.checkpoint:
        model = OrbituneGPT.load_checkpoint(args.checkpoint, map_location=args.device)
    else:
        torch.manual_seed(1234)
        model = OrbituneGPT(OrbituneConfig(vocab_size=len(TheoryRemiVocab())))
    model = model.to(args.device).eval()

    input_ids = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    # Warmup keeps one-time kernel initialization out of the measured loop.
    with torch.no_grad():
        model(input_ids)

    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.tokens):
            context = input_ids[:, -model.config.max_seq_len :]
            logits, _ = model(context)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
    elapsed = time.perf_counter() - started

    report = {
        "model": "orbitune-tiny-v0",
        "parameters": model.parameter_count(),
        "device": args.device,
        "torch_threads": torch.get_num_threads(),
        "generated_tokens": args.tokens,
        "elapsed_seconds": elapsed,
        "milliseconds_per_token": elapsed * 1000.0 / args.tokens,
        "tokens_per_second": args.tokens / elapsed,
        "method": "full-context recomputation; no KV cache",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
