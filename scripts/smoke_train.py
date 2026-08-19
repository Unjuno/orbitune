from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from orbitune.lora import LoRAConfig
from orbitune.model import OrbituneConfig
from orbitune.tokenizer.vocab import TheoryRemiVocab
from orbitune.training import TrainConfig, train_adapter, train_base


def _write_pattern(path: Path, *, style: str, bars: int = 128) -> None:
    roots = [48, 53, 55, 50] if style == "base" else [45, 48, 52, 50]
    lines: list[str] = []
    for bar in range(bars):
        lines.append("BAR")
        root = roots[bar % len(roots)]
        positions = [0, 4, 8, 12] if style == "base" else [0, 8]
        intervals = [0, 7, 12, 7]
        for index, position in enumerate(positions):
            pitch = root + intervals[index % len(intervals)]
            duration = 4 if style == "base" else 8
            velocity = 18 if style == "base" else 10 + (bar % 3)
            lines.extend(
                [
                    f"POSITION_{position}",
                    f"NOTE_PITCH_{pitch}",
                    f"NOTE_DURATION_{duration}",
                    f"VELOCITY_{velocity}",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU smoke test for orbitune-tiny-v0 and a rank-4 LoRA")
    parser.add_argument("--base-steps", type=int, default=40)
    parser.add_argument("--adapter-steps", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="smoke-training-report.json")
    args = parser.parse_args()

    vocab = TheoryRemiVocab()
    model_cfg = OrbituneConfig(vocab_size=len(vocab), max_seq_len=512, n_layer=4, n_embd=256, n_head=4, dropout=0.1)

    with tempfile.TemporaryDirectory(prefix="orbitune-smoke-") as temp:
        root = Path(temp)
        base_tokens = root / "base.tokens"
        style_tokens = root / "style.tokens"
        base_checkpoint = root / "orbitune-tiny-v0.pt"
        adapter_path = root / "adapter.safetensors"
        _write_pattern(base_tokens, style="base")
        _write_pattern(style_tokens, style="style")

        base_report = train_base(
            [base_tokens],
            base_checkpoint,
            model_cfg=model_cfg,
            train_cfg=TrainConfig(
                steps=args.base_steps,
                batch_size=4,
                seq_len=64,
                learning_rate=5e-4,
                device=args.device,
            ),
        )
        adapter_report = train_adapter(
            base_checkpoint,
            [style_tokens],
            adapter_path,
            lora_cfg=LoRAConfig(rank=4, alpha=8.0),
            train_cfg=TrainConfig(
                steps=args.adapter_steps,
                batch_size=4,
                seq_len=64,
                learning_rate=1e-3,
                device=args.device,
            ),
        )
        report = {
            "purpose": "pipeline smoke test only; not a music-quality benchmark",
            "model": "orbitune-tiny-v0",
            "base": base_report,
            "adapter": adapter_report,
            "base_checkpoint_bytes": base_checkpoint.stat().st_size,
            "adapter_bytes": adapter_path.stat().st_size,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
