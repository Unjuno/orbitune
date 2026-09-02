from __future__ import annotations

import importlib.util
from pathlib import Path

from orbitune.compound_tbptt_time_vectorized import tbptt_loss as time_vectorized_tbptt_loss

ROOT = Path(__file__).resolve().parents[1]


def _load_trainer():
    path = ROOT / "scripts" / "compound_tbptt_train.py"
    spec = importlib.util.spec_from_file_location("orbitune_compound_tbptt_train_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    trainer = _load_trainer()
    trainer.tbptt_loss = time_vectorized_tbptt_loss
    trainer.main()


if __name__ == "__main__":
    main()
