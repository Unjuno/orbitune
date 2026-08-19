from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from orbitune.compat import REFERENCE_PARAMETER_COUNT
from orbitune.lora import LoRAConfig
from orbitune.model import OrbituneConfig
from orbitune.tokenizer.vocab import TheoryRemiVocab
from orbitune.training import TrainConfig, train_adapter, train_base


def _write_pattern(path: Path, *, style: str, bars: int = 128, validation: bool = False) -> None:
    if style == "base": roots = [50,55,57,52] if validation else [48,53,55,50]; positions=[0,4,8,12]; duration=4
    else: roots=[47,50,54,52] if validation else [45,48,52,50]; positions=[0,8,12]; duration=8
    intervals=[0,7,12,7]; lines=[]
    for bar in range(bars):
        lines.append("BAR"); root=roots[bar % len(roots)]
        for index, position in enumerate(positions):
            pitch=root+intervals[index % len(intervals)]; velocity=18 if style=="base" else 10+(bar%3); lines.extend([f"POSITION_{position}",f"NOTE_PITCH_{pitch}",f"NOTE_DURATION_{duration}",f"VELOCITY_{velocity}"])
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def _validation_interval(steps:int)->int: return max(1,steps//2)


def main()->None:
    parser=argparse.ArgumentParser(description="CPU smoke test for the ~10M Orbitune reference Base and rank-4 LoRA"); parser.add_argument("--base-steps",type=int,default=20); parser.add_argument("--adapter-steps",type=int,default=20); parser.add_argument("--device",default="cpu"); parser.add_argument("--out",default="smoke-training-report.json"); args=parser.parse_args()
    vocab=TheoryRemiVocab(); model_cfg=OrbituneConfig(vocab_size=len(vocab)); assert model_cfg.n_layer==4 and model_cfg.n_embd==448 and model_cfg.n_head==7 and model_cfg.max_seq_len==1024
    with tempfile.TemporaryDirectory(prefix="orbitune-smoke-") as temp:
        root=Path(temp); base_tokens=root/"base.tokens"; base_validation=root/"base-validation.tokens"; style_tokens=root/"style.tokens"; style_validation=root/"style-validation.tokens"; base_checkpoint=root/"orbitune-reference.pt"; adapter_path=root/"adapter.safetensors"
        _write_pattern(base_tokens,style="base"); _write_pattern(base_validation,style="base",bars=32,validation=True); _write_pattern(style_tokens,style="style"); _write_pattern(style_validation,style="style",bars=32,validation=True)
        base_report=train_base([base_tokens],base_checkpoint,model_cfg=model_cfg,train_cfg=TrainConfig(steps=args.base_steps,batch_size=2,seq_len=64,learning_rate=5e-4,device=args.device,validation_interval=_validation_interval(args.base_steps)),validation_token_paths=[base_validation])
        if base_report["parameters"] != REFERENCE_PARAMETER_COUNT: raise RuntimeError(f"unexpected reference parameter count: {base_report['parameters']}")
        adapter_report=train_adapter(base_checkpoint,[style_tokens],adapter_path,lora_cfg=LoRAConfig(rank=4,alpha=8.0),train_cfg=TrainConfig(steps=args.adapter_steps,batch_size=2,seq_len=64,learning_rate=1e-3,device=args.device,validation_interval=_validation_interval(args.adapter_steps)),validation_token_paths=[style_validation])
        if adapter_report["trainable_parameters"] != 28_672: raise RuntimeError(f"unexpected reference adapter parameter count: {adapter_report['trainable_parameters']}")
        report={"purpose":"pipeline smoke test only; not a music-quality benchmark","model":"orbitune-reference-10m","base":base_report,"adapter":adapter_report,"base_checkpoint_bytes":base_checkpoint.stat().st_size,"adapter_bytes":adapter_path.stat().st_size}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
