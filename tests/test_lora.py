from pathlib import Path

import pytest
import torch

from orbitune.lora import LoRAConfig, LoRALinear, inject_lora, load_adapter, save_adapter, trainable_parameter_count
from orbitune.model import OrbituneConfig, OrbituneGPT


def _tiny_model() -> OrbituneGPT:
    return OrbituneGPT(OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0))


def test_lora_targets_only_q_and_v():
    model = _tiny_model()
    replaced = inject_lora(model, LoRAConfig(rank=4, alpha=8.0))
    assert any(name.endswith("q_proj") for name in replaced)
    assert any(name.endswith("v_proj") for name in replaced)
    assert trainable_parameter_count(model) == 2 * (4 * 32 + 32 * 4)


def test_adapter_save_load_changes_logits_and_pins_base_hash(tmp_path: Path):
    torch.manual_seed(0)
    base = _tiny_model().eval()
    base_path = tmp_path / "base.pt"
    base.save_checkpoint(base_path)

    adapted = OrbituneGPT.load_checkpoint(base_path).eval()
    cfg = LoRAConfig(rank=2, alpha=2.0)
    inject_lora(adapted, cfg)
    for module in adapted.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.lora_b.fill_(0.01)
    adapter_path = tmp_path / "adapter.safetensors"
    save_adapter(adapted, adapter_path, cfg)

    reloaded = OrbituneGPT.load_checkpoint(base_path).eval()
    load_adapter(reloaded, adapter_path)
    x = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        base_logits, _ = base(x)
        adapted_logits, _ = reloaded(x)
    assert not torch.allclose(base_logits, adapted_logits)

    other_base = _tiny_model().eval()
    other_path = tmp_path / "other-base.pt"
    other_base.save_checkpoint(other_path)
    incompatible = OrbituneGPT.load_checkpoint(other_path).eval()
    with pytest.raises(ValueError, match="different Base checkpoint"):
        load_adapter(incompatible, adapter_path)
