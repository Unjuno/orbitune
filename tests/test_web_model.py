import pytest
import torch

from orbitune.lora import LoRAConfig, inject_lora, save_adapter
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.web_model import ExternalLoRALogitsModel, empty_web_lora, pack_adapter_for_web


def _checkpoint_model(tmp_path, name: str, *, seed: int) -> OrbituneGPT:
    torch.manual_seed(seed)
    path = tmp_path / name
    OrbituneGPT(OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=2, n_embd=32, n_head=4, dropout=0.0)).save_checkpoint(path)
    return OrbituneGPT.load_checkpoint(path).eval()


def test_zero_external_lora_matches_base():
    torch.manual_seed(7)
    model = OrbituneGPT(OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=2, n_embd=32, n_head=4, dropout=0.0)).eval()
    input_ids = torch.randint(0, 32, (1, 8))
    a, b, scale = empty_web_lora(model)
    with torch.no_grad():
        base_logits, _ = model(input_ids)
        web_logits = ExternalLoRALogitsModel(model).eval()(input_ids, a, b, scale)
    assert torch.allclose(base_logits, web_logits, atol=1e-5, rtol=1e-5)


def test_saved_adapter_can_be_packed_only_for_its_exact_base(tmp_path):
    adapted = _checkpoint_model(tmp_path, "base.pt", seed=11)
    cfg = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)
    inject_lora(adapted, cfg)
    for name, parameter in adapted.named_parameters():
        if name.endswith("lora_b"):
            parameter.data.normal_(mean=0.0, std=0.02)
    adapter_path = tmp_path / "adapter.safetensors"
    save_adapter(adapted, adapter_path, cfg)

    base_shape = OrbituneGPT.load_checkpoint(tmp_path / "base.pt").eval()
    a, b, scale = pack_adapter_for_web(base_shape, adapter_path)
    assert a.shape == (2, 2, 4, 32)
    assert b.shape == (2, 2, 32, 4)
    assert float(scale.item()) == 2.0
    assert torch.count_nonzero(b) > 0

    incompatible = _checkpoint_model(tmp_path, "other.pt", seed=12)
    with pytest.raises(ValueError, match="different Base checkpoint"):
        pack_adapter_for_web(incompatible, adapter_path)
