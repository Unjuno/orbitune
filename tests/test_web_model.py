import torch

from orbitune.lora import LoRAConfig, inject_lora, save_adapter
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.web_model import ExternalLoRALogitsModel, empty_web_lora, pack_adapter_for_web


def test_zero_external_lora_matches_base():
    torch.manual_seed(7)
    model = OrbituneGPT(
        OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=2, n_embd=32, n_head=4, dropout=0.0)
    ).eval()
    input_ids = torch.randint(0, 32, (1, 8))
    a, b, scale = empty_web_lora(model)
    with torch.no_grad():
        base_logits, _ = model(input_ids)
        web_logits = ExternalLoRALogitsModel(model).eval()(input_ids, a, b, scale)
    assert torch.allclose(base_logits, web_logits, atol=1e-5, rtol=1e-5)


def test_saved_adapter_can_be_packed_as_runtime_inputs(tmp_path):
    torch.manual_seed(11)
    adapted = OrbituneGPT(
        OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=2, n_embd=32, n_head=4, dropout=0.0)
    ).eval()
    cfg = LoRAConfig(rank=4, alpha=8.0, dropout=0.0)
    inject_lora(adapted, cfg)
    for name, parameter in adapted.named_parameters():
        if name.endswith("lora_b"):
            parameter.data.normal_(mean=0.0, std=0.02)
    adapter_path = tmp_path / "adapter.safetensors"
    save_adapter(adapted, adapter_path, cfg)

    base_shape = OrbituneGPT(
        OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=2, n_embd=32, n_head=4, dropout=0.0)
    ).eval()
    a, b, scale = pack_adapter_for_web(base_shape, adapter_path)
    assert a.shape == (2, 2, 4, 32)
    assert b.shape == (2, 2, 32, 4)
    assert float(scale.item()) == 2.0
    assert torch.count_nonzero(b) > 0
