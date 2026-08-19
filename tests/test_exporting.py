import torch

from orbitune.exporting import capture_exported_program, capture_web_exported_program
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.web_model import empty_web_lora


def test_dynamic_export_capture_accepts_different_sequence_length():
    model = OrbituneGPT(
        OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0)
    ).eval()
    exported = capture_exported_program(model, example_seq_len=8)
    logits = exported.module()(torch.zeros((1, 5), dtype=torch.long))
    assert logits.shape == (1, 5, 32)


def test_dynamic_web_export_capture_accepts_external_lora_and_different_sequence_length():
    model = OrbituneGPT(
        OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=2, n_embd=32, n_head=4, dropout=0.0)
    ).eval()
    a, b, scale = empty_web_lora(model)
    exported = capture_web_exported_program(model, example_seq_len=8)
    logits = exported.module()(torch.zeros((1, 5), dtype=torch.long), a, b, scale)
    assert logits.shape == (1, 5, 32)
