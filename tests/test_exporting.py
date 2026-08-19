import torch

from orbitune.exporting import capture_exported_program
from orbitune.model import OrbituneConfig, OrbituneGPT


def test_dynamic_export_capture_accepts_different_sequence_length():
    model = OrbituneGPT(
        OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0)
    ).eval()
    exported = capture_exported_program(model, example_seq_len=8)
    logits = exported.module()(torch.zeros((1, 5), dtype=torch.long))
    assert logits.shape == (1, 5, 32)
