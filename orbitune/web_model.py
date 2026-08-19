from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from orbitune.compat import ADAPTER_FORMAT_ABI, validate_sha256
from orbitune.model import OrbituneGPT

WEB_LORA_RANK = 4
WEB_LORA_TARGETS = ("q_proj", "v_proj")


class ExternalLoRALogitsModel(nn.Module):
    """One Base graph with LoRA matrices supplied as runtime inputs."""

    def __init__(self, model: OrbituneGPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, lora_scale: torch.Tensor) -> torch.Tensor:
        model = self.model
        _, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device)
        x = model.drop(model.token_emb(input_ids) + model.pos_emb(positions)[None, :, :])
        scale = lora_scale[0]
        for layer_index, block in enumerate(model.blocks):
            h = block.ln1(x)
            attn = block.attn
            q = attn.q_proj(h)
            q_delta = (h @ lora_a[layer_index, 0].transpose(0, 1)) @ lora_b[layer_index, 0].transpose(0, 1)
            q = q + q_delta * scale
            k = attn.k_proj(h)
            v = attn.v_proj(h)
            v_delta = (h @ lora_a[layer_index, 1].transpose(0, 1)) @ lora_b[layer_index, 1].transpose(0, 1)
            v = v + v_delta * scale
            batch, time, channels = q.shape
            q = q.view(batch, time, attn.n_head, attn.head_dim).transpose(1, 2)
            k = k.view(batch, time, attn.n_head, attn.head_dim).transpose(1, 2)
            v = v.view(batch, time, attn.n_head, attn.head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(batch, time, channels)
            x = x + attn.out_proj(y)
            x = x + block.mlp(block.ln2(x))
        return model.lm_head(model.ln_f(x))


def empty_web_lora(model: OrbituneGPT, *, rank: int = WEB_LORA_RANK) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layers = model.config.n_layer
    hidden = model.config.n_embd
    a = torch.zeros((layers, 2, rank, hidden), dtype=torch.float32)
    b = torch.zeros((layers, 2, hidden, rank), dtype=torch.float32)
    scale = torch.ones((1,), dtype=torch.float32)
    return a, b, scale


def pack_adapter_for_web(model: OrbituneGPT, adapter_path: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(adapter_path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != ADAPTER_FORMAT_ABI:
        raise ValueError("unsupported adapter format")
    adapter_base_sha = str(metadata.get("base_sha256", "")).lower()
    model_base_sha = getattr(model, "base_sha256", None)
    if not validate_sha256(adapter_base_sha):
        raise ValueError("adapter is missing a valid base_sha256")
    if not isinstance(model_base_sha, str) or adapter_base_sha != model_base_sha.lower():
        raise ValueError("adapter was trained for a different Base checkpoint")
    rank = int(metadata["rank"])
    if rank != WEB_LORA_RANK:
        raise ValueError(f"web adapter ABI requires LoRA rank {WEB_LORA_RANK}, got {rank}")
    targets = tuple(json.loads(metadata["target_modules"]))
    if targets != WEB_LORA_TARGETS:
        raise ValueError(f"web runtime requires targets {WEB_LORA_TARGETS}, got {targets}")
    alpha = float(metadata["alpha"])
    state = load_file(str(path), device="cpu")
    a, b, _ = empty_web_lora(model, rank=rank)
    for layer_index in range(model.config.n_layer):
        for target_index, target in enumerate(WEB_LORA_TARGETS):
            prefix = f"blocks.{layer_index}.attn.{target}"
            key_a = f"{prefix}.lora_a"
            key_b = f"{prefix}.lora_b"
            if key_a not in state or key_b not in state:
                raise ValueError(f"adapter is missing {key_a} or {key_b}")
            a[layer_index, target_index].copy_(state[key_a].float())
            b[layer_index, target_index].copy_(state[key_b].float())
    return a, b, torch.tensor([alpha / rank], dtype=torch.float32)
