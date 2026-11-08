from __future__ import annotations

import hashlib
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from .common import seed_all


class ResidualReadout(nn.Module):
    def __init__(self, hidden, kind, rank):
        super().__init__()
        self.kind = kind
        if kind == "dense":
            self.delta = nn.Linear(hidden, hidden)
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)
        else:
            self.down = nn.Linear(hidden, rank, bias=False)
            self.up = nn.Linear(rank, hidden)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, h):
        if self.kind == "dense":
            return h + self.delta(h)
        return h + self.up(self.down(h))


def make_readouts(model, config):
    a = config["adapter"]
    return nn.ModuleList([ResidualReadout(model.config.hidden_size, a["kind"], a.get("rank", 64))
                          for _ in range(model.config.num_hidden_layers - 1)]).to("cuda:0", dtype=torch.float32)


def load_model(config):
    seed_all(config["seed"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one isolated CUDA GPU. Use run.py --gpu 1/2/3.")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["model_revision"],
        local_files_only=config.get("local_model_only", True), dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": "cuda:0"},
    ).eval()
    if model.config.model_type != "qwen2":
        raise ValueError("This model adapter supports Qwen2/Qwen2.5 checkpoints only")
    model.requires_grad_(False)
    return model


def weight_hash(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        h.update(memoryview(raw))
    return h.hexdigest()


def raw_hidden(model, ids, positions):
    """Gather only scored positions; no backbone graph or full vocabulary logits."""
    captured = []
    handles = []
    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h[0, positions].detach().clone())
    try:
        for block in model.model.layers:
            handles.append(block.register_forward_hook(hook))
        with torch.no_grad():
            model.model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def logits_for(model, hidden, readout=None):
    if readout is not None:
        hidden = readout(hidden.float()).to(model.dtype)
    return model.lm_head(model.model.norm(hidden))


@torch.inference_mode()
def verify_model(model, ids):
    """Check hook non-interference, final norm indexing, and causal-prefix parity."""
    length = min(ids.shape[1], 64)
    ids = ids[:, :length]
    positions = torch.arange(length, device=ids.device)
    original = model(input_ids=ids, use_cache=False).logits[0]
    hidden = raw_hidden(model, ids, positions)
    observed = logits_for(model, hidden[-1])
    max_error = (original.float() - observed.float()).abs().max().item()
    torch.testing.assert_close(observed, original, rtol=0, atol=0)
    prefix_length = max(1, length // 2)
    prefix = model(input_ids=ids[:, :prefix_length], use_cache=False).logits[0, -1]
    full = original[prefix_length - 1]
    # BF16 GEMM/SDPA kernels can change with sequence shape. Test causal
    # isolation at a fixed shape instead of treating such rounding as leakage.
    perturbed = ids.clone()
    perturbed[:, prefix_length:] = (perturbed[:, prefix_length:] + 1) % model.config.vocab_size
    changed = model(input_ids=perturbed, use_cache=False).logits[0, prefix_length - 1]
    torch.testing.assert_close(changed, full, rtol=0, atol=0)
    return {"final_logits_max_abs_error": max_error,
            "shape_changed_prefix_max_abs_error_bf16": (prefix.float() - full.float()).abs().max().item(),
            "shape_changed_prefix_top1_equal": bool(prefix.argmax() == full.argmax()),
            "causal_suffix_perturbation_max_abs_error": (changed.float() - full.float()).abs().max().item()}
