from __future__ import annotations

import hashlib
import torch
from torch import nn
import torch.nn.functional as F

from .cache import VisitedTokenCache
from layerwise_lm.model import load_model


class Delta(nn.Module):
    def __init__(self, width, rank, device):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(rank, width, device=device, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.down(x.float()))


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
        return self.base(x) + delta.to(x.dtype)


class RoutedNetwork(nn.Module):
    def __init__(self, base, rank=16):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        self.n_layers = base.config.num_hidden_layers
        self.device = next(base.parameters()).device
        self.dtype = next(base.parameters()).dtype
        width = base.config.hidden_size
        for layer in base.model.layers:
            for name in ("q_proj", "v_proj"):
                original = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, LoRALinear(original, rank))
        self.entry_delta = nn.ModuleList([Delta(width, rank, self.device) for _ in range(self.n_layers)])
        self.memory_delta = nn.ModuleList([Delta(width, rank, self.device) for _ in range(self.n_layers)])
        self.carry_gate = nn.Parameter(torch.full((self.n_layers,), -2.0, device=self.device))
        self.wrap_delta = Delta(width, rank, self.device)
        self.exit_scale = nn.Parameter(torch.ones(self.n_layers, width, device=self.device))

    def embedding(self, token):
        ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
        return self.base.model.embed_tokens(ids)

    def block(self, layer, h, cache, position):
        cache.set_positions([position])
        pos = torch.tensor([[position]], device=self.device)
        rotary = self.base.model.rotary_emb(h, pos)
        output = self.base.model.layers[layer](h, attention_mask=None, position_ids=pos,
                    position_embeddings=rotary, past_key_values=cache, use_cache=True)
        return output[0] if isinstance(output, tuple) else output

    def logits(self, layer, h):
        normalized = self.base.model.norm(h).float() * self.exit_scale[layer]
        return self.base.lm_head(normalized.to(self.dtype)).reshape(1, -1)

    @torch.no_grad()
    def prefill(self, context):
        if not context:
            raise ValueError("An episode needs at least one context token")
        cache = VisitedTokenCache(self.n_layers)
        # Full-depth prefill is only for the prompt preceding its last token.
        if len(context) > 1:
            ids = torch.tensor([context[:-1]], device=self.device)
            h = self.base.model.embed_tokens(ids)
            pos = torch.arange(len(context) - 1, device=self.device).unsqueeze(0)
            rotary = self.base.model.rotary_emb(h, pos)
            cache.set_positions(range(len(context) - 1))
            for block in self.base.model.layers:
                output = block(h, attention_mask=None, position_ids=pos,
                               position_embeddings=rotary, past_key_values=cache, use_cache=True)
                h = output[0] if isinstance(output, tuple) else output
        cache.detach_()
        return self.embedding(context[-1]), cache

    def inject(self, layer, token, previous):
        e = self.embedding(token).float()
        memory = self.base.model.norm(previous).float()
        scale = e.square().mean(dim=-1, keepdim=True).sqrt().detach()
        new = e + self.entry_delta[layer](e)
        new = new + self.carry_gate[layer].sigmoid() * scale * (memory + self.memory_delta[layer](memory))
        return new.to(self.dtype)

    def wrap(self, previous, current_token):
        memory = self.base.model.norm(previous).float()
        scale = self.embedding(current_token).float().square().mean(dim=-1, keepdim=True).sqrt().detach()
        return ((memory + self.wrap_delta(memory)) * scale).to(self.dtype)

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def restore_trainable(self, values):
        parameters = dict(self.named_parameters())
        expected = {n for n, p in parameters.items() if p.requires_grad}
        if expected != set(values):
            raise ValueError("Checkpoint architecture mismatch")
        with torch.no_grad():
            for name, value in values.items():
                parameters[name].copy_(value.to(parameters[name]))

    def frozen_hash(self):
        h = hashlib.sha256()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                h.update(name.encode())
                h.update(memoryview(p.detach().contiguous().view(torch.uint8).cpu().numpy()))
        return h.hexdigest()


def load(config):
    return RoutedNetwork(load_model(config), rank=config["rank"])
