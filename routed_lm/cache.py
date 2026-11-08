from __future__ import annotations


class VisitedTokenCache:
    """Per-layer latest K/V per visited input position, never a fabricated skipped state.

    A repeated visit to the same input token REPLACES its K/V. Old tensors are
    never modified in place, so autograd can retain the previous recurrence.
    """
    def __init__(self, layers):
        self.keys = [None] * layers
        self.values = [None] * layers
        self.positions = [[] for _ in range(layers)]
        self.current_positions = []

    def set_positions(self, positions):
        positions = list(positions)
        if not positions or positions != sorted(set(positions)):
            raise ValueError("Positions must be nonempty, unique and increasing")
        self.current_positions = positions

    def update(self, key, value, layer_idx, cache_kwargs=None):
        import torch
        positions = self.current_positions
        if key.shape[-2] != len(positions):
            raise ValueError("Cache position/key length mismatch")
        old = self.positions[layer_idx]
        if not old:
            new_key, new_value, new_positions = key, value, list(positions)
        elif len(positions) != 1:
            raise ValueError("Multi-token prefill is only allowed on an empty layer cache")
        elif positions[0] == old[-1]:
            new_key = torch.cat([self.keys[layer_idx][..., :-1, :], key], dim=-2)
            new_value = torch.cat([self.values[layer_idx][..., :-1, :], value], dim=-2)
            new_positions = list(old)
        elif positions[0] > old[-1]:
            new_key = torch.cat([self.keys[layer_idx], key], dim=-2)
            new_value = torch.cat([self.values[layer_idx], value], dim=-2)
            new_positions = [*old, positions[0]]
        else:
            raise ValueError("Future cache access or nonmonotonic token position")
        self.keys[layer_idx], self.values[layer_idx] = new_key, new_value
        self.positions[layer_idx] = new_positions
        assert max(new_positions) <= max(positions)
        return new_key, new_value

    def detach_(self):
        self.keys = [x.detach() if x is not None else None for x in self.keys]
        self.values = [x.detach() if x is not None else None for x in self.values]
