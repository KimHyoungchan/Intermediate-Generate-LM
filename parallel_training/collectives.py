from __future__ import annotations

import hashlib
import random
import torch
import torch.distributed as dist


def episode_rounds(episodes, seed, epoch, world_size):
    """Shuffle once globally, then assign at most one unique episode per rank.

    The last round may contain fewer episodes than ranks. No padding or drops.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive")
    order = list(episodes)
    random.Random(seed + epoch).shuffle(order)
    for start in range(0, len(order), world_size):
        yield order[start:start + world_size]


def parameter_hash(parameters):
    result = hashlib.sha256()
    for parameter in parameters:
        result.update(memoryview(parameter.detach().contiguous().view(torch.uint8).cpu().numpy()))
    return result.hexdigest()


class Collectives:
    def __init__(self, device):
        self.device = torch.device(device)
        self.distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1

    def sum_values(self, values):
        tensor = torch.tensor(values, dtype=torch.int64, device=self.device)
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()

    def gather(self, value):
        if not self.distributed:
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def barrier(self):
        if self.distributed:
            dist.barrier()

    @torch.no_grad()
    def broadcast_parameters(self, parameters):
        if not self.distributed:
            return
        packed = torch.cat([p.detach().reshape(-1) for p in parameters])
        dist.broadcast(packed, src=0)
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.copy_(packed[offset:offset + size].view_as(parameter))
            offset += size

    @torch.no_grad()
    def average_gradients(self, parameters, global_visits):
        """Sum unnormalized local CE gradients / total visited-head losses.

        Every rank participates for every parameter, including unused heads and
        ranks with no final episode. Globally unused parameters keep grad=None
        so AdamW does not apply stale momentum to them.
        """
        if global_visits <= 0:
            raise ValueError("Cannot average zero visits")
        used = torch.tensor([p.grad is not None for p in parameters],
                            dtype=torch.int32, device=self.device)
        packed = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                            for p in parameters])
        if self.distributed:
            dist.all_reduce(used, op=dist.ReduceOp.MAX)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(global_visits)
        offset = 0
        for parameter, active in zip(parameters, used.tolist()):
            size = parameter.numel()
            if active:
                # Clone so .grad does not retain the entire packed buffer.
                parameter.grad = packed[offset:offset + size].view_as(parameter).clone()
            else:
                parameter.grad = None
            offset += size
