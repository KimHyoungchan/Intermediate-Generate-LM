import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from parallel_training.cli import gpu_list
from parallel_training.collectives import Collectives, episode_rounds


def toy_parameters():
    return [torch.nn.Parameter(torch.tensor([0.5, -0.3])),
            torch.nn.Parameter(torch.tensor([0.2])),
            torch.nn.Parameter(torch.tensor([0.7]))]


def examples(step):
    # Unequal per-rank visit counts; p1 used only by rank 0, then unused globally.
    if step == 0:
        return [(0, 1., 2., True), (0, 3., -1., True), (1, 2., 1., False)]
    # The final round leaves rank 1 idle but it must join gradient reduction.
    return [(0, 2., 4., False)]


def loss_for(parameters, example):
    _, x, y, use_extra = example
    prediction = parameters[0].sum() * x
    if use_extra:
        prediction = prediction + parameters[1].sum()
    return (prediction - y).square()


def distributed_worker(rank, store, destination):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2)
    runtime = Collectives("cpu")
    parameters = toy_parameters()
    if rank == 1:
        with torch.no_grad():
            parameters[0].add_(10.)
    runtime.broadcast_parameters(parameters)
    optimizer = torch.optim.AdamW(parameters, lr=0.02, weight_decay=0)
    for step in range(2):
        batch = [example for example in examples(step) if example[0] == rank]
        if batch:
            torch.stack([loss_for(parameters, example) for example in batch]).sum().backward()
        visits = runtime.sum_values([len(batch)])[0]
        runtime.average_gradients(parameters, visits)
        assert parameters[2].grad is None
        if step == 1:
            assert parameters[1].grad is None
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    Path(destination, f"rank{rank}.json").write_text(json.dumps([p.detach().tolist() for p in parameters]))
    dist.destroy_process_group()


class ParallelTests(unittest.TestCase):
    def test_no_duplicates_or_drops_when_last_round_has_idle_ranks(self):
        data = list(range(7))
        epoch_orders = []
        for epoch in range(2):
            groups = list(episode_rounds(data, 42, epoch, 3))
            self.assertEqual([len(group) for group in groups], [3, 3, 1])
            order = [episode for group in groups for episode in group]
            self.assertEqual(sorted(order), data)
            epoch_orders.append(order)
        self.assertNotEqual(*epoch_orders)
        self.assertEqual(list(episode_rounds(data, 42, 0, 3)), list(episode_rounds(data, 42, 0, 3)))

    def test_forbidden_gpu_and_duplicate_gpu_are_rejected(self):
        import argparse
        for value in ("0", "0,1", "1,1", "4", "", "1,", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=value):
                gpu_list(value)
        self.assertEqual(gpu_list("1,2,3"), [1, 2, 3])
        self.assertEqual(gpu_list("3,1"), [3, 1])

    def test_weighted_distributed_updates_match_centralized_adamw(self):
        parameters = toy_parameters()
        optimizer = torch.optim.AdamW(parameters, lr=0.02, weight_decay=0)
        for step in range(2):
            batch = examples(step)
            torch.stack([loss_for(parameters, example) for example in batch]).mean().backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_worker, args=(str(Path(folder, "store")), folder), nprocs=2, join=True)
            for rank in range(2):
                actual = json.loads(Path(folder, f"rank{rank}.json").read_text())
                for expected, value in zip(parameters, actual):
                    torch.testing.assert_close(torch.tensor(value), expected.detach(), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
