"""CPU-only tests: no model downloads and no CUDA initialization."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from layerwise_lm.common import isolate_gpu
from layerwise_lm.data import articles, windows
from layerwise_lm.metrics import depth_and_stability, paired_delta_ci
from layerwise_lm.model import ResidualReadout


class ResearchTests(unittest.TestCase):
    def test_no_hit_and_unstable_hit(self):
        c = np.array([[0, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=bool)
        first, stable, flip = depth_and_stability(c)
        np.testing.assert_array_equal(first, [2, 0, 2])
        np.testing.assert_array_equal(stable, [4, 0, 0])
        np.testing.assert_array_equal(flip, [True, False, True])

    def test_chunk_targets_are_complete_unique_and_shifted(self):
        for length in (65, 511, 512, 513, 959, 960, 961, 1536):
            seen = []
            tokens = list(range(length))
            for start, chunk, positions in windows(tokens, 512, 64):
                for j in positions:
                    self.assertEqual(chunk[j - 1] + 1, chunk[j])
                    seen.append(start + j)
            self.assertEqual(seen, list(range(64, length)))

    def test_articles_do_not_split_on_section_heading(self):
        rows = [{"text": t} for t in [" = First = \n", "body", " = = Section = = \n", "more", " = Second = \n", "end"]]
        result = list(articles(rows, "test"))
        self.assertEqual(len(result), 2)
        self.assertIn("Section", result[0]["text"])
        self.assertIn("Second", result[1]["text"])

    def test_low_rank_identity_is_trainable(self):
        layer = ResidualReadout(8, "low_rank", 3)
        h = torch.randn(5, 8)
        torch.testing.assert_close(layer(h), h, atol=0, rtol=0)
        layer(h).square().sum().backward()
        self.assertGreater(layer.up.weight.grad.abs().sum().item(), 0)

    def test_cluster_bootstrap_preserves_constant_paired_change(self):
        a = np.array([1., 2., 3., 4., 5.])
        b = a - 2
        docs = np.array(["a", "a", "a", "b", "b"])
        low, high = paired_delta_ci(a, b, docs, draws=100)
        self.assertAlmostEqual(low, -2)
        self.assertAlmostEqual(high, -2)

    def test_physical_gpu_zero_is_rejected_before_query(self):
        with self.assertRaises(ValueError):
            isolate_gpu(0)


if __name__ == "__main__":
    unittest.main()
