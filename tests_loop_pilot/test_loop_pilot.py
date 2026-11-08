import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from loop_pilot.core import profile_episode, aggregate


class DummyCache:
    def detach_(self):
        pass


class Scripted:
    n_layers = 3
    def __init__(self, first_hit_at_loop=None):
        self.first_hit_at_loop = first_hit_at_loop
        self.reentries = 0
        self.current = None
        self.injections = []

    def prefill(self, context):
        return torch.tensor(0.), DummyCache()

    def block(self, layer, h, cache, position):
        if layer == 0:
            self.current = (position, self.reentries)
            if position == 0:
                self.reentries += 1
        return h + 1

    def logits(self, layer, h):
        z = torch.full((1, 16), -10.)
        predicted = 0
        if self.first_hit_at_loop is not None:
            if self.current == (0, self.first_hit_at_loop) and layer == 0:
                predicted = 11
            if self.current[0] == 1 and layer == 0:
                predicted = 12
        z[0, predicted] = 10.
        return z

    def inject(self, layer, token, previous):
        self.injections.append((layer, token))
        return previous + token

    def wrap(self, previous, token):
        return previous + 1


EPISODE = {"episode_id": "test", "document_id": "doc", "context": [7], "targets": [11, 12]}


class LoopPilotTests(unittest.TestCase):
    def test_continues_past_96_visits_until_complete_without_resetting(self):
        network = Scripted(first_hit_at_loop=100)
        episode, rows, _ = profile_episode(network, EPISODE, seconds=0, fingerprint=False)
        self.assertEqual(episode["outcome"], "complete")
        self.assertEqual(episode["observed_wraps"], 100)
        self.assertEqual(episode["visited_blocks"], 302)
        self.assertEqual([r["wraps"] for r in rows], [100, 0])
        self.assertEqual(network.injections, [(0, 11)])

    def test_planned_but_not_executed_wrap_does_not_count(self):
        episode, rows, _ = profile_episode(Scripted(), EPISODE, seconds=0, max_total_visits=3, fingerprint=False)
        self.assertEqual(episode["outcome"], "censored")
        self.assertEqual(episode["observed_wraps"], 0)
        self.assertIsNone(episode["wraps_to_completion"])
        self.assertEqual([r["status"] for r in rows], ["censored", "unreached"])

    def test_actual_reentry_counts_one_loop(self):
        episode, _, _ = profile_episode(Scripted(), EPISODE, seconds=0, max_total_visits=4, fingerprint=False)
        self.assertEqual(episode["observed_wraps"], 1)

    def test_timeout_is_censoring_not_completion_or_failure(self):
        tick = [0.]
        def clock():
            tick[0] += 0.01
            return tick[0]
        episode, rows, _ = profile_episode(Scripted(), EPISODE, seconds=0.05, fingerprint=False, clock=clock)
        self.assertEqual(episode["stop_reason"], "wall_time_watchdog")
        self.assertEqual(episode["outcome"], "censored")
        self.assertIsNone(episode["wraps_to_completion"])
        self.assertNotIn("failed", [r["status"] for r in rows])

    def test_censored_rows_are_excluded_from_completed_mean(self):
        complete, t1, _ = profile_episode(Scripted(2), EPISODE, seconds=0, fingerprint=False)
        censored, t2, _ = profile_episode(Scripted(), EPISODE, seconds=0, max_total_visits=30, fingerprint=False)
        summary = aggregate([complete, censored], t1 + t2, 2)
        self.assertEqual(summary["mean_wraps_to_completion_completed_only"], 2)
        self.assertEqual(summary["completed_episodes"], 1)
        self.assertEqual(summary["censored_episodes"], 1)


if __name__ == "__main__":
    unittest.main()
