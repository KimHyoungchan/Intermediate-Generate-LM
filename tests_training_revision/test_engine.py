import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import sys
import unittest
import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from training_revision.engine import walk_training


class Cache:
    def detach_(self):
        pass


class Scripted:
    n_layers = 3
    def __init__(self, predictions=None):
        self.predictions = predictions or {}
        self.injections = []
    def prefill(self, context):
        return torch.tensor(0.), Cache()
    def block(self, layer, h, cache, position):
        self.current = (position, layer)
        return h + 1
    def logits(self, layer, h):
        result = torch.full((1, 32), -5.)
        result[0, self.predictions.get(self.current, 0)] = 5.
        return result
    def inject(self, layer, token, previous):
        self.injections.append((layer, token))
        return previous + token
    def wrap(self, previous, token):
        return previous + 100


EPISODE = {"episode_id": "test", "context": [7], "targets": [11, 12, 13]}


class RevisionTests(unittest.TestCase):
    def test_failed_tokens_do_not_discard_later_training_labels(self):
        network = Scripted()
        events = [e for e, _, _ in walk_training(network, EPISODE, "next_token", "global")]
        self.assertEqual([e["target_id"] for e in events], [11]*3 + [12]*3 + [13]*3)
        self.assertEqual(network.injections, [(0, 11), (0, 12)])
        self.assertEqual(events[-1]["action"], "complete_forced")
        self.assertTrue(all(e["iteration"] <= e["iteration_limit"] for e in events))

    def test_per_token_i_plus_one_iteration_limit_is_distinct_from_global(self):
        events = [e for e, _, _ in walk_training(Scripted(), EPISODE, "next_token", "per_token")]
        self.assertEqual([sum(e["target_index"]==i for e in events) for i in range(3)], [3, 6, 9])

    def test_early_hit_puts_next_token_at_same_layer(self):
        network = Scripted({(0, 1): 11, (1, 1): 12})
        events = [e for e, _, _ in walk_training(network, EPISODE, "next_token", "global")]
        self.assertEqual([e["layer"] for e in events[:3]], [1, 2, 2])
        self.assertEqual(events[2]["input_token_id"], 11)

    def test_global_iteration_is_not_reset_after_forced_next_token(self):
        network = Scripted({(0, 1): 11})
        events = [e for e, _, _ in walk_training(network, EPISODE, "next_token", "global")]
        third = [e for e in events if e["target_index"] == 2]
        self.assertEqual(len(third), 3)
        self.assertTrue(all(e["global_iteration"] == 3 for e in third))
        self.assertEqual(third[-1]["action"], "complete_forced")

    def test_same_input_objective_is_explicit_and_does_not_shift_labels(self):
        network = Scripted()
        events = [e for e, _, _ in walk_training(network, EPISODE, "input_token", "global")]
        self.assertEqual(events[0]["input_token_id"], 11)
        self.assertEqual(events[0]["target_id"], 11)
        self.assertEqual(network.injections, [(0, 12), (0, 13)])
        self.assertTrue(all(e["input_token_id"]==e["target_id"] for e in events))

    def test_next_token_objective_keeps_current_label_out_of_input(self):
        events = [e for e, _, _ in walk_training(Scripted(), EPISODE, "next_token", "global")]
        self.assertEqual(events[0]["input_token_id"], 7)
        self.assertEqual(events[0]["target_id"], 11)

    def test_failure_transition_receives_gradient_from_following_token(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from routed_lm.network import RoutedNetwork
        torch.manual_seed(2)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=128)
        model = Qwen2ForCausalLM(config).eval()
        model.set_attn_implementation("sdpa")
        network = RoutedNetwork(model, rank=2)
        context = [1, 2, 3, 4]
        with torch.no_grad():
            h, cache = network.prefill(context)
            predicted = set()
            for l in range(3):
                h = network.block(l, h, cache, 3)
                predicted.add(int(network.logits(l, h).argmax()))
            first = next(i for i in range(64) if i not in predicted)
        ep = {"episode_id": "gradient", "context": context, "targets": [first, 17]}
        losses = []
        failed_hidden = None
        for event, loss, state in walk_training(network, ep, "next_token", "global"):
            if event["target_index"] == 0 and event["forced"]:
                failed_hidden = state.h
                failed_hidden.retain_grad()
            if event["target_index"] == 1:
                losses.append(loss)
        torch.stack(losses).sum().backward()
        self.assertIsNotNone(failed_hidden)
        self.assertGreater(failed_hidden.grad.abs().sum().item(), 0)
        groups = {"entry": 0., "wrap": 0., "lora": 0.}
        for name, parameter in network.named_parameters():
            if parameter.grad is None: continue
            value = parameter.grad.abs().sum().item()
            if "entry" in name or "memory" in name: groups["entry"] += value
            if "wrap" in name: groups["wrap"] += value
            if "lora" in name: groups["lora"] += value
        self.assertTrue(all(value > 0 for value in groups.values()), groups)


if __name__ == "__main__":
    unittest.main()
