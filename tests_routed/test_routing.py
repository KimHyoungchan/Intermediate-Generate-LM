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
from routed_lm.cache import VisitedTokenCache
from routed_lm.engine import walk
from routed_lm.network import RoutedNetwork
from routed_lm.workflow import evaluate_episodes


class EmptyCache:
    def detach_(self):
        pass


class Scripted:
    n_layers = 3
    def __init__(self, predictions):
        self.predictions = predictions
        self.counts = {}
        self.injected = []
        self.wraps = []

    def prefill(self, context):
        return torch.tensor(float(context[-1])), EmptyCache()

    def block(self, layer, h, cache, position):
        key = (position, layer)
        count = self.counts.get(key, 0)
        self.counts[key] = count + 1
        self.current = (position, layer, count)
        return h + 1

    def logits(self, layer, h):
        result = torch.full((1, 20), -10.)
        result[0, self.predictions.get(self.current, 0)] = 10.
        return result

    def inject(self, layer, token, previous):
        self.injected.append((layer, token, float(previous)))
        return previous + token

    def wrap(self, previous, token):
        self.wraps.append((float(previous), token))
        return previous + 100


class RoutingTests(unittest.TestCase):
    def test_correct_token_enters_same_block_without_deeper_old_token_work(self):
        net = Scripted({(0, 1, 0): 11, (1, 1, 0): 12})
        episode = {"episode_id": "x", "context": [7], "targets": [11, 12]}
        events = [e for e, _, _ in walk(net, episode, 6)]
        self.assertEqual([e["layer"] for e in events], [1, 2, 2])
        self.assertEqual([e["input_token_id"] for e in events], [7, 7, 11])
        self.assertEqual([e["target_index"] for e in events], [0, 0, 1])
        self.assertEqual(net.injected[0][:2], (1, 11))

    def test_wrap_carries_state_and_retries_same_target(self):
        net = Scripted({(0, 0, 1): 11})
        ep = {"episode_id": "x", "context": [7], "targets": [11]}
        events = [e for e, _, _ in walk(net, ep, 6)]
        self.assertEqual([e["layer"] for e in events], [1, 2, 3, 1])
        self.assertEqual([e["target_index"] for e in events], [0, 0, 0, 0])
        self.assertEqual(events[2]["action"], "wrap")
        self.assertEqual(net.wraps, [(10., 7)])

    def test_budget_failure_does_not_force_next_token_or_hide_unreached(self):
        net = Scripted({})
        ep = {"episode_id": "x", "document_id": "doc", "context": [7], "targets": [11, 12]}
        summary, rows, _ = evaluate_episodes(net, [ep], 5)
        self.assertEqual(summary["decode_block_visits"], 5)
        self.assertEqual([r["status"] for r in rows], ["failed", "unreached"])
        self.assertEqual(net.injected, [])

    def test_cache_revisit_replaces_without_inplace_mutation_or_fake_skipped_positions(self):
        c = VisitedTokenCache(2)
        a = torch.ones(1, 1, 1, 2, requires_grad=True)
        c.set_positions([0])
        c.update(a, a, 0)
        c.set_positions([3])
        old, _ = c.update(a * 2, a * 2, 0)
        replacement, _ = c.update(a * 3, a * 3, 0)
        self.assertEqual(c.positions[0], [0, 3])
        self.assertEqual(c.positions[1], [])
        self.assertEqual(float(old[0, 0, -1, 0].detach()), 2.)
        self.assertEqual(float(replacement[0, 0, -1, 0].detach()), 3.)
        replacement.sum().backward()
        self.assertTrue(torch.equal(a.grad, torch.full_like(a, 4)))

    def tiny(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(2)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=128, tie_word_embeddings=False)
        base = Qwen2ForCausalLM(config).eval()
        base.set_attn_implementation("sdpa")
        return RoutedNetwork(base, rank=2)

    def test_layerwise_prompt_cache_matches_normal_cached_full_depth(self):
        net = self.tiny()
        context = [1, 2, 3, 4]
        with torch.no_grad():
            normal = net.base(input_ids=torch.tensor([context]), use_cache=False).logits[:, -1]
            h, cache = net.prefill(context)
            for l in range(net.n_layers):
                h = net.block(l, h, cache, len(context) - 1)
            routed = net.logits(net.n_layers - 1, h)
        torch.testing.assert_close(routed, normal, rtol=1e-4, atol=1e-6)

    def test_real_blocks_backprop_through_token_entry_wrap_and_lora_with_tbptt(self):
        net = self.tiny()
        context = [1, 2, 3, 4]
        with torch.no_grad():
            h, c = net.prefill(context)
            h = net.block(0, h, c, 3)
            first = int(net.logits(0, h).argmax())
            h = net.inject(0, first, h)
            candidates = set()
            for layer in range(3):
                h = net.block(layer, h, c, 4)
                candidates.add(int(net.logits(layer, h).argmax()))
            second = next(i for i in range(64) if i not in candidates)
        episode = {"episode_id": "x", "context": context, "targets": [first, second]}
        parameters = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        losses, groups = [], set()
        for event, loss, state in walk(net, episode, 6):
            losses.append(loss)
            if len(losses) == 2 or event["action"] in ("complete", "budget_exhausted"):
                torch.stack(losses).sum().backward()
                for name, p in net.named_parameters():
                    if p.grad is not None and p.grad.abs().sum().item() > 0:
                        if "entry" in name or "memory" in name or "carry_gate" in name: groups.add("entry")
                        if "wrap" in name: groups.add("wrap")
                        if "lora" in name: groups.add("lora")
                state.detach_()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses = []
        self.assertEqual(groups, {"entry", "wrap", "lora"})


if __name__ == "__main__":
    unittest.main()
