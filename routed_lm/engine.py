from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class State:
    h: torch.Tensor
    cache: object
    layer: int
    position: int
    input_token: int

    def detach_(self):
        self.h = self.h.detach()
        self.cache.detach_()


def walk(network, episode, max_visits_per_token):
    """Yield AFTER a visited block and BEFORE taking its next transition.

    The caller may backpropagate and detach State at this boundary. Injection
    and wrap modules then stay attached to the NEXT visited block's loss.
    """
    context, targets = episode["context"], episode["targets"]
    hidden, cache = network.prefill(context)
    state = State(hidden, cache, 0, len(context) - 1, context[-1])
    target_index, visits = 0, 0
    while target_index < len(targets):
        state.h = network.block(state.layer, state.h, state.cache, state.position)
        logits = network.logits(state.layer, state.h).float()
        target = torch.tensor([targets[target_index]], device=logits.device)
        loss = F.cross_entropy(logits, target)
        predicted = int(logits.detach().argmax(dim=-1).item())
        visits += 1
        accepted = predicted == targets[target_index]
        terminal = accepted and target_index == len(targets) - 1
        exhausted = not accepted and visits >= max_visits_per_token
        action = ("complete" if terminal else "emit" if accepted else "budget_exhausted" if exhausted
                  else "wrap" if state.layer == network.n_layers - 1 else "advance")
        next_layer = None if terminal or exhausted else state.layer if accepted else (state.layer + 1) % network.n_layers
        event = {"episode_id": episode["episode_id"], "target_index": target_index,
                 "input_position": state.position, "input_token_id": state.input_token,
                 "layer": state.layer + 1, "predicted_id": predicted,
                 "target_id": targets[target_index], "accepted": accepted,
                 "visits_for_target": visits, "action": action,
                 "next_layer": None if next_layer is None else next_layer + 1,
                 "injected_token_id": predicted if action == "emit" else None}
        yield event, loss, state
        # Do not run even one extra block for the old token after acceptance.
        if terminal or exhausted:
            return
        if accepted:
            state.input_token = predicted
            state.position += 1
            state.h = network.inject(state.layer, predicted, state.h)
            target_index += 1
            visits = 0
        elif action == "wrap":
            state.h = network.wrap(state.h, state.input_token)
            state.layer = 0
        else:
            state.layer += 1
