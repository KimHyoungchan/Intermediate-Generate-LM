from __future__ import annotations

import torch
import torch.nn.functional as F
from routed_lm.engine import State


def walk_training(network, episode, label_mode, iteration_policy):
    """Visited-head CE with teacher-forced continuation instead of dropping data.

    Configuration is explicit: selecting a label alignment or iteration unit
    silently would change the experiment's objective.
    Yield boundaries match routed_lm.engine.walk, allowing TBPTT before the
    subsequent state transition is computed.
    """
    if label_mode not in ("next_token", "input_token"):
        raise ValueError("label_mode must explicitly be next_token or input_token")
    if iteration_policy not in ("per_token", "global", "advance_on_last"):
        raise ValueError("iteration_policy must explicitly be per_token, global or advance_on_last")
    targets = episode["targets"]
    context = episode["context"]
    if not targets:
        raise ValueError("Training episode must have a target")
    initial_context = context if label_mode == "next_token" else [*context, targets[0]]
    hidden, cache = network.prefill(initial_context)
    state = State(hidden, cache, 0, len(initial_context) - 1, initial_context[-1])
    index, token_iteration, global_iteration, visits = 0, 1, 1, 0
    while index < len(targets):
        state.h = network.block(state.layer, state.h, state.cache, state.position)
        logits = network.logits(state.layer, state.h).float()
        label = targets[index]
        loss = F.cross_entropy(logits, torch.tensor([label], device=logits.device))
        prediction = int(logits.detach().argmax(dim=-1).item())
        hit = prediction == label
        visits += 1
        at_last = state.layer == network.n_layers - 1
        iteration = global_iteration if iteration_policy == "global" else token_iteration
        forced = (not hit and at_last and
                  (iteration_policy == "advance_on_last" or iteration >= index + 1))
        finishing = (hit or forced) and index == len(targets) - 1
        if finishing:
            action = "complete_hit" if hit else "complete_forced"
        elif hit:
            action = "emit"
        elif forced:
            action = "force_next"
        elif at_last:
            action = "wrap"
        else:
            action = "advance"
        injected = None
        if action in ("emit", "force_next"):
            # next_token: feed the current label to predict the next label.
            # input_token: feed the NEXT raw input token to reconstruct itself.
            injected = label if label_mode == "next_token" else targets[index + 1]
        next_layer = (None if finishing else 0 if forced or action == "wrap"
                      else state.layer if hit else state.layer + 1)
        event = {"episode_id": episode["episode_id"], "target_index": index,
                 "input_position": state.position, "input_token_id": state.input_token,
                 "target_id": label, "label_mode": label_mode,
                 "layer": state.layer + 1, "predicted_id": prediction, "accepted": hit,
                 "forced": forced, "action": action, "injected_token_id": injected,
                 "iteration": iteration, "iteration_limit": index + 1,
                 "token_iteration": token_iteration, "global_iteration": global_iteration,
                 "visits_for_target": visits,
                 "next_layer": None if next_layer is None else next_layer + 1,
                 "context_tokens": len(context), "processed_targets_after_event": index + int(hit or forced)}
        yield event, loss, state
        if finishing:
            return
        if hit or forced:
            if forced:
                state.h = network.wrap(state.h, state.input_token)
                state.layer = 0
                global_iteration += 1
            state.position += 1
            state.input_token = injected
            state.h = network.inject(state.layer, injected, state.h)
            index += 1
            token_iteration = 1
            visits = 0
        elif action == "wrap":
            state.h = network.wrap(state.h, state.input_token)
            state.layer = 0
            token_iteration += 1
            global_iteration += 1
        else:
            state.layer += 1
