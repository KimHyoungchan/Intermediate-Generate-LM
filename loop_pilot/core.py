from __future__ import annotations

import hashlib
import json
import math
import time
import torch

from routed_lm.engine import walk


def state_fingerprint(state):
    """Observe equal states at the same token and L->1 boundary; do not stop.

    Past-position K/V is constant while retrying one token, so include current
    position K/V for every layer plus h and the per-layer presence flags.
    """
    cache = state.cache
    parts = [state.h.detach().reshape(-1)]
    present = []
    for positions, key, value in zip(cache.positions, cache.keys, cache.values):
        is_current = bool(positions and positions[-1] == state.position)
        present.append(is_current)
        if is_current:
            parts.extend([key[..., -1, :].detach().reshape(-1), value[..., -1, :].detach().reshape(-1)])
    h = hashlib.sha256(json.dumps([state.position, state.input_token, present]).encode())
    tensor = torch.cat(parts).contiguous().view(torch.uint8).cpu().numpy()
    h.update(memoryview(tensor))
    return h.hexdigest()


@torch.no_grad()
def profile_episode(network, episode, seconds=60, max_total_visits=0,
                    fingerprint=True, progress=None, clock=time.monotonic):
    """Continue one actual path until complete or an external watchdog fires.

    Inf disables the engine's per-token budget. Count only observed L->1
    entries, since a yielded event's planned transition has not executed yet.
    """
    rows = [{"episode_id": episode["episode_id"], "target_index": i, "target_id": t,
             "status": "unreached", "visits": 0, "wraps": 0, "start_layer": None,
             "exit_layer": None, "first_repeated_state_wrap": None,
             "observed_state_repeats": 0} for i, t in enumerate(episode["targets"])]
    generator = walk(network, episode, math.inf)
    started = clock()
    previous = None
    total_wraps, visits, injections = 0, 0, 0
    outcome, reason = "censored", "unknown"
    repeated = {}
    checkpoints = []
    current_index = None
    try:
        for event, _loss, state in generator:
            index = event["target_index"]
            row = rows[index]
            if current_index != index:
                repeated = {}
                current_index = index
            observed_wrap = False
            if previous is not None:
                if previous["action"] == "wrap":
                    assert event["layer"] == 1 and index == previous["target_index"]
                    total_wraps += 1
                    row["wraps"] += 1
                    observed_wrap = True
                elif previous["action"] == "emit":
                    assert event["layer"] == previous["layer"]
                    assert index == previous["target_index"] + 1
                    assert event["input_token_id"] == previous["predicted_id"]
                    injections += 1
            visits += 1
            row["visits"] += 1
            if row["start_layer"] is None:
                row["start_layer"] = event["layer"]
            row["status"] = "hit" if event["accepted"] else "running"
            if event["accepted"]:
                row["exit_layer"] = event["layer"]
            if fingerprint and event["action"] == "wrap":
                signature = state_fingerprint(state)
                boundary_number = row["wraps"] + 1
                if signature in repeated:
                    row["observed_state_repeats"] += 1
                    if row["first_repeated_state_wrap"] is None:
                        row["first_repeated_state_wrap"] = boundary_number
                else:
                    repeated[signature] = boundary_number
                    if len(repeated) > 4096:
                        del repeated[next(iter(repeated))]
            elapsed = clock() - started
            if observed_wrap or event["accepted"]:
                checkpoint = {"episode_id": episode["episode_id"], "target_index": index,
                    "total_wraps": total_wraps, "target_wraps": row["wraps"],
                    "visited_blocks": visits, "resolved_targets": sum(r["status"] == "hit" for r in rows),
                    "elapsed_seconds": elapsed, "event": "hit" if event["accepted"] else "observed_wrap"}
                if progress:
                    progress(checkpoint)
                else:
                    checkpoints.append(checkpoint)
            previous = event
            if event["action"] == "complete":
                outcome, reason = "complete", "all_targets_hit"
                break
            assert event["action"] != "budget_exhausted", "Per-token budget must be disabled"
            if seconds > 0 and elapsed >= seconds:
                reason = "wall_time_watchdog"
                break
            if max_total_visits > 0 and visits >= max_total_visits:
                reason = "episode_visit_watchdog"
                break
        else:
            raise RuntimeError("Unbounded route ended without completing the episode")
    except KeyboardInterrupt:
        reason = "user_interrupt"
    finally:
        generator.close()
    elapsed = clock() - started
    for row in rows:
        if row["status"] == "running":
            row["status"] = "censored"
    resolved = sum(r["status"] == "hit" for r in rows)
    result = {"episode_id": episode["episode_id"], "document_id": episode["document_id"],
        "outcome": outcome, "stop_reason": reason, "requested_targets": len(rows),
        "resolved_targets": resolved, "unreached_targets": sum(r["status"] == "unreached" for r in rows),
        "censored_targets": sum(r["status"] == "censored" for r in rows),
        "observed_wraps": total_wraps,
        "wraps_to_completion": total_wraps if outcome == "complete" else None,
        "completion_wraps_lower_bound": total_wraps,
        "max_observed_wraps_for_one_token": max(r["wraps"] for r in rows),
        "visited_blocks": visits, "confirmed_same_layer_injections": injections,
        "elapsed_seconds": elapsed, "last_target_index": current_index,
        "repeated_state_observed": any(r["observed_state_repeats"] for r in rows)}
    return result, rows, checkpoints


def aggregate(episodes, tokens, selected_count):
    completed = [e for e in episodes if e["outcome"] == "complete"]
    hit = [t for t in tokens if t["status"] == "hit"]
    def mean(values):
        return sum(values) / len(values) if values else None
    return {"selected_episodes": selected_count, "profiled_episodes": len(episodes),
        "unstarted_episodes": selected_count - len(episodes), "completed_episodes": len(completed),
        "censored_episodes": len(episodes) - len(completed),
        "completion_fraction": len(completed) / selected_count,
        "mean_wraps_to_completion_completed_only": mean([e["observed_wraps"] for e in completed]),
        "max_wraps_to_completion_completed_only": max([e["observed_wraps"] for e in completed], default=None),
        "resolved_targets": len(hit), "attempted_targets": sum(t["status"] != "unreached" for t in tokens),
        "mean_token_wraps_hit_only": mean([t["wraps"] for t in hit]),
        "max_observed_wraps_for_one_token": max([t["wraps"] for t in tokens], default=0),
        "total_observed_wraps": sum(e["observed_wraps"] for e in episodes),
        "total_visited_blocks": sum(e["visited_blocks"] for e in episodes),
        "state_repeat_episodes": sum(e["repeated_state_observed"] for e in episodes),
        "per_token_visit_limit": None,
        "loop_definition": "one actually observed L-to-1 transition for the same target token",
        "incomplete_semantics": "right-censored; required loops unknown; observed count is a lower bound"}
