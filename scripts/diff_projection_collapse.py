import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_benchmarks import AIBenchmarker
from src.belief.state import PredicateBelief


def load_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def top_deltas(curr, prev, limit=12):
    preds = sorted(set(curr.keys()) | set(prev.keys()))
    scored = []
    for pred in preds:
        delta = abs(curr.get(pred, 0.5) - prev.get(pred, 0.5))
        scored.append((pred, delta, curr.get(pred, 0.5), prev.get(pred, 0.5)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        {
            "predicate": pred,
            "abs_delta": delta,
            "current": current,
            "previous": previous,
        }
        for pred, delta, current, previous in scored[:limit]
    ]


def snapshot_step(benchmarker, belief_state, obs, top_k, planner_debug, cmd, args, prev_probs):
    per_world = []
    objects = list(benchmarker.env.blocks)
    for idx, world in enumerate(top_k):
        result = benchmarker.planner.det_planner.plan_with_diagnostics(world, "(on block_0 block_1)", objects)
        per_world.append({
            "world_index": idx,
            "solvable": result.solvable,
            "first_action": result.plan[0] if result.plan else None,
            "plan_length": len(result.plan) if result.plan else 0,
            "true_predicates": sorted([pred for pred, is_true in world.items() if is_true]),
        })

    belief_probs = dict(belief_state.probs)
    return {
        "visible_objects": list(obs.visible_objects),
        "gt_predicates": list(obs.gt_predicates),
        "selected_action": cmd,
        "selected_args": args,
        "planner_debug": planner_debug,
        "top_belief_deltas": top_deltas(belief_probs, prev_probs),
        "belief_probs": belief_probs,
        "top_k_worlds": per_world,
    }


def find_collapse_step(action_trace):
    prev_solvable = None
    for step in action_trace:
        curr = step.get("solvable_worlds", 0)
        if prev_solvable is not None and prev_solvable > 0 and curr == 0:
            return step.get("t")
        prev_solvable = curr
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Seeded benchmark JSONL trace")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--learned-head-path", type=str, default="outputs/perception/perception_compare_e200_learned_head.pt")
    parser.add_argument("--calibration-path", type=str, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collapse-step", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    episode_row = next((row for row in rows if row.get("episode") == args.episode), None)
    if episode_row is None:
        raise ValueError(f"Episode {args.episode} not found in {args.input}")

    collapse_t = args.collapse_step
    if collapse_t is None:
        collapse_t = find_collapse_step(episode_row.get("action_trace", []))
    if collapse_t is None:
        raise ValueError("Could not infer collapse step from action trace; pass --collapse-step explicitly.")
    target_steps = {max(0, collapse_t - 1), collapse_t}

    episode_seed = episode_row.get("seed", 0)
    accepted_seed = episode_row.get("accepted_seed", episode_seed)

    benchmarker = AIBenchmarker(
        mode=episode_row["mode"],
        noise_level=episode_row["noise_level"],
        k_worlds=3,
        device=args.device,
        alpha=episode_row.get("alpha", 1.0),
        decay=episode_row.get("decay", 1.0),
        seed=episode_seed - args.episode * 1000,
        perception_mode=episode_row.get("perception_mode", "zero_shot"),
        learned_head_path=args.learned_head_path if episode_row.get("perception_mode") == "learned_head" else None,
        calibration_path=args.calibration_path if episode_row.get("perception_mode") == "calibrated" else None,
    )

    benchmarker.env.reset(seed=accepted_seed)
    benchmarker.planner.reset_episode_state()
    belief_state = PredicateBelief()
    revealed_blocks = set()
    prev_probs = {}
    snapshots = {}

    for t in range(collapse_t + 1):
        obs = benchmarker.env._get_obs()
        benchmarker._sync_visibility_from_obs(belief_state, obs, timestep=t)
        raw_logits = benchmarker._run_vlm(obs.rgb, revealed_blocks)

        for pred, prob in raw_logits.items():
            if pred.startswith("visible("):
                continue
            updated = benchmarker.updater.update(belief_state.get_belief(pred), prob)
            belief_state.set_belief(pred, updated, timestep=t)

        if benchmarker.mode in ["threshold", "belief_no_verifier"]:
            top_k = [{k: v > 0.5 for k, v in belief_state.probs.items()}]
        elif benchmarker.mode == "belief_plus_verifier":
            top_k = benchmarker.projector.project_top_k_map_states(belief_state.probs, k=1)
        else:
            top_k = benchmarker.projector.project_top_k_map_states(belief_state.probs, k=benchmarker.k)

        cmd, action_args = benchmarker.planner.select_action(
            belief_state.probs,
            top_k,
            "(on block_0 block_1)",
            list(benchmarker.env.blocks),
        )
        planner_debug = copy.deepcopy(benchmarker.planner.last_debug)

        if t in target_steps:
            snapshots[str(t)] = snapshot_step(
                benchmarker,
                belief_state,
                obs,
                top_k,
                planner_debug,
                cmd,
                action_args,
                prev_probs,
            )

        prev_gt = set(obs.gt_predicates)
        if cmd == "reveal_side" and action_args:
            revealed_blocks.add(action_args[0])
        obs, reward, done = benchmarker.env.step(cmd, action_args)
        state_changed = prev_gt != set(obs.gt_predicates)

        if cmd == "reveal_side":
            if action_args:
                benchmarker._set_hard_belief(
                    belief_state,
                    f"visible({action_args[0]})",
                    action_args[0] in obs.visible_objects,
                    timestep=t,
                    source="sensing",
                )
        elif state_changed:
            benchmarker._sync_action_effects(belief_state, cmd, action_args, obs, timestep=t)
        else:
            benchmarker._apply_failed_action_correction(belief_state, cmd, action_args, obs, timestep=t)
        benchmarker.planner.register_action_feedback(cmd, action_args, state_changed)
        prev_probs = dict(belief_state.probs)

    report = {
        "input": args.input,
        "episode": args.episode,
        "episode_seed": episode_seed,
        "accepted_seed": accepted_seed,
        "collapse_step": collapse_t,
        "snapshots": snapshots,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"Saved projection collapse diff to {output}")
    print(f"Captured steps: {sorted(snapshots.keys(), key=int)}")


if __name__ == "__main__":
    main()
