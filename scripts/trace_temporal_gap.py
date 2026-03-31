import argparse
import copy
import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_benchmarks import AIBenchmarker
from tests.mocks.blocksworld_env import MockBlocksworldEnv
from src.belief.state import PredicateBelief
from src.perception.blocksworld_predicates import grounded_predicates


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, set):
        return sorted(to_jsonable(v) for v in value)
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


def reset_unsolved_state(seed_offset: int = 0) -> Dict:
    for attempt in range(50):
        random.seed(seed_offset + attempt)
        env = MockBlocksworldEnv(num_blocks=3)
        env.reset()
        if env.state.get("on", {}).get("block_0") != "block_1":
            return copy.deepcopy(env.state)
    raise RuntimeError("Failed to sample unsolved initial state for trace comparison.")


def reset_planner_state(benchmarker: AIBenchmarker):
    benchmarker.planner._last_sensing_target = None
    benchmarker.planner._blocked_actions.clear()
    benchmarker.planner.last_debug = {}


def run_single_episode(benchmarker: AIBenchmarker, initial_state: Dict, episode_id: int, max_steps: int) -> Dict:
    benchmarker.env.state = copy.deepcopy(initial_state)
    reset_planner_state(benchmarker)

    belief_state = PredicateBelief()
    goal_str = "(on block_0 block_1)"
    revealed_blocks = set()
    predicates = grounded_predicates(benchmarker.env.blocks)

    ep = {
        "episode": episode_id,
        "perception_mode": benchmarker.perception_mode,
        "success": False,
        "steps": max_steps,
        "trace": [],
    }

    for t in range(max_steps):
        obs = benchmarker.env._get_obs()
        benchmarker._sync_visibility_from_obs(belief_state, obs, timestep=t)
        raw_probs = benchmarker._run_vlm(obs.rgb, revealed_blocks)

        if benchmarker.mode == "threshold":
            for pred, prob in raw_probs.items():
                if pred.startswith("visible("):
                    continue
                belief_state.set_belief(pred, prob, t)
        else:
            for pred, prob in raw_probs.items():
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

        planner_objects = list(benchmarker.env.blocks)
        cmd, args = benchmarker.planner.select_action(belief_state.probs, top_k, goal_str, planner_objects)
        planner_debug = dict(benchmarker.planner.last_debug)
        pre_gt = set(obs.gt_predicates)

        if cmd == "reveal_side" and args:
            revealed_blocks.add(args[0])

        obs, reward, done = benchmarker.env.step(cmd, args)
        post_gt = set(obs.gt_predicates)
        state_changed = pre_gt != post_gt

        if cmd == "reveal_side":
            if args:
                benchmarker._set_hard_belief(
                    belief_state,
                    f"visible({args[0]})",
                    args[0] in obs.visible_objects,
                    timestep=t,
                    source="sensing",
                )
        elif state_changed:
            benchmarker._sync_action_effects(belief_state, cmd, args, obs, timestep=t)
        else:
            benchmarker._apply_failed_action_correction(belief_state, cmd, args, obs, timestep=t)

        benchmarker.planner.register_action_feedback(cmd, args, state_changed)

        ep["trace"].append(
            {
                "t": t,
                "raw_probs": {pred: raw_probs.get(pred, 0.5) for pred in predicates},
                "belief_probs": {pred: belief_state.get_belief(pred) for pred in predicates},
                "solvable_worlds": planner_debug.get("solvable_worlds", 0),
                "num_worlds": planner_debug.get("num_worlds", len(top_k)),
                "first_action_votes": planner_debug.get("first_action_votes", {}),
                "blocked_actions": planner_debug.get("blocked_actions", {}),
                "action": cmd,
                "args": args,
                "changed_state": state_changed,
                "reward": reward,
                "done": done,
                "visible_objects": list(obs.visible_objects),
                "gt_predicates": list(obs.gt_predicates),
            }
        )

        if done:
            ep["success"] = True
            ep["steps"] = t
            break

    return ep


def average_abs_diff(a: Dict[str, float], b: Dict[str, float], preds: List[str]) -> Dict[str, float]:
    return {pred: abs(a.get(pred, 0.5) - b.get(pred, 0.5)) for pred in preds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--mode", type=str, default="full_top_k")
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--learned-head-path", type=str, default="outputs/perception/perception_compare_e200_learned_head.pt")
    parser.add_argument("--output", type=str, default="outputs/perception/temporal_gap_trace.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    zero_shot = AIBenchmarker(
        mode=args.mode,
        noise_level=args.noise,
        k_worlds=args.k,
        device=args.device,
        alpha=args.alpha,
        decay=args.decay,
        perception_mode="zero_shot",
    )
    learned = AIBenchmarker(
        mode=args.mode,
        noise_level=args.noise,
        k_worlds=args.k,
        device=args.device,
        alpha=args.alpha,
        decay=args.decay,
        perception_mode="learned_head",
        learned_head_path=args.learned_head_path,
    )

    predicates = grounded_predicates(zero_shot.env.blocks)
    episodes = []
    aggregate_raw = {pred: [] for pred in predicates}
    aggregate_belief = {pred: [] for pred in predicates}

    for episode_id in range(args.episodes):
        initial_state = reset_unsolved_state(seed_offset=episode_id * 1000)
        zero_trace = run_single_episode(zero_shot, initial_state, episode_id, args.max_steps)
        learned_trace = run_single_episode(learned, initial_state, episode_id, args.max_steps)

        for z_step, l_step in zip(zero_trace["trace"], learned_trace["trace"]):
            raw_diff = average_abs_diff(z_step["raw_probs"], l_step["raw_probs"], predicates)
            belief_diff = average_abs_diff(z_step["belief_probs"], l_step["belief_probs"], predicates)
            for pred in predicates:
                aggregate_raw[pred].append(raw_diff[pred])
                aggregate_belief[pred].append(belief_diff[pred])
            z_step["vs_learned_raw_abs_diff"] = raw_diff
            z_step["vs_learned_belief_abs_diff"] = belief_diff

        episodes.append(
            {
                "episode": episode_id,
                "initial_state": to_jsonable(initial_state),
                "zero_shot": zero_trace,
                "learned_head": learned_trace,
            }
        )

    summary = {
        "config": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "mode": args.mode,
            "noise": args.noise,
            "k": args.k,
            "alpha": args.alpha,
            "decay": args.decay,
            "device": args.device,
        },
        "avg_raw_abs_diff": {
            pred: (sum(vals) / len(vals) if vals else 0.0)
            for pred, vals in aggregate_raw.items()
        },
        "avg_belief_abs_diff": {
            pred: (sum(vals) / len(vals) if vals else 0.0)
            for pred, vals in aggregate_belief.items()
        },
        "episodes": episodes,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved temporal trace to {args.output}")
    print(
        "Top raw-probability differences:",
        sorted(summary["avg_raw_abs_diff"].items(), key=lambda kv: kv[1], reverse=True)[:8],
    )
    print(
        "Top belief differences:",
        sorted(summary["avg_belief_abs_diff"].items(), key=lambda kv: kv[1], reverse=True)[:8],
    )


if __name__ == "__main__":
    main()
