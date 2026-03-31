import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(values):
    return mean(values) if values else 0.0


def seed_file(prefix: str, seed: int) -> Path:
    return Path(f"{prefix}{seed}_perception_learned_head.jsonl")


def summarize_group(files):
    episodes = []
    per_t = defaultdict(lambda: {
        "solvable_ratio": [],
        "sensing": [],
        "changed_state": [],
        "plan_action": [],
        "deadlock": [],
        "low_conf": [],
    })
    action_counter = Counter()
    reason_counter = Counter()
    blocked_counter = Counter()

    for file in files:
        rows = load_jsonl(file)
        for ep in rows:
            trace = ep.get("action_trace", [])
            solvable_steps = [
                step.get("solvable_worlds", 0) / max(1, step.get("num_worlds", 1))
                for step in trace
            ]
            plan_steps = [step for step in trace if step.get("action") != "reveal_side"]
            changed_steps = [step for step in trace if step.get("changed_state")]
            first_plan_t = next((step.get("t") for step in trace if step.get("action") != "reveal_side"), None)
            first_changed_t = next((step.get("t") for step in trace if step.get("changed_state")), None)

            collapse_t = None
            for idx in range(len(trace) - 2):
                if (
                    any(s > 0 for s in solvable_steps[: idx + 1])
                    and all(s == 0 for s in solvable_steps[idx : idx + 3])
                ):
                    collapse_t = idx
                    break

            episodes.append({
                "success": ep.get("success", False),
                "avg_solvable_ratio": safe_mean(solvable_steps),
                "num_plan_steps": len(plan_steps),
                "num_changed_steps": len(changed_steps),
                "first_plan_t": first_plan_t,
                "first_changed_t": first_changed_t,
                "collapse_t": collapse_t,
            })

            for step in trace:
                t = step.get("t", 0)
                solvable_ratio = step.get("solvable_worlds", 0) / max(1, step.get("num_worlds", 1))
                reason = step.get("decision_reason", "unknown")
                per_t[t]["solvable_ratio"].append(solvable_ratio)
                per_t[t]["sensing"].append(1.0 if step.get("action") == "reveal_side" else 0.0)
                per_t[t]["changed_state"].append(1.0 if step.get("changed_state") else 0.0)
                per_t[t]["plan_action"].append(1.0 if step.get("action") != "reveal_side" else 0.0)
                per_t[t]["deadlock"].append(1.0 if reason == "sensing_deadlock" else 0.0)
                per_t[t]["low_conf"].append(1.0 if reason == "sensing_low_confidence" else 0.0)
                action_counter.update([step.get("action", "unknown")])
                reason_counter.update([reason])
                blocked_counter.update((step.get("blocked_actions") or {}).keys())

    per_t_summary = {
        str(t): {
            metric: safe_mean(values)
            for metric, values in bucket.items()
        }
        for t, bucket in sorted(per_t.items())
    }

    return {
        "num_files": len(files),
        "num_episodes": len(episodes),
        "avg_success": safe_mean([1.0 if ep["success"] else 0.0 for ep in episodes]),
        "avg_solvable_ratio": safe_mean([ep["avg_solvable_ratio"] for ep in episodes]),
        "avg_num_plan_steps": safe_mean([ep["num_plan_steps"] for ep in episodes]),
        "avg_num_changed_steps": safe_mean([ep["num_changed_steps"] for ep in episodes]),
        "avg_first_plan_t": safe_mean([ep["first_plan_t"] for ep in episodes if ep["first_plan_t"] is not None]),
        "avg_first_changed_t": safe_mean([ep["first_changed_t"] for ep in episodes if ep["first_changed_t"] is not None]),
        "avg_collapse_t": safe_mean([ep["collapse_t"] for ep in episodes if ep["collapse_t"] is not None]),
        "top_actions": action_counter.most_common(8),
        "decision_reasons": dict(reason_counter),
        "top_blocked_actions": blocked_counter.most_common(8),
        "per_t": per_t_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Common file prefix before the seed number")
    parser.add_argument("--high-seeds", required=True, help="Comma-separated seed list for promising seeds")
    parser.add_argument("--low-seeds", required=True, help="Comma-separated seed list for collapse seeds")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    high_seeds = [int(x) for x in args.high_seeds.split(",") if x.strip()]
    low_seeds = [int(x) for x in args.low_seeds.split(",") if x.strip()]

    high_files = [seed_file(args.prefix, seed) for seed in high_seeds]
    low_files = [seed_file(args.prefix, seed) for seed in low_seeds]

    report = {
        "prefix": args.prefix,
        "high_seeds": high_seeds,
        "low_seeds": low_seeds,
        "high_group": summarize_group(high_files),
        "low_group": summarize_group(low_files),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    print(f"Saved seed bifurcation report to {output}")
    for label in ["high_group", "low_group"]:
        group = report[label]
        print(
            f"{label}: solvable={100.0 * group['avg_solvable_ratio']:.1f}% | "
            f"plan_steps={group['avg_num_plan_steps']:.1f} | "
            f"changed_steps={group['avg_num_changed_steps']:.1f} | "
            f"first_plan_t={group['avg_first_plan_t']:.1f} | "
            f"collapse_t={group['avg_collapse_t']:.1f}"
        )
        print(f"  reasons={group['decision_reasons']}")
        print(f"  top_actions={group['top_actions']}")


if __name__ == "__main__":
    main()
