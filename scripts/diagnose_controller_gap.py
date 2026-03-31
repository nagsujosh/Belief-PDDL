import argparse
import json
from collections import Counter
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


def analyze_trace(rows):
    total_steps = 0
    sensing_steps = 0
    solvable_steps = 0
    sensed_with_solvable = 0
    blocked_mass_steps = 0
    zero_action_score_with_solvable = 0
    avg_best_scores = []
    avg_solvable_mass = []
    avg_blocked_mass = []
    reason_counts = Counter()
    reason_counts_sensed_with_solvable = Counter()
    blocked_action_counts = Counter()
    vote_counts = Counter()
    first_gap_events = []

    for episode in rows:
        first_gap = None
        for step in episode.get("action_trace", []):
            total_steps += 1
            action = step.get("action")
            reason = step.get("decision_reason", "unknown")
            solvable_worlds = step.get("solvable_worlds", 0)
            best_score = float(step.get("best_score", 0.0) or 0.0)
            solvable_mass = float(step.get("weighted_solvable_mass", 0.0) or 0.0)
            blocked_mass = float(step.get("weighted_blocked_mass", 0.0) or 0.0)
            action_scores = step.get("action_scores", {}) or {}
            blocked_first_actions = step.get("blocked_first_actions", {}) or {}
            first_action_votes = step.get("first_action_votes", {}) or {}

            reason_counts[reason] += 1
            if action == "reveal_side":
                sensing_steps += 1
            if solvable_worlds > 0:
                solvable_steps += 1
                if blocked_mass > 0:
                    blocked_mass_steps += 1
                if not action_scores:
                    zero_action_score_with_solvable += 1
                if action == "reveal_side":
                    sensed_with_solvable += 1
                    reason_counts_sensed_with_solvable[reason] += 1
                    avg_best_scores.append(best_score)
                    avg_solvable_mass.append(solvable_mass)
                    avg_blocked_mass.append(blocked_mass)
                    blocked_action_counts.update(blocked_first_actions)
                    vote_counts.update(first_action_votes)
                    if first_gap is None:
                        first_gap = {
                            "episode": episode.get("episode"),
                            "t": step.get("t"),
                            "solvable_worlds": solvable_worlds,
                            "num_worlds": step.get("num_worlds", 0),
                            "best_plan_action": step.get("best_plan_action"),
                            "best_score": best_score,
                            "weighted_solvable_mass": solvable_mass,
                            "weighted_blocked_mass": blocked_mass,
                            "decision_reason": reason,
                            "blocked_first_actions": blocked_first_actions,
                            "first_action_votes": first_action_votes,
                        }
        if first_gap is not None:
            first_gap_events.append(first_gap)

    return {
        "episodes": len(rows),
        "total_steps": total_steps,
        "sensing_steps": sensing_steps,
        "solvable_steps": solvable_steps,
        "sensed_with_solvable_steps": sensed_with_solvable,
        "sensed_with_solvable_fraction": (sensed_with_solvable / solvable_steps) if solvable_steps else 0.0,
        "solvable_steps_with_blocked_mass": blocked_mass_steps,
        "solvable_steps_with_no_action_scores": zero_action_score_with_solvable,
        "decision_reason_counts": dict(reason_counts),
        "sensed_with_solvable_reason_counts": dict(reason_counts_sensed_with_solvable),
        "avg_best_score_when_sensed_with_solvable": safe_mean(avg_best_scores),
        "avg_weighted_solvable_mass_when_sensed_with_solvable": safe_mean(avg_solvable_mass),
        "avg_weighted_blocked_mass_when_sensed_with_solvable": safe_mean(avg_blocked_mass),
        "top_blocked_first_actions_when_sensed_with_solvable": blocked_action_counts.most_common(5),
        "top_first_action_votes_when_sensed_with_solvable": vote_counts.most_common(5),
        "first_gap_events": first_gap_events[:10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Benchmark JSONL trace produced by run_benchmarks.py")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = load_jsonl(input_path)
    report = analyze_trace(rows)
    report["input"] = str(input_path)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"Controller gap report saved to {output_path}")

    print(f"Input: {input_path}")
    print(f"Episodes: {report['episodes']}")
    print(f"Total steps: {report['total_steps']}")
    print(f"Sensing steps: {report['sensing_steps']}")
    print(f"Solvable steps: {report['solvable_steps']}")
    print(
        "Sensed despite solvable worlds: "
        f"{report['sensed_with_solvable_steps']} "
        f"({100.0 * report['sensed_with_solvable_fraction']:.1f}%)"
    )
    print(f"Solvable steps with blocked mass: {report['solvable_steps_with_blocked_mass']}")
    print(f"Solvable steps with no action scores: {report['solvable_steps_with_no_action_scores']}")
    print(
        "Avg sensed-with-solvable best score / solvable mass / blocked mass: "
        f"{report['avg_best_score_when_sensed_with_solvable']:.3f} / "
        f"{report['avg_weighted_solvable_mass_when_sensed_with_solvable']:.3f} / "
        f"{report['avg_weighted_blocked_mass_when_sensed_with_solvable']:.3f}"
    )
    print("Decision reasons:", report["decision_reason_counts"])
    print("Sensed-with-solvable reasons:", report["sensed_with_solvable_reason_counts"])
    print("Top blocked first actions:", report["top_blocked_first_actions_when_sensed_with_solvable"])
    print("Top first-action votes:", report["top_first_action_votes_when_sensed_with_solvable"])


if __name__ == "__main__":
    main()
