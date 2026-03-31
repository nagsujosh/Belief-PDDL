import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean


def parse_seeds(seed_arg: str, seed_start: int, seed_count: int):
    if seed_arg:
        return [int(part.strip()) for part in seed_arg.split(",") if part.strip()]
    return [seed_start + i for i in range(seed_count)]


def benchmark_output_path(args, seed: int) -> Path:
    name = (
        f"{args.mode}_noise_{args.noise}"
        f"_alpha_{args.alpha}_decay_{args.decay}"
        f"_seed_{seed}_perception_{args.perception_mode}.jsonl"
    )
    return Path("outputs/benchmarks") / name


def load_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_run(rows):
    action_steps = [
        step for episode in rows for step in episode.get("action_trace", [])
        if isinstance(step, dict)
    ]
    solvable_ratio = (
        sum(step.get("solvable_worlds", 0) / max(1, step.get("num_worlds", 1)) for step in action_steps) / len(action_steps)
        if action_steps else 0.0
    )
    return {
        "episodes": len(rows),
        "success": sum(1 for row in rows if row.get("success")) / max(1, len(rows)),
        "sensing": sum(row.get("sensing_actions", 0) for row in rows) / max(1, len(rows)),
        "steps": sum(row.get("steps", 0) for row in rows) / max(1, len(rows)),
        "inconsistency": sum(row.get("inconsistency_events", 0) for row in rows) / max(1, len(rows)),
        "solvable_ratio": solvable_ratio,
    }


def build_command(args, seed: int):
    cmd = [
        sys.executable,
        "scripts/run_benchmarks.py",
        "--mode",
        args.mode,
        "--noise",
        str(args.noise),
        "--k",
        str(args.k),
        "--episodes",
        str(args.episodes),
        "--device",
        args.device,
        "--alpha",
        str(args.alpha),
        "--decay",
        str(args.decay),
        "--seed",
        str(seed),
        "--perception-mode",
        args.perception_mode,
    ]
    if args.learned_head_path:
        cmd.extend(["--learned-head-path", args.learned_head_path])
    if args.calibration_path:
        cmd.extend(["--calibration-path", args.calibration_path])
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full_top_k")
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--perception-mode", type=str, default="zero_shot")
    parser.add_argument("--learned-head-path", type=str, default=None)
    parser.add_argument("--calibration-path", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated explicit seed list, e.g. 0,1,2,3,4")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON summary path")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds, args.seed_start, args.seed_count)
    summaries = []

    for seed in seeds:
        cmd = build_command(args, seed)
        print(f"\n=== Running seed {seed} ===")
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        out_path = benchmark_output_path(args, seed)
        if not out_path.exists():
            raise FileNotFoundError(f"Expected benchmark output was not created: {out_path}")

        rows = load_jsonl(out_path)
        summary = summarize_run(rows)
        summary["seed"] = seed
        summary["output"] = str(out_path)
        summaries.append(summary)

    aggregate = {
        "mode": args.mode,
        "noise": args.noise,
        "alpha": args.alpha,
        "decay": args.decay,
        "perception_mode": args.perception_mode,
        "episodes_per_seed": args.episodes,
        "seeds": seeds,
        "runs": summaries,
        "aggregate": {
            "success": mean(run["success"] for run in summaries) if summaries else 0.0,
            "solvable_ratio": mean(run["solvable_ratio"] for run in summaries) if summaries else 0.0,
            "sensing": mean(run["sensing"] for run in summaries) if summaries else 0.0,
            "steps": mean(run["steps"] for run in summaries) if summaries else 0.0,
            "inconsistency": mean(run["inconsistency"] for run in summaries) if summaries else 0.0,
        },
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregate, indent=2))
        print(f"\nSeed sweep summary saved to {output_path}")

    print(
        f"\n{'Seed':<6} | {'Success':<8} | {'Solvable':<8} | {'Sensing':<8} | {'Steps':<6}"
    )
    print("-" * 56)
    for run in summaries:
        print(
            f"{run['seed']:<6} | {100.0 * run['success']:>5.1f}%   | "
            f"{100.0 * run['solvable_ratio']:>5.1f}%   | {run['sensing']:>6.1f}   | {run['steps']:>5.1f}"
        )

    agg = aggregate["aggregate"]
    print("-" * 56)
    print(
        f"{'avg':<6} | {100.0 * agg['success']:>5.1f}%   | "
        f"{100.0 * agg['solvable_ratio']:>5.1f}%   | {agg['sensing']:>6.1f}   | {agg['steps']:>5.1f}"
    )


if __name__ == "__main__":
    main()
