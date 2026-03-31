import glob
import json
import os
import re


def parse_name(name: str):
    mode_match = re.match(r"^(.*)_noise_", name)
    noise_match = re.search(r"_noise_([0-9.]+)", name)
    alpha_match = re.search(r"_alpha_([0-9.]+)", name)
    decay_match = re.search(r"_decay_([0-9.]+)", name)
    seed_match = re.search(r"_seed_([0-9]+)", name)
    perception_match = re.search(r"_perception_([a-z_]+)$", name)
    return {
        "mode": mode_match.group(1) if mode_match else name,
        "noise": float(noise_match.group(1)) if noise_match else None,
        "alpha": float(alpha_match.group(1)) if alpha_match else 1.0,
        "decay": float(decay_match.group(1)) if decay_match else 1.0,
        "seed": int(seed_match.group(1)) if seed_match else None,
        "perception": perception_match.group(1) if perception_match else "zero_shot",
    }


rows_by_name = {}
for file in sorted(glob.glob("outputs/benchmarks/*.jsonl")):
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
    if not data:
        continue

    name = os.path.basename(file).replace(".jsonl", "")
    meta = parse_name(name)
    action_steps = [
        step for episode in data for step in episode.get("action_trace", [])
        if isinstance(step, dict)
    ]
    avg_solvable = (
        sum(step.get("solvable_worlds", 0) / max(1, step.get("num_worlds", 1)) for step in action_steps) / len(action_steps)
        if action_steps else 0.0
    )
    rows_by_name[name] = {
        "name": name,
        "mode": meta["mode"],
        "noise": meta["noise"],
        "alpha": meta["alpha"],
        "decay": meta["decay"],
        "seed": meta["seed"],
        "perception": meta["perception"],
        "success": sum(1 for d in data if d["success"]) / len(data) * 100,
        "inconsistency": sum(d["inconsistency_events"] for d in data) / len(data),
        "sensing": sum(d.get("sensing_actions", 0) for d in data) / len(data),
        "steps": sum(d["steps"] for d in data) / len(data),
        "solvable_ratio": avg_solvable * 100,
    }

rows = list(rows_by_name.values())
rows.sort(key=lambda r: (r["mode"], r["noise"] if r["noise"] is not None else 999, r["alpha"], r["decay"], r["seed"] if r["seed"] is not None else -1, r["name"]))

print(
    f"{'Mode':<24} | {'Perception':<11} | {'Noise':<5} | {'Alpha':<5} | {'Decay':<5} | {'Seed':<5} | "
    f"{'Success':<8} | {'Solvable':<8} | {'Sensing':<8} | {'Steps':<6}"
)
print("-" * 122)
for row in rows:
    noise_str = f"{row['noise']:.1f}" if row["noise"] is not None else "-"
    seed_str = str(row["seed"]) if row["seed"] is not None else "-"
    print(
        f"{row['mode']:<24} | {row['perception']:<11} | {noise_str:<5} | {row['alpha']:<5.2f} | {row['decay']:<5.2f} | {seed_str:<5} | "
        f"{row['success']:>5.1f}%   | {row['solvable_ratio']:>5.1f}%   | "
        f"{row['sensing']:>6.1f}   | {row['steps']:>5.1f}"
    )
