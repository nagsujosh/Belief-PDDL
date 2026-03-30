import json, glob, os

print(f"{'Experiment':<35} | {'Success':<8} | {'Inconsistencies':<15} | {'Sensing':<8} | {'Steps':<6}")
print("-" * 80)
for file in sorted(glob.glob("outputs/benchmarks/*.jsonl")):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
        if not data: continue
        
        name = os.path.basename(file).replace(".jsonl", "")
        success_rate = sum(1 for d in data if d["success"]) / len(data) * 100
        avg_inconsistencies = sum(d["inconsistency_events"] for d in data) / len(data)
        avg_sensing = sum(d.get("sensing_actions", 0) for d in data) / len(data)
        avg_steps = sum(d["steps"] for d in data) / len(data)
        
        print(f"{name:<35} | {success_rate:>5.1f}%   | {avg_inconsistencies:>12.1f}    | {avg_sensing:>6.1f}   | {avg_steps:>5.1f}")
