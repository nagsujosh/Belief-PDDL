import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_collapse_index(trace):
    solvable = [step.get("solvable_worlds", 0) for step in trace]
    for idx in range(1, len(trace) - 1):
        if solvable[idx - 1] > 0 and solvable[idx] == 0 and solvable[idx + 1] == 0:
            return idx
    for idx in range(1, len(trace)):
        if solvable[idx - 1] > 0 and solvable[idx] == 0:
            return idx
    return None


def extract_episode_window(seed, episode, trace, window):
    collapse_idx = find_collapse_index(trace)
    if collapse_idx is None:
        collapse_idx = min(len(trace) - 1, window)
    start = max(0, collapse_idx - window)
    end = min(len(trace), collapse_idx + window + 1)
    return {
        "seed": seed,
        "episode": episode,
        "collapse_index": collapse_idx,
        "window": trace[start:end],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Common file prefix before the seed number")
    parser.add_argument("--seeds", required=True, help="Comma-separated seed list")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    windows = []
    for seed in seeds:
        path = Path(f"{args.prefix}{seed}_perception_learned_head.jsonl")
        rows = load_jsonl(path)
        for ep in rows:
            windows.append(
                extract_episode_window(
                    seed=seed,
                    episode=ep.get("episode"),
                    trace=ep.get("action_trace", []),
                    window=args.window,
                )
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"seeds": seeds, "windows": windows}, indent=2))
    print(f"Saved collapse-window report to {output}")
    for item in windows:
        print(f"seed={item['seed']} episode={item['episode']} collapse_t={item['collapse_index']}")


if __name__ == "__main__":
    main()
