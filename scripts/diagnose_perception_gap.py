import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.mocks.blocksworld_env import MockBlocksworldEnv
from src.belief.projection import BeliefProjector
from src.perception.blocksworld_predicates import PREDICATE_ORDER, grounded_predicates, semantic_queries
from src.perception.calibrate import TemperatureScalar
from src.perception.clip_vision import CLIPVisionBackbone
from src.perception.unary_head import UnaryPredicateHead
from src.planning.deterministic_planner import DeterministicPlanner


@dataclass
class FrameRecord:
    image: np.ndarray
    gt_predicates: set
    visible_objects: List[str]


def random_valid_action(env: MockBlocksworldEnv) -> Tuple[str, List[str]]:
    actions: List[Tuple[str, List[str]]] = []
    blocks = env.blocks

    hidden_blocks = [b for b in blocks if b not in env.state["visible"]]
    if hidden_blocks:
        actions.extend([("reveal_side", [b]) for b in hidden_blocks])

    if env.state["arm_empty"]:
        for b in blocks:
            if b in env.state["clear"] and b in env.state["on_table"] and b in env.state["visible"]:
                actions.append(("pickup", [b]))
        for child, parent in env.state["on"].items():
            if child in env.state["clear"] and child in env.state["visible"]:
                actions.append(("unstack", [child, parent]))
    else:
        held = env.state["holding"]
        if held is not None:
            actions.append(("putdown", [held]))
            for b in blocks:
                if b != held and b in env.state["clear"] and b in env.state["visible"]:
                    actions.append(("stack", [held, b]))

    if not actions:
        return "reveal_side", [random.choice(blocks)]
    return random.choice(actions)


def collect_frame_records(num_episodes: int, max_steps: int, seed: int) -> List[FrameRecord]:
    random.seed(seed)
    np.random.seed(seed)
    records: List[FrameRecord] = []

    for _ in range(num_episodes):
        env = MockBlocksworldEnv(num_blocks=3)
        obs = env.reset()
        records.append(
            FrameRecord(
                image=obs.rgb.copy(),
                gt_predicates=set(obs.gt_predicates),
                visible_objects=list(obs.visible_objects),
            )
        )

        for _step in range(max_steps):
            action, args = random_valid_action(env)
            obs, _reward, _done = env.step(action, args)
            records.append(
                FrameRecord(
                    image=obs.rgb.copy(),
                    gt_predicates=set(obs.gt_predicates),
                    visible_objects=list(obs.visible_objects),
                )
            )
    return records


class PerceptionComparator:
    def __init__(self, device: str, learned_head_path: str | None, calibration_path: str | None):
        self.vlm = CLIPVisionBackbone(device=device)
        self.device = self.vlm.device
        self._predicate_texts = {pred: semantic_queries(pred)[0] for pred in PREDICATE_ORDER}
        self._text_embeds = self.vlm.encode_texts([self._predicate_texts[p] for p in PREDICATE_ORDER]).detach()

        self.learned_head = None
        if learned_head_path:
            ckpt = torch.load(learned_head_path, map_location="cpu")
            embed_dim = ckpt.get("embed_dim", 512)
            self.learned_head = UnaryPredicateHead(visual_dim=embed_dim, text_dim=embed_dim, hidden_dim=256).to(self.device)
            state_dict = ckpt["state_dict"]
            if any(key.startswith("head.") for key in state_dict):
                state_dict = {key.removeprefix("head."): value for key, value in state_dict.items()}
            self.learned_head.load_state_dict(state_dict)
            self.learned_head.eval()

        self.calibrators: Dict[str, TemperatureScalar] = {}
        if calibration_path:
            ckpt = torch.load(calibration_path, map_location="cpu")
            for pred, temp in ckpt["temperatures"].items():
                scalar = TemperatureScalar(init_temp=float(temp))
                with torch.no_grad():
                    scalar.temperature.copy_(torch.tensor([float(temp)]))
                scalar.eval()
                self.calibrators[pred] = scalar

    def predict_probs(self, image: np.ndarray, predicates: List[str], mode: str) -> Dict[str, float]:
        image_embed = self.vlm.encode_image(image)
        outputs: Dict[str, float] = {}
        for pred in predicates:
            outputs[pred] = self._predict_prob(pred, image_embed, mode)
        return outputs

    def _predict_prob(self, predicate: str, image_embed: torch.Tensor, mode: str) -> float:
        pos_text, neg_text = semantic_queries(predicate)
        if mode == "learned_head":
            if self.learned_head is None:
                raise ValueError("learned_head mode requested but no learned head checkpoint was provided")
            idx = PREDICATE_ORDER.index(predicate)
            text_embed = self._text_embeds[idx:idx + 1].to(self.device)
            with torch.no_grad():
                logit = self.learned_head(image_embed, text_embed)
                return torch.sigmoid(logit)[0, 0].item()

        if not neg_text:
            return 0.5

        text_embeds = self.vlm.encode_texts([pos_text, neg_text])
        with torch.no_grad():
            logit_scale = self.vlm.model.logit_scale.exp()
            logits = logit_scale * (image_embed @ text_embeds.T)
            pair_probs = torch.softmax(logits[0], dim=0)
        zero_shot_prob = float(pair_probs[0].cpu())
        if mode == "calibrated":
            scalar = self.calibrators.get(predicate)
            if scalar is not None:
                zero_shot_prob = max(1e-4, min(1.0 - 1e-4, zero_shot_prob))
                prob_logit = torch.tensor([[math.log(zero_shot_prob / (1.0 - zero_shot_prob))]], dtype=torch.float32)
                with torch.no_grad():
                    return scalar(prob_logit)[0, 0].item()
        return zero_shot_prob


def apply_noise(prob: float, noise: float) -> float:
    if noise <= 0:
        return max(1e-4, min(1.0 - 1e-4, prob))
    mixed = ((1.0 - noise) * prob) + (noise * 0.5)
    return max(1e-4, min(1.0 - 1e-4, mixed))


def binary_stats(entries: List[Tuple[float, int]]) -> Dict[str, float]:
    if not entries:
        return {"accuracy": 0.0, "brier": 0.0, "avg_prob_true": 0.0, "avg_prob_false": 0.0}
    probs = np.array([p for p, _ in entries], dtype=np.float32)
    labels = np.array([y for _, y in entries], dtype=np.float32)
    preds = (probs >= 0.5).astype(np.float32)
    true_mask = labels == 1
    false_mask = labels == 0
    return {
        "accuracy": float((preds == labels).mean()),
        "brier": float(((probs - labels) ** 2).mean()),
        "avg_prob_true": float(probs[true_mask].mean()) if true_mask.any() else 0.0,
        "avg_prob_false": float(probs[false_mask].mean()) if false_mask.any() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--learned-head-path", type=str, default="outputs/perception/perception_compare_e200_learned_head.pt")
    parser.add_argument("--calibration-path", type=str, default="outputs/perception/perception_compare_e200_calibration.pt")
    parser.add_argument("--output", type=str, default="outputs/perception/perception_gap_diagnosis.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Collecting diagnostic frames with seed={args.seed}...")
    records = collect_frame_records(args.episodes, args.max_steps, args.seed)
    blocks = [f"block_{i}" for i in range(3)]
    predicates = grounded_predicates(blocks)
    critical_prefixes = ("arm_empty()", "clear(", "holding(", "on(", "on_table(")
    critical_preds = [pred for pred in predicates if pred.startswith(critical_prefixes) or pred == "arm_empty()"]

    print(f"Loading perception models on {args.device}...")
    comparator = PerceptionComparator(
        device=args.device,
        learned_head_path=args.learned_head_path if os.path.exists(args.learned_head_path) else None,
        calibration_path=args.calibration_path if os.path.exists(args.calibration_path) else None,
    )
    projector = BeliefProjector("domains/blocksworld/constraints.yaml")
    planner = DeterministicPlanner("domains/blocksworld/domain.pddl")
    goal_str = "(on block_0 block_1)"
    modes = ["zero_shot", "calibrated", "learned_head"]
    if comparator.learned_head is None:
        modes.remove("learned_head")
    if not comparator.calibrators:
        modes.remove("calibrated")

    per_mode_entries: Dict[str, Dict[str, List[Tuple[float, int]]]] = {
        mode: defaultdict(list) for mode in modes
    }
    per_mode_critical_entries: Dict[str, Dict[str, List[Tuple[float, int]]]] = {
        mode: defaultdict(list) for mode in modes
    }
    per_mode_world_stats = {
        mode: {"solvable_ratio_sum": 0.0, "frames": 0, "solvable_frames": 0}
        for mode in modes
    }
    disagreement = defaultdict(list)

    for record in records:
        gt = record.gt_predicates
        all_probs = {
            mode: comparator.predict_probs(record.image, predicates, mode)
            for mode in modes
        }
        for mode, probs in all_probs.items():
            noisy_probs = {pred: apply_noise(prob, args.noise) for pred, prob in probs.items()}

            for pred, prob in noisy_probs.items():
                label = 1 if pred in gt else 0
                per_mode_entries[mode][pred].append((prob, label))
                if pred in critical_preds:
                    per_mode_critical_entries[mode][pred].append((prob, label))

            worlds = projector.project_top_k_map_states(noisy_probs, k=args.k)
            solvable_worlds = 0
            for world in worlds:
                result = planner.plan_with_diagnostics(world, goal_str, blocks)
                if result.solvable and result.plan:
                    solvable_worlds += 1
            ratio = solvable_worlds / max(1, len(worlds))
            per_mode_world_stats[mode]["solvable_ratio_sum"] += ratio
            per_mode_world_stats[mode]["frames"] += 1
            if solvable_worlds > 0:
                per_mode_world_stats[mode]["solvable_frames"] += 1

        if "zero_shot" in all_probs and "learned_head" in all_probs:
            for pred in critical_preds:
                disagreement[pred].append(abs(all_probs["zero_shot"][pred] - all_probs["learned_head"][pred]))

    summary = {
        "config": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "device": str(comparator.device),
            "noise": args.noise,
            "k": args.k,
            "num_frames": len(records),
        },
        "modes": {},
        "zero_shot_vs_learned_disagreement": {},
    }

    for mode in modes:
        predicate_metrics = {
            pred: binary_stats(per_mode_entries[mode][pred])
            for pred in predicates
        }
        critical_metrics = {
            pred: binary_stats(per_mode_critical_entries[mode][pred])
            for pred in critical_preds
        }
        world_stats = per_mode_world_stats[mode]
        frames = max(1, world_stats["frames"])
        summary["modes"][mode] = {
            "overall": binary_stats(
                [entry for pred in predicates for entry in per_mode_entries[mode][pred]]
            ),
            "critical_overall": binary_stats(
                [entry for pred in critical_preds for entry in per_mode_critical_entries[mode][pred]]
            ),
            "planning_proxy": {
                "avg_solvable_world_ratio": world_stats["solvable_ratio_sum"] / frames,
                "solvable_frame_fraction": world_stats["solvable_frames"] / frames,
            },
            "per_predicate": predicate_metrics,
            "critical_predicates": critical_metrics,
        }

    for pred, values in disagreement.items():
        if values:
            summary["zero_shot_vs_learned_disagreement"][pred] = float(np.mean(values))

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved diagnostic report to {args.output}")
    for mode, payload in summary["modes"].items():
        overall = payload["overall"]
        proxy = payload["planning_proxy"]
        print(
            f"{mode:<12} "
            f"acc={overall['accuracy']:.3f} "
            f"brier={overall['brier']:.3f} "
            f"solvable_ratio={proxy['avg_solvable_world_ratio']:.3f} "
            f"solvable_frames={proxy['solvable_frame_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
