import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.mocks.blocksworld_env import MockBlocksworldEnv
from src.perception.clip_vision import CLIPVisionBackbone
from src.perception.calibrate import TemperatureScalar
from src.perception.unary_head import UnaryPredicateHead
from src.perception.blocksworld_predicates import PREDICATE_ORDER, semantic_queries


@dataclass
class FrameRecord:
    image: np.ndarray
    gt_predicates: set


class SharedPredicateHead(nn.Module):
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.head = UnaryPredicateHead(visual_dim=embed_dim, text_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        return self.head(image_embeds, text_embeds)



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
            if child in env.state["clear"] and env.state["arm_empty"] and child in env.state["visible"]:
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
        records.append(FrameRecord(image=obs.rgb.copy(), gt_predicates=set(obs.gt_predicates)))

        for _step in range(max_steps):
            action, args = random_valid_action(env)
            obs, _reward, done = env.step(action, args)
            records.append(FrameRecord(image=obs.rgb.copy(), gt_predicates=set(obs.gt_predicates)))
            if done:
                break

    return records


def split_records(records: List[FrameRecord], train_frac: float, val_frac: float):
    records = list(records)
    random.shuffle(records)
    n = len(records)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return records[:train_end], records[train_end:val_end], records[val_end:]


def encode_records(records: List[FrameRecord], backbone: CLIPVisionBackbone, predicate_texts: Dict[str, str], device: str):
    image_embeds = []
    label_matrix = []

    for record in records:
        image_embed = backbone.encode_image(record.image).detach().cpu().squeeze(0)
        image_embeds.append(image_embed)
        label_matrix.append([1.0 if pred in record.gt_predicates else 0.0 for pred in PREDICATE_ORDER])

    text_embeds = backbone.encode_texts([predicate_texts[p] for p in PREDICATE_ORDER]).detach().cpu()
    return {
        "image_embeds": torch.stack(image_embeds),
        "labels": torch.tensor(label_matrix, dtype=torch.float32),
        "text_embeds": text_embeds,
    }


def zero_shot_logits(backbone: CLIPVisionBackbone, records: List[FrameRecord]) -> torch.Tensor:
    probs = []
    for record in records:
        image_embed = backbone.encode_image(record.image)
        row = []
        for pred in PREDICATE_ORDER:
            pos, neg = semantic_queries(pred)
            if neg:
                prob = backbone.zero_shot_prob(image_embed, pos, neg)
            else:
                prob = 0.5
            prob = max(1e-4, min(1.0 - 1e-4, prob))
            row.append(math.log(prob / (1.0 - prob)))
        probs.append(row)
    return torch.tensor(probs, dtype=torch.float32)


def fit_temperature_per_predicate(val_logits: torch.Tensor, val_labels: torch.Tensor) -> List[TemperatureScalar]:
    calibrators = []
    for idx in range(val_logits.shape[1]):
        scalar = TemperatureScalar(init_temp=1.0)
        scalar.calibrate(val_logits[:, idx:idx+1], val_labels[:, idx:idx+1], epochs=50, lr=0.05)
        calibrators.append(scalar.eval())
    return calibrators


def apply_calibrators(logits: torch.Tensor, calibrators: List[TemperatureScalar]) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for idx, scalar in enumerate(calibrators):
            probs = scalar(logits[:, idx:idx+1]).squeeze(1)
            outputs.append(probs)
    return torch.stack(outputs, dim=1)


def train_shared_head(
    train_pack: Dict[str, torch.Tensor],
    val_pack: Dict[str, torch.Tensor],
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> SharedPredicateHead:
    model = SharedPredicateHead(embed_dim=train_pack["image_embeds"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    def build_examples(pack: Dict[str, torch.Tensor]):
        image_embeds = pack["image_embeds"]
        labels = pack["labels"]
        text_embeds = pack["text_embeds"]
        x_img = image_embeds.repeat_interleave(len(PREDICATE_ORDER), dim=0)
        x_txt = text_embeds.repeat(image_embeds.shape[0], 1)
        y = labels.reshape(-1, 1)
        return x_img, x_txt, y

    train_img, train_txt, train_y = build_examples(train_pack)
    val_img, val_txt, val_y = build_examples(val_pack)

    best_state = None
    best_val = float("inf")
    indices = torch.arange(train_y.shape[0])

    for _epoch in range(epochs):
        perm = indices[torch.randperm(indices.numel())]
        model.train()
        for start in range(0, perm.numel(), batch_size):
            batch_idx = perm[start:start + batch_size]
            img = train_img[batch_idx].to(device)
            txt = train_txt[batch_idx].to(device)
            y = train_y[batch_idx].to(device)
            logits = model(img, txt)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_img.to(device), val_txt.to(device))
            val_loss = criterion(val_logits, val_y.to(device)).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_shared_head(model: SharedPredicateHead, pack: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
    probs = []
    with torch.no_grad():
        for image_embed in pack["image_embeds"]:
            img = image_embed.unsqueeze(0).repeat(len(PREDICATE_ORDER), 1).to(device)
            txt = pack["text_embeds"].to(device)
            logits = model(img, txt).squeeze(1)
            probs.append(torch.sigmoid(logits).cpu())
    return torch.stack(probs)


def compute_metrics(probs: torch.Tensor, labels: torch.Tensor) -> Dict[str, Dict[str, float]]:
    preds = (probs >= 0.5).float()
    metrics = {}

    def binary_stats(prob_col: torch.Tensor, pred_col: torch.Tensor, y: torch.Tensor):
        acc = (pred_col == y).float().mean().item()
        tp = ((pred_col == 1) & (y == 1)).sum().item()
        fp = ((pred_col == 1) & (y == 0)).sum().item()
        fn = ((pred_col == 0) & (y == 1)).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        brier = ((prob_col - y.float()) ** 2).mean().item()
        return acc, precision, recall, brier

    overall_acc = (preds == labels).float().mean().item()
    overall_brier = ((probs - labels) ** 2).mean().item()
    metrics["overall"] = {
        "accuracy": overall_acc,
        "brier": overall_brier,
    }

    per_pred = {}
    for idx, pred in enumerate(PREDICATE_ORDER):
        probs_col = probs[:, idx]
        pred_col = preds[:, idx]
        label_col = labels[:, idx]
        acc, precision, recall, brier = binary_stats(probs_col, pred_col, label_col)
        per_pred[pred] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "brier": brier,
            "positive_rate": float(label_col.mean().item()),
        }
    metrics["per_predicate"] = per_pred
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="outputs/perception/perception_comparison.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Collecting synthetic dataset with seed={args.seed}...")
    records = collect_frame_records(args.episodes, args.max_steps, args.seed)
    train_records, val_records, test_records = split_records(records, args.train_frac, args.val_frac)
    print(f"Frames: train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    print(f"Loading CLIP backbone on {device}...")
    backbone = CLIPVisionBackbone(device=device)

    predicate_texts = {pred: semantic_queries(pred)[0] for pred in PREDICATE_ORDER}
    train_pack = encode_records(train_records, backbone, predicate_texts, device)
    val_pack = encode_records(val_records, backbone, predicate_texts, device)
    test_pack = encode_records(test_records, backbone, predicate_texts, device)

    print("Evaluating zero-shot CLIP...")
    val_zero_logits = zero_shot_logits(backbone, val_records)
    test_zero_logits = zero_shot_logits(backbone, test_records)
    test_zero_probs = torch.sigmoid(test_zero_logits)

    print("Fitting calibrated CLIP temperatures...")
    calibrators = fit_temperature_per_predicate(val_zero_logits, val_pack["labels"])
    test_cal_probs = apply_calibrators(test_zero_logits, calibrators)

    print("Training shared learned predicate head...")
    head = train_shared_head(
        train_pack=train_pack,
        val_pack=val_pack,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )
    test_head_probs = predict_shared_head(head, test_pack, device)

    summary = {
        "config": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "device": device,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
        },
        "dataset_sizes": {
            "train_frames": len(train_records),
            "val_frames": len(val_records),
            "test_frames": len(test_records),
        },
        "methods": {
            "zero_shot_clip": compute_metrics(test_zero_probs, test_pack["labels"]),
            "calibrated_clip": compute_metrics(test_cal_probs, test_pack["labels"]),
            "learned_head": compute_metrics(test_head_probs, test_pack["labels"]),
        },
        "temperatures": {
            pred: float(torch.clamp(calibrators[idx].temperature.detach().cpu(), min=1e-3).item())
            for idx, pred in enumerate(PREDICATE_ORDER)
        },
    }

    torch.save(
        {
            "state_dict": head.state_dict(),
            "predicate_order": PREDICATE_ORDER,
            "embed_dim": int(train_pack["image_embeds"].shape[1]),
        },
        os.path.splitext(args.output)[0] + "_learned_head.pt",
    )
    torch.save(
        {
            "predicate_order": PREDICATE_ORDER,
            "temperatures": summary["temperatures"],
        },
        os.path.splitext(args.output)[0] + "_calibration.pt",
    )

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved comparison report to {args.output}")
    for method_name, metrics in summary["methods"].items():
        print(
            f"{method_name:<18} "
            f"acc={metrics['overall']['accuracy']:.3f} "
            f"brier={metrics['overall']['brier']:.3f}"
        )


if __name__ == "__main__":
    main()
