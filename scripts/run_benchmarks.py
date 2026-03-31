import os
import sys
import json
import random
import argparse
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.mocks.blocksworld_env import MockBlocksworldEnv
from src.belief.state import PredicateBelief
from src.belief.update import BeliefUpdater
from src.belief.projection import BeliefProjector
from src.planning.deterministic_planner import DeterministicPlanner
from src.planning.sample_belief_planner import SampleBeliefPlanner
from src.perception.clip_vision import CLIPVisionBackbone
from src.perception.calibrate import TemperatureScalar
from src.perception.unary_head import UnaryPredicateHead
from src.perception.blocksworld_predicates import grounded_predicates, semantic_queries, PREDICATE_ORDER
import torch
import numpy as np

class AIBenchmarker:
    def __init__(
        self,
        mode="full_top_k",
        noise_level=0.3,
        k_worlds=3,
        device="auto",
        alpha=1.0,
        decay=1.0,
        seed=0,
        perception_mode="zero_shot",
        learned_head_path=None,
        calibration_path=None,
    ):
        print(
            f"Initializing AI Benchmark. Mode: {mode} | Noise: {noise_level} | "
            f"K: {k_worlds} | Alpha: {alpha} | Decay: {decay} | Perception: {perception_mode}"
        )
        self.mode = mode
        self.noise = noise_level
        self.k = k_worlds
        self.alpha = alpha
        self.decay = decay
        self.seed = seed
        self.perception_mode = perception_mode
        self.learned_head_path = learned_head_path
        self.calibration_path = calibration_path
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.env = MockBlocksworldEnv(num_blocks=3, seed=self.seed)
        self.updater = BeliefUpdater(alpha=self.alpha, decay=self.decay)
        self.projector = BeliefProjector("domains/blocksworld/constraints.yaml")
        
        print("Loading Authentic CLIP Vision Backbone into Benchmark Array...")
        self.vlm = CLIPVisionBackbone(device=self.device)
        # Warmup: one forward pass so first episode timing isn't skewed
        _w = self.vlm.encode_image(np.zeros((256, 256, 3), dtype=np.uint8))
        print(f"[CLIP] Warmup embed shape: {_w.shape}")
        self._predicate_texts = {pred: semantic_queries(pred)[0] for pred in PREDICATE_ORDER}
        self._text_embeds = self.vlm.encode_texts([self._predicate_texts[p] for p in PREDICATE_ORDER]).detach()
        self.learned_head = None
        self.calibrators = {}
        if self.perception_mode == "learned_head":
            if not self.learned_head_path:
                raise ValueError("--learned-head-path is required for perception_mode=learned_head")
            ckpt = torch.load(self.learned_head_path, map_location="cpu")
            embed_dim = ckpt.get("embed_dim", 512)
            self.learned_head = UnaryPredicateHead(visual_dim=embed_dim, text_dim=embed_dim, hidden_dim=256).to(self.device)
            state_dict = ckpt["state_dict"]
            if any(key.startswith("head.") for key in state_dict):
                # Older comparison checkpoints save the wrapper SharedPredicateHead state.
                state_dict = {
                    key.removeprefix("head."): value
                    for key, value in state_dict.items()
                }
            self.learned_head.load_state_dict(state_dict)
            self.learned_head.eval()
        if self.perception_mode == "calibrated":
            if not self.calibration_path:
                raise ValueError("--calibration-path is required for perception_mode=calibrated")
            ckpt = torch.load(self.calibration_path, map_location="cpu")
            for pred, temp in ckpt["temperatures"].items():
                scalar = TemperatureScalar(init_temp=float(temp))
                with torch.no_grad():
                    scalar.temperature.copy_(torch.tensor([float(temp)]))
                scalar.eval()
                self.calibrators[pred] = scalar
        
        # Disable info gain/sensing depending on mode
        sensing = ["reveal_side"] if self.mode == "full_top_k" else []
        self.planner = SampleBeliefPlanner(
            DeterministicPlanner("domains/blocksworld/domain.pddl"), 
            sensing_actions=sensing
        )
        self.metrics = []

    def _goal_already_satisfied(self) -> bool:
        return self.env.state.get("on", {}).get("block_0") == "block_1"

    def _reset_unsolved_episode(self, episode_seed: int, max_attempts: int = 50):
        for attempt in range(max_attempts):
            accepted_seed = episode_seed + attempt
            self.env.reset(seed=accepted_seed)
            if not self._goal_already_satisfied():
                if attempt > 0:
                    print(f"Reset accepted after {attempt + 1} attempts to avoid pre-solved initial state.")
                return {
                    "episode_seed": episode_seed,
                    "accepted_seed": accepted_seed,
                    "attempt": attempt,
                }
        raise RuntimeError("Failed to sample an unsolved Blocksworld episode after repeated resets.")

    def _apply_noise(self, prob: float) -> float:
        if self.noise <= 0:
            return prob
        # Higher noise should represent less informative perception, not fully adversarial inversion.
        mixed = ((1.0 - self.noise) * prob) + (self.noise * 0.5)
        return max(1e-4, min(1.0 - 1e-4, mixed))

    def _predict_prob(self, predicate: str, image_embed: torch.Tensor) -> float:
        pos_text, neg_text = semantic_queries(predicate)
        if self.perception_mode == "learned_head":
            idx = PREDICATE_ORDER.index(predicate)
            text_embed = self._text_embeds[idx:idx+1].to(self.device)
            with torch.no_grad():
                logit = self.learned_head(image_embed, text_embed)
                prob = torch.sigmoid(logit)[0, 0].item()
            return prob

        if not neg_text:
            return 0.5

        text_embeds = self.vlm.encode_texts([pos_text, neg_text])
        with torch.no_grad():
            logit_scale = self.vlm.model.logit_scale.exp()
            logits = logit_scale * (image_embed @ text_embeds.T)
            pair_probs = torch.softmax(logits[0], dim=0)
        zero_shot_prob = float(pair_probs[0].cpu())
        if self.perception_mode == "calibrated":
            scalar = self.calibrators.get(predicate)
            if scalar is not None:
                zero_shot_prob = max(1e-4, min(1.0 - 1e-4, zero_shot_prob))
                prob_logit = torch.tensor(
                    [[math.log(zero_shot_prob / (1.0 - zero_shot_prob))]],
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    prob = scalar(prob_logit)[0, 0].item()
                return prob
        return zero_shot_prob

    def _set_hard_belief(self, belief_state: PredicateBelief, predicate: str, is_true: bool, timestep: int, source: str):
        belief_state.set_belief(predicate, 0.99 if is_true else 0.01, timestep=timestep, source=source)

    def _sync_visibility_from_obs(self, belief_state: PredicateBelief, obs, timestep: int):
        visible_now = set(obs.visible_objects)
        for block in self.env.blocks:
            self._set_hard_belief(
                belief_state,
                f"visible({block})",
                block in visible_now,
                timestep=timestep,
                source="env_visible"
            )

    def _sync_structural_neighborhood(self, belief_state: PredicateBelief, anchors, obs, timestep: int, source: str):
        """
        After a failed action in the mock environment, hard-sync the local structural
        neighborhood around the acted-on blocks. This preserves alternative solvable
        worlds instead of leaving the projector with stale hand/table/support beliefs.
        """
        gt = set(obs.gt_predicates)

        def has(pred: str) -> bool:
            return pred in gt

        impacted = set(anchors)
        for anchor in anchors:
            for block in self.env.blocks:
                if block == anchor:
                    continue
                if has(f"on({anchor},{block})") or has(f"on({block},{anchor})"):
                    impacted.add(block)

        self._set_hard_belief(belief_state, "arm_empty()", has("arm_empty()"), timestep, source=source)

        for block in impacted:
            for pred in [
                f"clear({block})",
                f"on_table({block})",
                f"holding({block})",
                f"visible({block})",
            ]:
                self._set_hard_belief(belief_state, pred, has(pred), timestep, source=source)

            for other in self.env.blocks:
                if other == block:
                    continue
                for pred in [f"on({block},{other})", f"on({other},{block})"]:
                    self._set_hard_belief(belief_state, pred, has(pred), timestep, source=source)

    def _sync_action_effects(self, belief_state: PredicateBelief, cmd: str, args, obs, timestep: int):
        gt = set(obs.gt_predicates)

        def has(pred: str) -> bool:
            return pred in gt

        if cmd == "pickup" and args:
            x = args[0]
            for pred in [f"holding({x})", f"on_table({x})", f"clear({x})", "arm_empty()"]:
                self._set_hard_belief(belief_state, pred, has(pred), timestep, source="action_effect")
        elif cmd == "putdown" and args:
            x = args[0]
            for pred in [f"holding({x})", f"on_table({x})", f"clear({x})", "arm_empty()"]:
                self._set_hard_belief(belief_state, pred, has(pred), timestep, source="action_effect")
        elif cmd == "stack" and len(args) == 2:
            x, y = args
            for pred in [f"holding({x})", f"on({x},{y})", f"clear({x})", f"clear({y})", "arm_empty()"]:
                self._set_hard_belief(belief_state, pred, has(pred), timestep, source="action_effect")
        elif cmd == "unstack" and len(args) == 2:
            x, y = args
            for pred in [f"holding({x})", f"on({x},{y})", f"clear({x})", f"clear({y})", "arm_empty()"]:
                self._set_hard_belief(belief_state, pred, has(pred), timestep, source="action_effect")

    def _apply_failed_action_correction(self, belief_state: PredicateBelief, cmd: str, args, obs, timestep: int):
        gt = set(obs.gt_predicates)

        def has(pred: str) -> bool:
            return pred in gt

        if cmd == "putdown" and args:
            x = args[0]
            self._set_hard_belief(belief_state, f"holding({x})", False, timestep, source="failed_action")
            self._set_hard_belief(belief_state, "arm_empty()", has("arm_empty()"), timestep, source="failed_action")
        elif cmd == "pickup" and args:
            x = args[0]
            self._sync_structural_neighborhood(belief_state, [x], obs, timestep, source="failed_action")
        elif cmd == "stack" and len(args) == 2:
            x, y = args
            self._sync_structural_neighborhood(belief_state, [x, y], obs, timestep, source="failed_action")
            self._set_hard_belief(belief_state, f"on({x},{y})", False, timestep, source="failed_action")
        elif cmd == "unstack" and len(args) == 2:
            x, y = args
            self._sync_structural_neighborhood(belief_state, [x, y], obs, timestep, source="failed_action")
            self._set_hard_belief(belief_state, f"holding({x})", False, timestep, source="failed_action")

    def _run_vlm(self, rgb, revealed_blocks):
        """Score predicates from real CLIP embeddings of the rendered scene."""
        image_embed = self.vlm.encode_image(rgb)  # (1, D) unit-norm tensor
        raw_logits = {}

        preds_to_guess = grounded_predicates(self.env.blocks)

        gt = self.env._get_gt_predicates()  # used only for revealed blocks
        for p in preds_to_guess:
            is_revealed = any(b in p for b in revealed_blocks)
            if is_revealed:
                revealed_prob = 0.95 if p in gt else 0.05
                raw_logits[p] = self._apply_noise(revealed_prob)
                continue

            prob = self._predict_prob(p, image_embed)
            raw_logits[p] = self._apply_noise(prob)

        return raw_logits

    def run_sweep(self, max_episodes=5):
        for ep in range(max_episodes):
            print(f"\nEvaluating Episode: {ep} - Mode: {self.mode}")
            episode_seed = self.seed + ep * 1000
            reset_info = self._reset_unsolved_episode(episode_seed=episode_seed)
            self.planner.reset_episode_state()
            belief_state = PredicateBelief()
            goal_str = "(on block_0 block_1)" # Static goal synchronized to internal Mock evaluator in valid PDDL format
            
            ep_trace = {
                "episode": ep, 
                "mode": self.mode,
                "noise_level": self.noise,
                "alpha": self.alpha,
                "decay": self.decay,
                "seed": episode_seed,
                "accepted_seed": reset_info["accepted_seed"],
                "reset_attempt": reset_info["attempt"],
                "perception_mode": self.perception_mode,
                "success": False, 
                "steps": 0, 
                "inconsistency_events": 0,
                "sensing_actions": 0,
                "replans": 0,
                "action_trace": []
            }
            
            revealed_blocks = set()
            
            for t in range(25):
                obs = self.env._get_obs()
                # In the mock benchmark, visibility is directly observable from the renderer/env.
                # Keep it as a hard fact instead of re-noising it through CLIP each step.
                self._sync_visibility_from_obs(belief_state, obs, timestep=t)
                raw_logits = self._run_vlm(obs.rgb, revealed_blocks)
                
                # Belief Structure Ablation
                if self.mode == "threshold":
                    # Discrete instantaneous states mapping (No continuous marginals)
                    for pred, prob in raw_logits.items():
                        if pred.startswith("visible("):
                            continue
                        belief_state.set_belief(pred, prob, t)
                else:
                    # Factorized Symbolics mapping
                    for pred, prob in raw_logits.items():
                        if pred.startswith("visible("):
                            continue
                        n = self.updater.update(belief_state.get_belief(pred), prob)
                        belief_state.set_belief(pred, n, timestep=t)

                # Metric: Measure raw VLM Inconsistency
                for b in self.env.blocks:
                    # Tracking physical contradictions (Targeting Table 2 / Figure 3)
                    if belief_state.get_belief(f"holding({b})") > 0.5 and belief_state.get_belief(f"on_table({b})") > 0.5:
                        ep_trace["inconsistency_events"] += 1

                # Projection Structure Ablation
                if self.mode in ["threshold", "belief_no_verifier"]:
                    # Raw states passing straight to Pyperplan
                    top_k = [{k: v > 0.5 for k, v in belief_state.probs.items()}]
                elif self.mode == "belief_plus_verifier":
                    # Single MAP
                    top_k = self.projector.project_top_k_map_states(belief_state.probs, k=1)
                else:
                    # Full Top-K Constraints
                    top_k = self.projector.project_top_k_map_states(belief_state.probs, k=self.k)
                
                # Deterministic Planning Execution
                planner_objects = list(self.env.blocks)
                cmd, args = self.planner.select_action(belief_state.probs, top_k, goal_str, planner_objects)
                planner_debug = dict(self.planner.last_debug)
                pre_gt = set(obs.gt_predicates)
                if cmd == "reveal_side": 
                    ep_trace["sensing_actions"] += 1
                    revealed_blocks.add(args[0])
                else:
                    ep_trace["replans"] += 1
                
                obs, reward, done = self.env.step(cmd, args)
                post_gt = set(obs.gt_predicates)
                state_changed = pre_gt != post_gt

                if cmd == "reveal_side":
                    if args:
                        self._set_hard_belief(belief_state, f"visible({args[0]})", args[0] in obs.visible_objects, timestep=t, source="sensing")
                elif state_changed:
                    self._sync_action_effects(belief_state, cmd, args, obs, timestep=t)
                else:
                    self._apply_failed_action_correction(belief_state, cmd, args, obs, timestep=t)

                self.planner.register_action_feedback(cmd, args, state_changed)

                ep_trace["action_trace"].append({
                    "t": t,
                    "action": cmd,
                    "args": args,
                    "changed_state": state_changed,
                    "solvable_worlds": planner_debug.get("solvable_worlds", 0),
                    "num_worlds": planner_debug.get("num_worlds", len(top_k)),
                    "weighted_solvable_mass": planner_debug.get("weighted_solvable_mass", 0.0),
                    "weighted_blocked_mass": planner_debug.get("weighted_blocked_mass", 0.0),
                    "first_action_votes": planner_debug.get("first_action_votes", {}),
                    "blocked_first_actions": planner_debug.get("blocked_first_actions", {}),
                    "blocked_actions": planner_debug.get("blocked_actions", {}),
                    "action_scores": planner_debug.get("action_scores", {}),
                    "best_plan_action": planner_debug.get("best_plan_action"),
                    "best_score": planner_debug.get("best_score", 0.0),
                    "decision_threshold": planner_debug.get("decision_threshold", 0.40),
                    "decision_reason": planner_debug.get("decision_reason", "unknown"),
                    "selected_action": planner_debug.get("selected_action", cmd),
                    "selected_args": planner_debug.get("selected_args", args),
                    "selected_sensing_target": planner_debug.get("selected_sensing_target"),
                    "selected_sensing_ig": planner_debug.get("selected_sensing_ig"),
                })
                if done:
                    ep_trace["success"] = True
                    ep_trace["steps"] = t
                    break

            if not ep_trace["success"]: ep_trace["steps"] = 25 # Timeout penalty
            self.metrics.append(ep_trace)

        out_path = (
            f"outputs/benchmarks/{self.mode}_noise_{self.noise}"
            f"_alpha_{self.alpha}_decay_{self.decay}"
            f"_seed_{self.seed}"
            f"_perception_{self.perception_mode}.jsonl"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            for m in self.metrics:
                f.write(json.dumps(m) + "\n")
        print(f"Benchmark Array Loop Exported: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full_top_k", choices=["threshold", "belief_no_verifier", "belief_plus_verifier", "full_top_k_no_sense", "full_top_k"])
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perception-mode", type=str, default="zero_shot", choices=["zero_shot", "calibrated", "learned_head"])
    parser.add_argument("--learned-head-path", type=str, default=None)
    parser.add_argument("--calibration-path", type=str, default=None)
    args = parser.parse_args()
    
    benchmarker = AIBenchmarker(
        mode=args.mode,
        noise_level=args.noise,
        k_worlds=args.k,
        device=args.device,
        alpha=args.alpha,
        decay=args.decay,
        seed=args.seed,
        perception_mode=args.perception_mode,
        learned_head_path=args.learned_head_path,
        calibration_path=args.calibration_path,
    )
    benchmarker.run_sweep(max_episodes=args.episodes)
