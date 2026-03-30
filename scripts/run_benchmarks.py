import os
import sys
import json
import random
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.mocks.blocksworld_env import MockBlocksworldEnv
from src.belief.state import PredicateBelief
from src.belief.update import BeliefUpdater
from src.belief.projection import BeliefProjector
from src.planning.deterministic_planner import DeterministicPlanner
from src.planning.sample_belief_planner import SampleBeliefPlanner

class AIBenchmarker:
    def __init__(self, mode="full_top_k", noise_level=0.3, k_worlds=3):
        print(f"Initializing AI Benchmark. Mode: {mode} | Noise: {noise_level} | K: {k_worlds}")
        self.mode = mode
        self.noise = noise_level
        self.k = k_worlds
        self.env = MockBlocksworldEnv(num_blocks=5)
        self.updater = BeliefUpdater(alpha=1.0)
        self.projector = BeliefProjector("domains/blocksworld/constraints.yaml")
        
        # Disable info gain/sensing depending on mode
        sensing = ["reveal"] if self.mode == "full_top_k" else []
        self.planner = SampleBeliefPlanner(
            DeterministicPlanner("domains/blocksworld/domain.pddl"), 
            sensing_actions=sensing
        )
        self.metrics = []

    def _simulate_vlm(self, gt_predicates, noise_level):
        """Simulate a VLM returning noisy logits. Higher noise = worse accuracy."""
        raw_logits = {}
        # In reality, the VLM proposes thousands of truths. We simulate a subset mapping blocks.
        for b1 in self.env.blocks:
            preds_to_guess = [f"clear({b1})", f"on_table({b1})", f"holding(agent,{b1})"]
            for b2 in self.env.blocks:
                if b1 != b2: preds_to_guess.append(f"on({b1},{b2})")
                
            for p in preds_to_guess:
                is_true = p in gt_predicates
                # With probability `noise_level`, the VLM hallucinates the exact opposite.
                if random.random() < noise_level:
                    raw_logits[p] = 0.95 if not is_true else 0.05
                else:
                    raw_logits[p] = 0.95 if is_true else 0.05
        return raw_logits

    def run_sweep(self, max_episodes=5):
        for ep in range(max_episodes):
            print(f"\nEvaluating Episode: {ep} - Mode: {self.mode}")
            self.env.reset()
            belief_state = PredicateBelief()
            goal_str = "(on block_0 block_1)" # Static goal synchronized to internal Mock evaluator in valid PDDL format
            
            ep_trace = {
                "episode": ep, 
                "mode": self.mode,
                "noise_level": self.noise,
                "success": False, 
                "steps": 0, 
                "inconsistency_events": 0,
                "sensing_actions": 0,
                "replans": 0
            }
            
            for t in range(25):
                noisy_vision_states = self.env._get_gt_predicates()
                raw_logits = self._simulate_vlm(noisy_vision_states, self.noise)
                
                # Belief Structure Ablation
                if self.mode == "threshold":
                    # Discrete instantaneous states mapping (No continuous marginals)
                    for pred, prob in raw_logits.items():
                        belief_state.set_belief(pred, prob, t)
                else:
                    # Factorized Symbolics mapping
                    for pred, prob in raw_logits.items():
                        n = self.updater.update(belief_state.get_belief(pred), prob)
                        belief_state.set_belief(pred, n, timestep=t)

                # Metric: Measure raw VLM Inconsistency
                for b in self.env.blocks:
                    # Tracking physical contradictions (Targeting Table 2 / Figure 3)
                    if belief_state.get_belief(f"holding(agent,{b})") > 0.5 and belief_state.get_belief(f"on_table({b})") > 0.5:
                        ep_trace["inconsistency_events"] += 1

                # Projection Structure Ablation
                if self.mode in ["threshold", "belief_no_verifier"]:
                    # Raw states passing straight to Pyperplan
                    top_k = [{k: v > 0.5 for k, v in belief_state.probs.items()}]
                    ep_trace["replans"] += 1
                elif self.mode == "belief_plus_verifier":
                    # Single MAP
                    top_k = self.projector.project_top_k_map_states(belief_state.probs, k=1)
                else:
                    # Full Top-K Constraints
                    top_k = self.projector.project_top_k_map_states(belief_state.probs, k=self.k)
                
                # Deterministic Planning Execution
                pddl_objects = self.env.blocks + ["agent"]
                cmd, args = self.planner.select_action(belief_state.probs, top_k, goal_str, pddl_objects)
                if cmd == "reveal": ep_trace["sensing_actions"] += 1
                
                obs, reward, done = self.env.step(cmd, args)
                if done:
                    ep_trace["success"] = True
                    ep_trace["steps"] = t
                    break

            if not ep_trace["success"]: ep_trace["steps"] = 25 # Timeout penalty
            self.metrics.append(ep_trace)

        out_path = f"outputs/benchmarks/{self.mode}_noise_{self.noise}.jsonl"
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
    args = parser.parse_args()
    
    benchmarker = AIBenchmarker(mode=args.mode, noise_level=args.noise, k_worlds=args.k)
    benchmarker.run_sweep(max_episodes=args.episodes)
