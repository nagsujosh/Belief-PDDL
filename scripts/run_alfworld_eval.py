import os
import sys
import csv
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.alfworld_env import ALFWorldEnvWrapper
from src.belief.state import PredicateBelief
from src.belief.update import BeliefUpdater
from src.belief.projection import BeliefProjector
from src.planning.deterministic_planner import DeterministicPlanner
from src.planning.sample_belief_planner import SampleBeliefPlanner
from src.perception.clip_vision import CLIPVisionBackbone
from src.perception.calibrate import TemperatureScalar

class ALFWorldCLIPEvaluator:
    def __init__(self, task_config="config.yaml"):
        print("Initializing ALFWorld Embodied Pipeline...")
        self.env = ALFWorldEnvWrapper(task_config=task_config)
        self.updater = BeliefUpdater(alpha=1.0)
        self.projector = BeliefProjector("domains/alfworld/constraints.yaml")
        
        # Core Neuro-Symbolic Reasoning Engine
        self.planner = SampleBeliefPlanner(
            DeterministicPlanner("domains/alfworld/domain.pddl"), 
            sensing_actions=["look"]
        )
        
        # Real VLM
        self.clip = CLIPVisionBackbone(model_name="openai/clip-vit-base-patch32")
        self.calibrator = TemperatureScalar(init_temp=1.2) # Soften the contrastive logits slightly
        
    def execute_episode(self, goal_str="(in apple_1 fridge_1)", max_steps=20):
        # Starts the Unity Embodied Client
        obs = self.env.reset()
        belief_state = PredicateBelief()
        
        # Vocabulary
        objects = ["apple_1", "fridge_1", "countertop_1"]
        predicates = ["visible", "closed", "open"] 
        
        # Tracking for Reproducibility Output
        self.metrics = []
        
        for t in range(max_steps):
            print(f"\n--- STEP {t} ---")
            step_record = {"step": t, "action": None, "beliefs": {}}
            
            # 1. Real Perception: Feed Unity RGB -> CLIP VLM -> Probabilities
            for obj in objects:
                for pred in predicates:
                    # e.g., "visible(apple_1)" translates to standard English for CLIP
                    prompt = f"an image showing that {obj.replace('_',' ')} is {pred}"
                    
                    # Compute Cosine Similarity between Prompt and current Agent vision
                    image_features = self.clip.extract_image_features(obs.rgb)
                    text_features = self.clip.extract_text_features([prompt])
                    
                    raw_logits = self.clip.compute_similarity(image_features, text_features)
                    
                    # Calibrate raw similarity scores to proper Probability (0.0 -> 1.0) via Forward pass
                    logit_tensor = torch.tensor([[raw_logits[0][0]]], dtype=torch.float32)
                    obs_p = self.calibrator(logit_tensor).item()
                    
                    # Log-Odds Bayesian continuous tracking
                    n = self.updater.update(belief_state.get_belief(f"{pred}({obj})"), obs_prob=obs_p)
                    
                    step_record["beliefs"][f"{pred}({obj})"] = {
                        "vlm_raw": float(raw_logits[0][0]),
                        "calibrated_prob": float(obs_p),
                        "log_odds": float(n)
                    }

            # 2. Logic Protection: Compilation of Constraints into feasible mathematical universes.
            top_k_projected = self.projector.project_top_k_map_states(belief_state.probs, k=3)
            step_record["cp_sat_projected"] = top_k_projected[0] if top_k_projected else {}

            # 3. Planning: Generate deterministic execution tree avoiding MAP conflicts
            cmd, args = self.planner.select_action(belief_state.probs, top_k_projected, goal_str, objects)
            
            print(f"Action Selected: {cmd} {args}")
            step_record["action"] = f"{cmd} {args}"
            self.metrics.append(step_record)
            
            # 4. Actuating the Unity Physics Engine
            obs, reward, done = self.env.step(cmd, args)
            
            if done:
                print(f"Goal Complete at Step {t}!")
                self._save_metrics("success")
                return True
                
        self._save_metrics("timeout")
        return False
        
    def _save_metrics(self, status: str):
        out_path = "outputs/eval/alfworld_metrics.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"status": status, "trace": self.metrics}, f, indent=4)
        print(f"Saved evaluation track trace to {out_path} for reviewer transparency.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config", type=str, default="config.yaml", help="Path to ALFworld parameters")
    args = parser.parse_args()
    
    evaluator = ALFWorldCLIPEvaluator(task_config=args.task_config)
    evaluator.execute_episode()
