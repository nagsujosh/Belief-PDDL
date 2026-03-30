import sys
import os
import csv
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.blocksworld_env import MockBlocksworldEnv
from src.belief.state import PredicateBelief
from src.belief.update import BeliefUpdater
from src.belief.projection import BeliefProjector
from src.planning.deterministic_planner import DeterministicPlanner
from src.planning.sample_belief_planner import SampleBeliefPlanner

class ReplanningEvalLoop:
    def __init__(self, use_verifier=True, noise_level=0.1, num_blocks=3):
        self.use_verifier = use_verifier
        self.noise_level = noise_level
        self.num_blocks = num_blocks
        self.env = MockBlocksworldEnv(num_blocks=num_blocks)
        self.updater = BeliefUpdater(alpha=1.0)
        self.projector = BeliefProjector("domains/blocksworld/constraints.yaml")
        self.planner = SampleBeliefPlanner(
            DeterministicPlanner("domains/blocksworld/domain.pddl"), 
            sensing_actions=["reveal_side"]
        )

    def _apply_noise(self, true_prob: float) -> float:
        """Injects artificial neural detection failure"""
        if random.random() < self.noise_level:
            return 1.0 - true_prob # Flip the probability drastically
        return true_prob

    def execute_episode(self, goal_str="(on block_0 block_1)", max_steps=20):
        obs = self.env.reset()
        belief_state = PredicateBelief()
        objects = [f"block_{i}" for i in range(self.num_blocks)]
        steps = 0
        sensed = 0
        
        # Dynamic Vocabulary Loop
        unary_preds = ["visible", "clear", "on_table", "holding"]
        binary_preds = ["on"]
        
        for t in range(max_steps):
            steps += 1
            
            # Dynamically push noise evaluations exactly matching `predicates.yaml` domains
            for obj in objects:
                # Global unary check
                for pred in unary_preds:
                    gt_label = f"{pred}({obj})" in obs.gt_predicates
                    obs_p = self._apply_noise(0.95 if gt_label else 0.05)
                    n = self.updater.update(belief_state.get_belief(f"{pred}({obj})"), obs_prob=obs_p)
                    belief_state.set_belief(f"{pred}({obj})", n, timestep=t)
                    
                # Pairwise binary checks
                for target in objects:
                    if target == obj: continue
                    for bipred in binary_preds:
                        gt_label = f"{bipred}({obj},{target})" in obs.gt_predicates
                        obs_p = self._apply_noise(0.95 if gt_label else 0.05)
                        n = self.updater.update(belief_state.get_belief(f"{bipred}({obj},{target})"), obs_prob=obs_p)
                        belief_state.set_belief(f"{bipred}({obj},{target})", n, timestep=t)
                        
            # Global rule
            gt_label = "arm_empty()" in obs.gt_predicates
            n = self.updater.update(belief_state.get_belief("arm_empty()"), obs_prob=self._apply_noise(0.99 if gt_label else 0.01))
            belief_state.set_belief("arm_empty()", n, timestep=t)
            
            # Projection Step: This is what we evaluate! Belief-PDDL vs Threshold Baseline
            if self.use_verifier:
                projected = self.projector.project_map_state(belief_state.probs)
            else:
                # Baseline 1: Naive Thresholding. If it's > 0.5, we think it's true.
                projected = {k: v > 0.5 for k, v in belief_state.probs.items()}

            # Planning
            cmd, args = self.planner.select_action(belief_state.probs, projected, goal_str, objects)
            
            # print(f"Timestep {t}: {cmd} {args}") # debugging
            
            if cmd == "noop": break
            if cmd in ["reveal_side"]: sensed += 1
            
            # Execute
            obs, reward, done = self.env.step(cmd, args)
            if done:
                return {"success": 1, "steps": steps, "sensing_actions": sensed}
                
        return {"success": 0, "steps": steps, "sensing_actions": sensed}

def run_large_scale_evals():
    os.makedirs("outputs/eval", exist_ok=True)
    out_file = "outputs/eval/results.csv"
    
    configs = [
        {"model": "Belief-PDDL-CPSAT", "use_verifier": True, "noise": 0.0, "blocks": 3},
        {"model": "Belief-PDDL-CPSAT", "use_verifier": True, "noise": 0.25, "blocks": 3},
        {"model": "Thresholded-Base", "use_verifier": False, "noise": 0.25, "blocks": 3},
        {"model": "Belief-PDDL-CPSAT", "use_verifier": True, "noise": 0.0, "blocks": 5},
        {"model": "Belief-PDDL-CPSAT", "use_verifier": True, "noise": 0.25, "blocks": 5},
        {"model": "Belief-PDDL-CPSAT", "use_verifier": True, "noise": 0.0, "blocks": 7},
    ]
    
    episodes_per_config = 5
    
    with open(out_file, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Blocks', 'NoiseLevel', 'SuccessRate', 'AvgSteps', 'AvgSensing']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print("🚀 Starting Automated Scaling Evaluation Benchmarks...")
        for conf in configs:
            m_name = conf["model"]
            n_lvl = conf["noise"]
            b_cnt = conf["blocks"]
            print(f"Testing {m_name} | {b_cnt} Blocks | Noise {n_lvl}...")
            
            evaluator = ReplanningEvalLoop(use_verifier=conf["use_verifier"], noise_level=n_lvl, num_blocks=b_cnt)
            
            successes = 0
            tot_steps = 0
            tot_sense = 0
            
            for i in range(episodes_per_config):
                random.seed(42 + i + int(n_lvl*100) + b_cnt) # Replicable noise
                res = evaluator.execute_episode()
                successes += res["success"]
                tot_steps += res["steps"]
                tot_sense += res["sensing_actions"]
                
            writer.writerow({
                'Model': m_name,
                'Blocks': b_cnt,
                'NoiseLevel': n_lvl,
                'SuccessRate': successes / episodes_per_config,
                'AvgSteps': tot_steps / episodes_per_config,
                'AvgSensing': tot_sense / episodes_per_config
            })
            print(f"  -> Success: {successes}/{episodes_per_config}")

    print(f"\n✅ Massive Evaluation Completed! Results saved to {out_file}")

if __name__ == "__main__":
    run_large_scale_evals()
