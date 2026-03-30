import math
from typing import Dict, List, Tuple
from collections import defaultdict
from src.planning.deterministic_planner import DeterministicPlanner

class SampleBeliefPlanner:
    def __init__(self, deterministic_planner: DeterministicPlanner, sensing_actions: List[str]):
        """
        AI Planning Layer (Contribution C):
        Plans sequentially over Top-K feasible worlds, scoring first-actions 
        by weighted joint probabilities. Selects deterministic actions unless
        uncertainty limits breach, triggering True Expected Entropy Reduction (IG) sensing.
        """
        self.det_planner = deterministic_planner
        self.sensing_actions = sensing_actions

    def _entropy(self, p: float) -> float:
        p = max(1e-4, min(1.0 - 1e-4, p))
        return -p * math.log2(p) - (1-p) * math.log2(1-p)

    def _world_weight(self, world: Dict[str, bool], belief_probs: Dict[str, float]) -> float:
        """
        Computes the log joint-probability of the projected constraint universe.
        """
        weight = 0.0
        for k, is_true in world.items():
            p = belief_probs.get(k, 0.5)
            p = max(1e-4, min(1.0 - 1e-4, p))
            if is_true: weight += math.log(p)
            else: weight += math.log(1.0 - p)
        return weight

    def _calculate_ig(self, belief_probs: Dict[str, float], obj: str) -> float:
        """
        Calculates the explicit Expected Entropy Reduction (Information Gain) 
        gained by actively looking at a specific object.
        """
        visible_prob = belief_probs.get(f"visible({obj})", 0.0)
        # If we look at it, uncertainty collapses. Thus IG is proportional to current entropy 
        # regarding all predicates specifically bound to that object.
        obj_entropy = sum(self._entropy(v) for k, v in belief_probs.items() if obj in k)
        
        # High IG means looking at this object solves massive mathematical uncertainty
        return obj_entropy * (1.0 - visible_prob)

    def select_action(self, belief_probs: Dict[str, float], top_k_worlds: List[Dict[str, bool]],
                      goal_str: str, objects: List[str]) -> Tuple[str, List[str]]:
        
        # 1. Weight the generated universally feasible worlds
        weights = [self._world_weight(w, belief_probs) for w in top_k_worlds]
        # Normalize weights to probabilities via Softmax over log space
        max_w = max(weights) if weights else 0
        exp_w = [math.exp(w - max_w) for w in weights]
        sum_exp = sum(exp_w)
        norm_weights = [w / sum_exp for w in exp_w] if sum_exp > 0 else []

        action_scores = defaultdict(float)
        
        # 2. Top-K Joint Planning Loop
        for world, weight in zip(top_k_worlds, norm_weights):
            plan = self.det_planner.plan(world, goal_str, objects)
            if plan:
                first_action = plan[0]
                action_scores[first_action] += weight

        # 3. Action Aggregation
        best_plan_action = None
        best_score = 0.0
        if action_scores:
            best_plan_action = max(action_scores, key=action_scores.get)
            best_score = action_scores[best_plan_action]

        # 4. Uncertainty Thresholds & Active Sensing
        # If the highest voted deterministic action doesn't even hold 40% confidence across
        # the constraints models, we lack sufficient geometric knowledge. Engage Active Sensing!
        if best_score < 0.40 and self.sensing_actions:
            best_ig = -1
            best_sense_obj = objects[0]
            
            # Explicitly hunt for the Object hiding the highest entropy density
            for obj in objects:
                ig = self._calculate_ig(belief_probs, obj)
                if ig > best_ig:
                    best_ig = ig
                    best_sense_obj = obj
                    
            print(f"Fallback Active Sensing Trigged! Highest Entropy Target: {best_sense_obj}")
            return self.sensing_actions[0], [best_sense_obj]
            
        # 5. Execute Highly Voted Mathematical Task Progress
        if best_plan_action:
            parts = best_plan_action.split(" ")
            return parts[0], parts[1:]
            
        # 6. Complete Mathematical Deadlock
        print("CRITICAL: ALL Top-K Worlds are unplannable. Executing desperate visual entropy sweep.")
        if self.sensing_actions:
            import random
            return self.sensing_actions[0], [random.choice(objects)]
        return "noop", []
