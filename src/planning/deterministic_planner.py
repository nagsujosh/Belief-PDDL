import os
import subprocess
import uuid
from typing import Dict, List, Optional

class DeterministicPlanner:
    def __init__(self, domain_path: str):
        self.domain_path = os.path.abspath(domain_path)
        self.temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'eval'))
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Dynamically extract domain name from the file (e.g. blocksworld, alfworld)
        self.domain_name = "blocksworld"
        if "alfworld" in self.domain_path.lower():
            self.domain_name = "alfworld"

    def _generate_problem_pddl(self, state: Dict[str, bool], goal_str: str, objects: List[str]) -> str:
        """
        Dynamically constructs a PDDL problem string from the boolean True states.
        """
        if self.domain_name == "alfworld":
            # Very basic string type mapping for prototype purposes, real system must parse types
            obj_decl = " ".join([o for o in objects if "agent" not in o]) + " - object"
        else:
            obj_decl = " ".join(objects) + " - block" 
        
        init_preds = []
        for pred, is_true in state.items():
            if is_true:
                if "(" in pred:
                    p_name = pred.split('(')[0].strip()
                    p_args = pred.split('(')[1].replace(')','').split(',')
                    p_args_str = " ".join([a.strip() for a in p_args if a.strip()])
                    if p_args_str:
                        init_preds.append(f"({p_name} {p_args_str})")
                    else:
                        init_preds.append(f"({p_name})")
                else:
                    init_preds.append(f"({pred})")
        init_str = "\n    ".join(init_preds)
        
        problem = f"""(define (problem auto_gen)
  (:domain {self.domain_name})
  (:objects {obj_decl})
  (:init 
    {init_str}
  )
  (:goal (and {goal_str}))
)
"""
        return problem

    def plan(self, state: Dict[str, bool], goal_str: str, objects: List[str]) -> Optional[List[str]]:
        """
        Executes pyperplan to find a sequence of deterministic actions reaching the goal.
        Returns first action string or None if unplannable.
        """
        prob_id = uuid.uuid4().hex[:6]
        prob_path = os.path.join(self.temp_dir, f"prob_{prob_id}.pddl")
        
        with open(prob_path, 'w') as f:
            f.write(self._generate_problem_pddl(state, goal_str, objects))
            
        try:
            # -H hff -s gbf are standard fast heuristics in pyperplan
            cmd = ["pyperplan", "-l", "error", "-H", "hff", "-s", "gbf", self.domain_path, prob_path]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Pyperplan writes solution to prob_path.soln
            sol_path = f"{prob_path}.soln"
            if os.path.exists(sol_path):
                with open(sol_path, 'r') as f:
                    plan = [line.strip()[1:-1] for line in f.readlines() if line.startswith('(')]
                os.remove(sol_path)
                os.remove(prob_path)
                return plan
            else:
                print(f"Pyperplan Failed! STDERR: {res.stderr}")
                os.remove(prob_path)
                return None
        except Exception as e:
            if os.path.exists(prob_path):
                os.remove(prob_path)
            print(f"Planning Error: {e}")
            return None
