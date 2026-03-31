import os
import sys
import subprocess
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class PlanResult:
    plan: Optional[List[str]]
    solvable: bool
    returncode: int
    stdout: str
    stderr: str

class DeterministicPlanner:
    def __init__(self, domain_path: str):
        self.domain_path = os.path.abspath(domain_path)
        self.temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'eval'))
        os.makedirs(self.temp_dir, exist_ok=True)
        self.typed_objects: Optional[Dict[str, List[str]]] = None
        
        # Dynamically extract domain name from the file (e.g. blocksworld, alfworld)
        self.domain_name = "blocksworld"
        if "alfworld" in self.domain_path.lower():
            self.domain_name = "alfworld"

    def set_typed_objects(self, typed_objects: Dict[str, List[str]]):
        self.typed_objects = typed_objects

    def _format_object_declarations(self, objects: Union[List[str], Dict[str, List[str]]]) -> str:
        if isinstance(objects, dict):
            lines = []
            for obj_type, names in objects.items():
                if names:
                    lines.append(f"{' '.join(names)} - {obj_type}")
            return "\n    ".join(lines)

        if self.typed_objects:
            return self._format_object_declarations(self.typed_objects)

        if self.domain_name == "alfworld":
            return " ".join(objects)
        return " ".join([o for o in objects if o != "agent"]) + " - block"

    def _generate_problem_pddl(self, state: Dict[str, bool], goal_str: str, objects: Union[List[str], Dict[str, List[str]]]) -> str:
        """
        Dynamically constructs a PDDL problem string from the boolean True states.
        """
        obj_decl = self._format_object_declarations(objects)
        
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

    def plan(self, state: Dict[str, bool], goal_str: str, objects: Union[List[str], Dict[str, List[str]]]) -> Optional[List[str]]:
        """
        Executes pyperplan to find a sequence of deterministic actions reaching the goal.
        Returns first action string or None if unplannable.
        """
        result = self.plan_with_diagnostics(state, goal_str, objects)
        return result.plan

    def plan_with_diagnostics(self, state: Dict[str, bool], goal_str: str, objects: Union[List[str], Dict[str, List[str]]]) -> PlanResult:
        """
        Executes pyperplan and returns the parsed plan plus basic diagnostics.
        """
        prob_id = uuid.uuid4().hex[:6]
        prob_path = os.path.join(self.temp_dir, f"prob_{prob_id}.pddl")
        
        with open(prob_path, 'w') as f:
            f.write(self._generate_problem_pddl(state, goal_str, objects))
            
        try:
            # Invoke pyperplan through the current interpreter so we use the active environment.
            cmd = [sys.executable, "-m", "pyperplan", "-H", "hff", "-s", "gbf", self.domain_path, prob_path]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Pyperplan writes solution to prob_path.soln
            sol_path = f"{prob_path}.soln"
            if os.path.exists(sol_path):
                with open(sol_path, 'r') as f:
                    plan = [line.strip()[1:-1] for line in f.readlines() if line.startswith('(')]
                os.remove(sol_path)
                os.remove(prob_path)
                return PlanResult(
                    plan=plan,
                    solvable=True,
                    returncode=res.returncode,
                    stdout=res.stdout,
                    stderr=res.stderr,
                )
            else:
                os.remove(prob_path)
                return PlanResult(
                    plan=None,
                    solvable=False,
                    returncode=res.returncode,
                    stdout=res.stdout,
                    stderr=res.stderr,
                )
        except Exception as e:
            if os.path.exists(prob_path):
                os.remove(prob_path)
            return PlanResult(
                plan=None,
                solvable=False,
                returncode=1,
                stdout="",
                stderr=str(e),
            )
