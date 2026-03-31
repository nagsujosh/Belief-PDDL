import math
import yaml
import itertools
from ortools.sat.python import cp_model
from typing import Dict, List, Tuple

class BeliefProjector:
    def __init__(self, constraints_path: str):
        """
        Generic Constraint Compiler:
        Dynamically parses YAML mutex/implication rules, grounds them against
        the current observation vocabulary, and projects into Top-K feasible universes.
        """
        with open(constraints_path, 'r') as f:
            self.constraints = yaml.safe_load(f)

    def _get_score(self, prob: float) -> int:
        p = max(1e-4, min(1.0 - 1e-4, prob))
        log_odds = math.log(p / (1.0 - p))
        return int(log_odds * 1000)

    def _parse_pred(self, literal: str) -> Tuple[str, List[str]]:
        if "(" not in literal: return literal.strip(), []
        idx = literal.index("(")
        pred = literal[:idx].strip()
        args = [a.strip() for a in literal[idx+1:-1].split(",") if a.strip()]
        return pred, args

    def _get_generic_vars(self, templates: List[str]) -> List[str]:
        vars_set = set()
        for t in templates:
            _, args = self._parse_pred(t)
            for a in args:
                if a not in ["agent"]: # 'agent' is a static type, others are generic variables
                    vars_set.add(a)
        return list(vars_set)

    def _ground_templates(self, templates: List[str], objects_list: List[str]) -> List[List[str]]:
        """
        Ground a list of template literals against the current object vocabulary.
        Returns one grounded literal list per valid binding.
        """
        generic_vars = self._get_generic_vars(templates)
        grounded_groups = []

        for perm in itertools.product(objects_list, repeat=max(1, len(generic_vars))):
            bind_map = dict(zip(generic_vars, perm))

            if len(generic_vars) > 1 and len(set(perm)) != len(perm):
                continue

            grounded = []
            for template in templates:
                literal = template
                for g_var, obj_val in bind_map.items():
                    literal = literal.replace(f"{g_var}", obj_val)
                grounded.append(literal)
            grounded_groups.append(grounded)

        return grounded_groups

    def _ground_union_literals(self, templates: List[str], objects_list: List[str]) -> List[str]:
        """
        Ground a disjunctive template group into a single union of literals.
        Example: ["arm_empty()", "holding(x)"] becomes
        ["arm_empty()", "holding(block_0)", "holding(block_1)", ...].
        This is the right semantics for existential completeness constraints.
        """
        grounded = []
        seen = set()
        for template in templates:
            generic_vars = self._get_generic_vars([template])
            if not generic_vars:
                if template not in seen:
                    grounded.append(template)
                    seen.add(template)
                continue

            for perm in itertools.product(objects_list, repeat=max(1, len(generic_vars))):
                if len(generic_vars) > 1 and len(set(perm)) != len(perm):
                    continue
                literal = template
                for g_var, obj_val in zip(generic_vars, perm):
                    literal = literal.replace(f"{g_var}", obj_val)
                if literal not in seen:
                    grounded.append(literal)
                    seen.add(literal)
        return grounded

    def project_top_k_map_states(self, belief_probs: Dict[str, float], k: int = 3) -> List[Dict[str, bool]]:
        """
        Finds the sequentially Top-K Maximum A Posteriori (MAP) assignments 
        of predicates that respect all logical constraints natively.
        """
        model = cp_model.CpModel()
        vars_dict = {}
        objects = set()

        # 1. Decision Variables & Goal Weights
        for pred_str, p in belief_probs.items():
            var = model.NewBoolVar(pred_str)
            vars_dict[pred_str] = var
            _, args = self._parse_pred(pred_str)
            objects.update(args)
            
        objects_list = list(objects)

        objective_coeffs = [self._get_score(p) for str_k, p in belief_probs.items()]
        objective_vars = list(vars_dict.values())
        model.Maximize(sum(c * v for c, v in zip(objective_coeffs, objective_vars)))

        # 2. Dynamic Constraint Compiler (Generically binds YAML abstract variables to physical objects)
        
        # A. Mutex Compilation
        if "mutex" in self.constraints:
            for mutex_pair in self.constraints["mutex"]:
                generic_vars = self._get_generic_vars(mutex_pair)
                
                # Ground all permutations of objects against generic template rules
                for perm in itertools.product(objects_list, repeat=max(1, len(generic_vars))):
                    bind_map = dict(zip(generic_vars, perm))
                    
                    g_pred1 = mutex_pair[0]
                    g_pred2 = mutex_pair[1]
                    for g_var, obj_val in bind_map.items():
                        g_pred1 = g_pred1.replace(f"{g_var}", obj_val)
                        g_pred2 = g_pred2.replace(f"{g_var}", obj_val)
                    
                    # Prevent binding identical abstract vectors if explicitly split in logic patterns (e.g. l1 != l2)
                    if len(generic_vars) > 1 and len(set(perm)) != len(perm):
                        continue
                        
                    v1 = vars_dict.get(g_pred1)
                    v2 = vars_dict.get(g_pred2)
                    
                    if v1 is not None and v2 is not None:
                        # Mutex natively compiled: v1 -> not v2
                        model.AddImplication(v1, v2.Not())

        # B. Implications Compilation
        if "implications" in self.constraints:
            for imp in self.constraints["implications"]:
                generic_vars = self._get_generic_vars([imp["if"], imp["then"].replace('not ', '')])
                
                for perm in itertools.product(objects_list, repeat=max(1, len(generic_vars))):
                    bind_map = dict(zip(generic_vars, perm))
                    
                    cond = imp["if"]
                    conseq = imp["then"]
                    for g_var, obj_val in bind_map.items():
                        cond = cond.replace(f"{g_var}", obj_val)
                        conseq = conseq.replace(f"{g_var}", obj_val)
                    
                    is_not = conseq.startswith("not ")
                    if is_not: conseq = conseq[4:].strip()
                    
                    v_if = vars_dict.get(cond)
                    v_then = vars_dict.get(conseq)
                    
                    if v_if is not None and v_then is not None:
                        if is_not: model.AddImplication(v_if, v_then.Not())
                        else: model.AddImplication(v_if, v_then)

        # C. At-least-one groups. Useful for enforcing state completeness such as
        # "either the arm is empty or exactly one block is being held".
        if "at_least_one" in self.constraints:
            for group in self.constraints["at_least_one"]:
                templates = group if isinstance(group, list) else group.get("literals", [])
                grounded_group = self._ground_union_literals(templates, objects_list)
                grounded_vars = [vars_dict[g] for g in grounded_group if g in vars_dict]
                if grounded_vars:
                    model.AddBoolOr(grounded_vars)

        # 3. Successive Blocking for Top-K extraction
        top_k_worlds = []
        solver = cp_model.CpSolver()
        
        for i in range(k):
            status = solver.Solve(model)
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                world = {p: bool(solver.Value(v)) for p, v in vars_dict.items()}
                top_k_worlds.append(world)
                
                # Force CP-SAT to find the NEXT best mathematical assignment by explicitly
                # outlawing the exact combination of boolean variables just found.
                b_vars = [v if solver.Value(v) else v.Not() for v in vars_dict.values()]
                model.AddBoolOr([v.Not() for v in b_vars])
            else:
                break
                
        if not top_k_worlds:
            print("WARNING: Verifier failed. Returning fallback MAP.")
            return [{k: v > 0.5 for k, v in belief_probs.items()}]
            
        return top_k_worlds
