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
        self._last_sensing_target = None
        self._blocked_actions = {}
        self._blocked_action_cooldown = 3
        self._stale_sensing_targets = {}
        self._stale_sensing_cooldown = 3
        self._plan_pool_targets = []
        self._plan_pool_seen = set()
        self._plan_pool_history = set()
        self._plan_pool_goal = ("", "")
        self.last_debug = {}

    def reset_episode_state(self):
        self._last_sensing_target = None
        self._blocked_actions.clear()
        self._stale_sensing_targets.clear()
        self._plan_pool_targets = []
        self._plan_pool_seen = set()
        self._plan_pool_history = set()
        self._plan_pool_goal = ("", "")
        self.last_debug = {}

    def register_action_feedback(self, action: str, args: List[str], changed_state: bool):
        key = " ".join([action] + list(args))
        if not key.strip():
            return
        if action in self.sensing_actions and args and self.last_debug.get("decision_reason") == "sensing_blocked_recovery":
            target = args[0]
            if changed_state:
                self._stale_sensing_targets.clear()
            else:
                self._stale_sensing_targets[target] = self._stale_sensing_cooldown
        if changed_state:
            self._blocked_actions.clear()
            return
        self._blocked_actions[key] = self._blocked_action_cooldown

    def _decay_blocked_actions(self):
        expired = []
        for key in list(self._blocked_actions.keys()):
            self._blocked_actions[key] -= 1
            if self._blocked_actions[key] <= 0:
                expired.append(key)
        for key in expired:
            del self._blocked_actions[key]

    def _decay_stale_sensing_targets(self):
        expired = []
        for key in list(self._stale_sensing_targets.keys()):
            self._stale_sensing_targets[key] -= 1
            if self._stale_sensing_targets[key] <= 0:
                expired.append(key)
        for key in expired:
            del self._stale_sensing_targets[key]

    def _pick_sensing_target(self, candidates: List[str], belief_probs: Dict[str, float],
                             prefer_information_gain: bool = True,
                             use_stale_filter: bool = False) -> Tuple[str, float]:
        filtered = list(candidates)
        if use_stale_filter:
            filtered = [obj for obj in filtered if obj not in self._stale_sensing_targets]
        if self._last_sensing_target is not None and len(filtered) > 1:
            non_last = [obj for obj in filtered if obj != self._last_sensing_target]
            if non_last:
                filtered = non_last
        if not filtered:
            filtered = list(candidates)
        if not filtered:
            return None, None

        if not prefer_information_gain:
            return filtered[0], None

        best_obj = None
        best_ig = -1.0
        for obj in filtered:
            ig = self._calculate_ig(belief_probs, obj)
            if ig > best_ig:
                best_ig = ig
                best_obj = obj
        return best_obj, best_ig

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

    def _belief_prob(self, belief_probs: Dict[str, float], predicate: str, default: float = 0.5) -> float:
        return max(1e-4, min(1.0 - 1e-4, belief_probs.get(predicate, default)))

    def _parse_goal_target(self, goal_str: str) -> Tuple[str, str]:
        goal = goal_str.strip()
        if not (goal.startswith("(") and goal.endswith(")")):
            return "", ""
        inner = goal[1:-1].strip()
        if not inner:
            return "", ""
        parts = inner.split()
        if len(parts) != 3:
            return "", ""
        _, obj, receptacle = parts
        return obj, receptacle

    def _should_defer_goal_receptacle_return(self, belief_probs: Dict[str, float], action_str: str, goal_str: str) -> bool:
        goal_obj, goal_receptacle = self._parse_goal_target(goal_str)
        if not goal_obj or not goal_receptacle:
            return False
        if self._belief_prob(belief_probs, f"holding(agent,{goal_obj})", default=0.0) > 0.50:
            return False
        if self._belief_prob(belief_probs, f"open({goal_receptacle})") < 0.80:
            return False

        parts = action_str.split(" ")
        if not parts:
            return False
        action = parts[0]
        args = parts[1:]
        if action == "goto_location" and args and args[-1] == goal_receptacle:
            return True
        if action == "open_receptacle" and args and args[-1] == goal_receptacle:
            return True
        return False

    def _is_goal_receptacle_action(self, action_str: str, goal_str: str) -> bool:
        goal_obj, goal_receptacle = self._parse_goal_target(goal_str)
        if not goal_obj or not goal_receptacle:
            return False
        parts = action_str.split(" ")
        if not parts:
            return False
        action = parts[0]
        args = parts[1:]
        if action == "goto_location" and args and args[-1] == goal_receptacle:
            return True
        if action == "open_receptacle" and args and args[-1] == goal_receptacle:
            return True
        return False

    def _action_support(self, belief_probs: Dict[str, float], action_str: str) -> float:
        parts = action_str.split(" ")
        if not parts:
            return 0.0
        action = parts[0]
        args = parts[1:]

        if action == "goto_location":
            if len(args) < 2:
                return 0.0
            agent, to = args[0], args[-1]
            return self._belief_prob(belief_probs, f"agent_entity({agent})", default=0.99)
        if action == "open_receptacle":
            if len(args) < 2:
                return 0.0
            agent, recep = args[0], args[-1]
            return (
                self._belief_prob(belief_probs, f"agent_entity({agent})", default=0.99)
                * self._belief_prob(belief_probs, f"at({agent},{recep})")
                * self._belief_prob(belief_probs, f"closed({recep})")
            )
        if action == "close_receptacle":
            if len(args) < 2:
                return 0.0
            agent, recep = args[0], args[-1]
            return (
                self._belief_prob(belief_probs, f"agent_entity({agent})", default=0.99)
                * self._belief_prob(belief_probs, f"at({agent},{recep})")
                * self._belief_prob(belief_probs, f"open({recep})")
            )
        if action == "take_from_surface":
            if len(args) < 3:
                return 0.0
            agent, obj, recep = args[0], args[1], args[2]
            return (
                self._belief_prob(belief_probs, f"agent_entity({agent})", default=0.99)
                * self._belief_prob(belief_probs, f"at({agent},{recep})")
                * self._belief_prob(belief_probs, f"on({obj},{recep})")
                * self._belief_prob(belief_probs, f"visible({obj})")
            )
        if action == "take_from_container":
            if len(args) < 3:
                return 0.0
            agent, obj, recep = args[0], args[1], args[2]
            return (
                self._belief_prob(belief_probs, f"agent_entity({agent})", default=0.99)
                * self._belief_prob(belief_probs, f"at({agent},{recep})")
                * self._belief_prob(belief_probs, f"in({obj},{recep})")
                * self._belief_prob(belief_probs, f"open({recep})")
                * self._belief_prob(belief_probs, f"visible({obj})")
            )
        if action == "put_in_container":
            if len(args) < 3:
                return 0.0
            agent, obj, recep = args[0], args[1], args[2]
            return (
                self._belief_prob(belief_probs, f"agent_entity({agent})", default=0.99)
                * self._belief_prob(belief_probs, f"at({agent},{recep})")
                * self._belief_prob(belief_probs, f"holding({agent},{obj})")
                * self._belief_prob(belief_probs, f"open({recep})")
            )
        if action == "put_on_surface":
            if len(args) < 3:
                return 0.0
            agent, obj, recep = args[0], args[1], args[2]
            return (
                self._belief_prob(belief_probs, f"agent_entity({agent})", default=0.99)
                * self._belief_prob(belief_probs, f"at({agent},{recep})")
                * self._belief_prob(belief_probs, f"holding({agent},{obj})")
            )
        if action == "look_inside":
            # In ALFWorld text mode, opening a receptacle already reveals its visible
            # contents, so explicit look-inside actions are usually redundant.
            return 0.0
        return 0.0

    def _clear_plan_pool(self):
        self._plan_pool_targets = []
        self._plan_pool_seen = set()
        self._plan_pool_goal = ("", "")

    def _next_plan_pool_target(self, belief_probs: Dict[str, float], sensing_candidates: List[str]) -> str:
        for target in self._plan_pool_targets:
            if target in self._plan_pool_seen:
                continue
            if sensing_candidates and target not in sensing_candidates:
                continue
            if target in self._stale_sensing_targets:
                continue
            # Once we have entered an explicit plan-pool search mode, preserve the
            # unresolved candidate queue across steps even if agent-location beliefs
            # are noisy or stale. The queue itself already tracks which targets were
            # actually visited via `_plan_pool_seen`.
            return target
        return None

    def _plan_pool_candidates_from_scores(
        self,
        goto_target_scores: Dict[str, float],
        goto_target_earliness: Dict[str, float],
        belief_probs: Dict[str, float],
        sensing_candidates: List[str],
        goal_receptacle: str,
    ) -> List[str]:
        candidates = []
        deferred_candidates = []
        for target in goto_target_scores:
            if target == goal_receptacle:
                continue
            if sensing_candidates and target not in sensing_candidates:
                continue
            if belief_probs.get(f"at(agent,{target})", 0.0) >= 0.80:
                continue
            if target in self._stale_sensing_targets:
                continue
            if target in self._plan_pool_history:
                deferred_candidates.append(target)
            else:
                candidates.append(target)

        def _sort_targets(items: List[str]) -> List[str]:
            return sorted(
                items,
                key=lambda target: (
                    goto_target_scores[target],
                    goto_target_earliness[target],
                    target,
                ),
                reverse=True,
            )

        primary = _sort_targets(candidates)
        if primary:
            return primary
        return _sort_targets(deferred_candidates)

    def select_action(self, belief_probs: Dict[str, float], top_k_worlds: List[Dict[str, bool]],
                      goal_str: str, objects: List[str], sensing_candidates: List[str] = None) -> Tuple[str, List[str]]:
        self._decay_blocked_actions()
        self._decay_stale_sensing_targets()
        decision_threshold = 0.40
        sensing_candidates = list(sensing_candidates) if sensing_candidates is not None else list(objects)
        goal_obj, goal_receptacle = self._parse_goal_target(goal_str)
        holding_goal = bool(goal_obj) and self._belief_prob(
            belief_probs,
            f"holding(agent,{goal_obj})",
            default=0.0,
        ) > 0.50

        if self._plan_pool_goal and self._plan_pool_goal != (goal_obj, goal_receptacle):
            self._plan_pool_history = set()
            self._clear_plan_pool()
        if holding_goal:
            self._plan_pool_history = set()
            self._clear_plan_pool()

        # 1. Weight the generated universally feasible worlds
        weights = [self._world_weight(w, belief_probs) for w in top_k_worlds]
        # Normalize weights to probabilities via Softmax over log space
        max_w = max(weights) if weights else 0
        exp_w = [math.exp(w - max_w) for w in weights]
        sum_exp = sum(exp_w)
        norm_weights = [w / sum_exp for w in exp_w] if sum_exp > 0 else []

        action_scores = defaultdict(float)
        blocked_action_scores = defaultdict(float)
        plan_presence_scores = defaultdict(float)
        plan_earliness_scores = defaultdict(float)
        goto_target_scores = defaultdict(float)
        goto_target_earliness = defaultdict(float)
        plan_support_scores = {}
        solvable_worlds = 0
        unsolvable_worlds = 0
        first_action_votes = defaultdict(int)
        blocked_first_actions = defaultdict(int)
        weighted_solvable_mass = 0.0
        weighted_blocked_mass = 0.0
        
        # 2. Top-K Joint Planning Loop
        for world, weight in zip(top_k_worlds, norm_weights):
            result = self.det_planner.plan_with_diagnostics(world, goal_str, objects)
            plan = result.plan
            if plan:
                solvable_worlds += 1
                weighted_solvable_mass += weight
                for idx, action in enumerate(plan):
                    if action in self._blocked_actions:
                        continue
                    if self._should_defer_goal_receptacle_return(belief_probs, action, goal_str):
                        continue
                    plan_presence_scores[action] += weight
                    plan_earliness_scores[action] += weight / float(idx + 1)
                    parts = action.split(" ")
                    if parts and parts[0] == "goto_location" and len(parts) >= 3:
                        target = parts[-1]
                        goto_target_scores[target] += weight
                        goto_target_earliness[target] += weight / float(idx + 1)
                first_action = plan[0]
                if self._should_defer_goal_receptacle_return(belief_probs, first_action, goal_str):
                    continue
                if first_action in self._blocked_actions:
                    blocked_first_actions[first_action] += 1
                    blocked_action_scores[first_action] += weight
                    weighted_blocked_mass += weight
                    continue
                action_scores[first_action] += weight
                first_action_votes[first_action] += 1
            else:
                unsolvable_worlds += 1

        # 3. Action Aggregation
        best_plan_action = None
        best_score = 0.0
        if action_scores:
            best_plan_action = max(action_scores, key=action_scores.get)
            best_score = action_scores[best_plan_action]

        best_shared_action = None
        best_shared_presence = 0.0
        best_shared_earliness = 0.0
        if plan_presence_scores:
            for action in plan_presence_scores:
                plan_support_scores[action] = self._action_support(belief_probs, action)
            best_shared_action = max(
                plan_presence_scores,
                key=lambda action: (
                    plan_presence_scores[action],
                    plan_support_scores.get(action, 0.0),
                    plan_earliness_scores[action],
                ),
            )
            best_shared_presence = plan_presence_scores[best_shared_action]
            best_shared_earliness = plan_earliness_scores[best_shared_action]

        best_pick_action = None
        best_pick_presence = 0.0
        if plan_presence_scores:
            pickup_actions = [
                action
                for action in plan_presence_scores
                if action.startswith("take_from_surface ") or action.startswith("take_from_container ")
            ]
            if pickup_actions:
                best_pick_action = max(
                    pickup_actions,
                    key=lambda action: (
                        plan_presence_scores[action],
                        plan_support_scores.get(action, 0.0),
                        plan_earliness_scores[action],
                    ),
                )
                best_pick_presence = plan_presence_scores[best_pick_action]

        non_goal_goto_candidates = self._plan_pool_candidates_from_scores(
            goto_target_scores,
            goto_target_earliness,
            belief_probs,
            sensing_candidates,
            goal_receptacle,
        )

        self.last_debug = {
            "num_worlds": len(top_k_worlds),
            "solvable_worlds": solvable_worlds,
            "unsolvable_worlds": unsolvable_worlds,
            "weighted_solvable_mass": weighted_solvable_mass,
            "weighted_blocked_mass": weighted_blocked_mass,
            "first_action_votes": dict(first_action_votes),
            "blocked_first_actions": dict(blocked_first_actions),
            "blocked_actions": dict(self._blocked_actions),
            "action_scores": dict(action_scores),
            "blocked_action_scores": dict(blocked_action_scores),
            "plan_presence_scores": dict(plan_presence_scores),
            "plan_earliness_scores": dict(plan_earliness_scores),
            "goto_target_scores": dict(goto_target_scores),
            "goto_target_earliness": dict(goto_target_earliness),
            "plan_support_scores": dict(plan_support_scores),
            "best_plan_action": best_plan_action,
            "best_score": best_score,
            "best_shared_action": best_shared_action,
            "best_shared_presence": best_shared_presence,
            "best_shared_earliness": best_shared_earliness,
            "best_shared_support": plan_support_scores.get(best_shared_action, 0.0) if best_shared_action else 0.0,
            "best_pick_action": best_pick_action,
            "best_pick_presence": best_pick_presence,
            "best_pick_support": plan_support_scores.get(best_pick_action, 0.0) if best_pick_action else 0.0,
            "non_goal_goto_candidates": list(non_goal_goto_candidates),
            "decision_threshold": decision_threshold,
            "decision_reason": "unknown",
            "selected_action": None,
            "selected_args": [],
            "stale_sensing_targets": dict(self._stale_sensing_targets),
            "plan_pool_targets": list(self._plan_pool_targets),
            "plan_pool_seen": sorted(self._plan_pool_seen),
            "plan_pool_history": sorted(self._plan_pool_history),
            "plan_pool_goal": list(self._plan_pool_goal),
        }

        if self._plan_pool_targets and not holding_goal:
            next_target = self._next_plan_pool_target(belief_probs, sensing_candidates)
            if next_target is not None:
                self._plan_pool_seen.add(next_target)
                self._plan_pool_history.add(next_target)
                self._last_sensing_target = next_target
                self.last_debug.update({
                    "decision_reason": "sensing_plan_pool_continue",
                    "selected_action": self.sensing_actions[0] if self.sensing_actions else "noop",
                    "selected_args": [next_target],
                    "selected_sensing_target": next_target,
                    "selected_sensing_ig": None,
                    "plan_pool_targets": list(self._plan_pool_targets),
                    "plan_pool_seen": sorted(self._plan_pool_seen),
                    "plan_pool_history": sorted(self._plan_pool_history),
                    "plan_pool_goal": list(self._plan_pool_goal),
                })
                if self.sensing_actions:
                    return self.sensing_actions[0], [next_target]
            else:
                self._clear_plan_pool()

        shared_goal_return = (
            best_shared_action is not None
            and self._is_goal_receptacle_action(best_shared_action, goal_str)
        )
        if shared_goal_return and self.sensing_actions and not holding_goal:
            if non_goal_goto_candidates:
                if self._plan_pool_goal != (goal_obj, goal_receptacle):
                    self._clear_plan_pool()
                existing = list(self._plan_pool_targets)
                seen = set(existing)
                for target in non_goal_goto_candidates:
                    if target not in seen:
                        existing.append(target)
                        seen.add(target)
                self._plan_pool_targets = existing
                self._plan_pool_goal = (goal_obj, goal_receptacle)
                next_target = self._next_plan_pool_target(belief_probs, sensing_candidates)
                if next_target is not None:
                    self._plan_pool_seen.add(next_target)
                    self._plan_pool_history.add(next_target)
                    self._last_sensing_target = next_target
                    self.last_debug.update({
                        "decision_reason": "sensing_plan_pool_defer_goal_return",
                        "selected_action": self.sensing_actions[0],
                        "selected_args": [next_target],
                        "selected_sensing_target": next_target,
                        "selected_sensing_ig": None,
                        "plan_pool_targets": list(self._plan_pool_targets),
                        "plan_pool_seen": sorted(self._plan_pool_seen),
                        "plan_pool_history": sorted(self._plan_pool_history),
                        "plan_pool_goal": list(self._plan_pool_goal),
                    })
                    return self.sensing_actions[0], [next_target]

        if (
            best_pick_action
            and not holding_goal
            and weighted_solvable_mass >= 0.50
            and best_pick_presence >= 0.50
            and plan_support_scores.get(best_pick_action, 0.0) >= 0.50
        ):
            parts = best_pick_action.split(" ")
            self._last_sensing_target = None
            self.last_debug.update({
                "decision_reason": "execute_pick_subgoal",
                "selected_action": parts[0],
                "selected_args": parts[1:],
            })
            return parts[0], parts[1:]

        # In environments like ALFWorld, multiple projected worlds can all be solvable
        # while disagreeing on the first exact search branch. When that happens, prefer
        # an action that appears early across many solvable plans rather than dropping
        # straight into sensing.
        if (
            best_shared_action
            and weighted_solvable_mass >= 0.80
            and best_shared_presence >= 0.80
            and plan_support_scores.get(best_shared_action, 0.0) >= 0.50
        ):
            parts = best_shared_action.split(" ")
            self._last_sensing_target = None
            self.last_debug.update({
                "decision_reason": "execute_shared_subgoal",
                "selected_action": parts[0],
                "selected_args": parts[1:],
            })
            return parts[0], parts[1:]

        # 4. Uncertainty Thresholds & Active Sensing
        # If the highest voted deterministic action doesn't even hold 40% confidence across
        # the constraints models, we lack sufficient geometric knowledge. Engage Active Sensing!
        if best_score < decision_threshold and self.sensing_actions:
            if weighted_solvable_mass >= 0.80 and goto_target_scores:
                candidate_targets = list(non_goal_goto_candidates)

                if self._last_sensing_target is not None and len(candidate_targets) > 1:
                    non_last = [target for target in candidate_targets if target != self._last_sensing_target]
                    if non_last:
                        candidate_targets = non_last

                if candidate_targets:
                    self._plan_pool_targets = list(candidate_targets)
                    self._plan_pool_seen = set()
                    self._plan_pool_goal = (goal_obj, goal_receptacle)
                    best_target = max(
                        candidate_targets,
                        key=lambda target: (
                            goto_target_scores[target],
                            goto_target_earliness[target],
                            target,
                        ),
                    )
                    self._plan_pool_seen.add(best_target)
                    self._plan_pool_history.add(best_target)
                    self._last_sensing_target = best_target
                    self.last_debug.update({
                        "decision_reason": "sensing_plan_pool",
                        "selected_action": self.sensing_actions[0],
                        "selected_args": [best_target],
                        "selected_sensing_target": best_target,
                        "selected_sensing_ig": None,
                        "plan_pool_targets": list(self._plan_pool_targets),
                        "plan_pool_seen": sorted(self._plan_pool_seen),
                        "plan_pool_history": sorted(self._plan_pool_history),
                        "plan_pool_goal": list(self._plan_pool_goal),
                    })
                    return self.sensing_actions[0], [best_target]

            low_visibility = [
                obj for obj in sensing_candidates
                if belief_probs.get(f"visible({obj})", 0.0) < 0.95
            ]
            best_sense_obj, best_ig = self._pick_sensing_target(low_visibility, belief_probs)

            if best_sense_obj is not None:
                self._last_sensing_target = best_sense_obj
                self.last_debug.update({
                    "decision_reason": "sensing_low_confidence",
                    "selected_action": self.sensing_actions[0],
                    "selected_args": [best_sense_obj],
                    "selected_sensing_target": best_sense_obj,
                    "selected_sensing_ig": best_ig,
                })
                print(
                    f"Fallback Active Sensing Trigged! "
                    f"solvable_worlds={solvable_worlds}/{len(top_k_worlds)} "
                    f"target={best_sense_obj}"
                )
                return self.sensing_actions[0], [best_sense_obj]
            
        # 5. Execute Highly Voted Mathematical Task Progress
        if best_plan_action:
            self._last_sensing_target = None
            parts = best_plan_action.split(" ")
            self.last_debug.update({
                "decision_reason": "execute_plan_action",
                "selected_action": parts[0],
                "selected_args": parts[1:],
            })
            return parts[0], parts[1:]

        # 5b. Recovery retry: if all currently solvable mass is tied to blocked actions,
        # allow the dominant blocked action back in once its cooldown is almost over.
        if blocked_action_scores:
            best_blocked_action = max(blocked_action_scores, key=blocked_action_scores.get)
            cooldown_remaining = self._blocked_actions.get(best_blocked_action, 0)
            if cooldown_remaining <= 1:
                self._last_sensing_target = None
                parts = best_blocked_action.split(" ")
                self.last_debug.update({
                    "decision_reason": "execute_recovery_retry",
                    "selected_action": parts[0],
                    "selected_args": parts[1:],
                    "best_plan_action": best_blocked_action,
                    "best_score": blocked_action_scores[best_blocked_action],
                    "recovery_retry_action": best_blocked_action,
                    "recovery_retry_cooldown_remaining": cooldown_remaining,
                })
                return parts[0], parts[1:]

            # If the blocked action is still cooling down, use sensing to inspect the
            # exact local objects that the blocked action depends on instead of falling
            # back to a generic deadlock sweep.
            if self.sensing_actions:
                blocked_parts = best_blocked_action.split(" ")
                blocked_args = blocked_parts[1:]
                visible_blocked_args = [
                    obj for obj in blocked_args
                    if obj in sensing_candidates and belief_probs.get(f"visible({obj})", 0.0) < 0.95
                ]
                candidates = visible_blocked_args or [obj for obj in blocked_args if obj in sensing_candidates]
                if not candidates:
                    candidates = list(sensing_candidates)
                best_sense_obj, best_ig = self._pick_sensing_target(
                    candidates,
                    belief_probs,
                    use_stale_filter=True,
                )
                if best_sense_obj is not None:
                    self._last_sensing_target = best_sense_obj
                    self.last_debug.update({
                        "decision_reason": "sensing_blocked_recovery",
                        "selected_action": self.sensing_actions[0],
                        "selected_args": [best_sense_obj],
                        "selected_sensing_target": best_sense_obj,
                        "selected_sensing_ig": best_ig if best_ig >= 0 else None,
                        "recovery_retry_action": best_blocked_action,
                        "recovery_retry_cooldown_remaining": cooldown_remaining,
                    })
                    print(
                        f"Blocked-action recovery sensing! "
                        f"blocked={best_blocked_action} cooldown={cooldown_remaining} "
                        f"target={best_sense_obj}"
                    )
                    return self.sensing_actions[0], [best_sense_obj]
            
        # 6. Complete Mathematical Deadlock
        print("CRITICAL: ALL Top-K Worlds are unplannable. Executing desperate visual entropy sweep.")
        if self.sensing_actions:
            import random
            candidates = [obj for obj in sensing_candidates if belief_probs.get(f"visible({obj})", 0.0) < 0.95]
            if not candidates:
                candidates = list(sensing_candidates)
            filtered = [obj for obj in candidates if obj not in self._stale_sensing_targets]
            if self._last_sensing_target is not None and len(filtered) > 1:
                non_last = [obj for obj in filtered if obj != self._last_sensing_target]
                if non_last:
                    filtered = non_last
            if not filtered:
                filtered = candidates
            target = random.choice(filtered)
            self._last_sensing_target = target
            self.last_debug.update({
                "decision_reason": "sensing_deadlock",
                "selected_action": self.sensing_actions[0],
                "selected_args": [target],
                "selected_sensing_target": target,
                "selected_sensing_ig": None,
            })
            return self.sensing_actions[0], [target]
        self.last_debug.update({
            "decision_reason": "noop_deadlock",
            "selected_action": "noop",
            "selected_args": [],
        })
        return "noop", []
