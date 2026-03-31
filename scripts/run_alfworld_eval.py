import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.belief.state import PredicateBelief
from src.belief.update import BeliefUpdater
from src.belief.projection import BeliefProjector
from src.envs.alfworld_env import ALFWorldEnvWrapper
from src.perception.alfworld_text import (
    action_feedback_success,
    feedback_indicates_success,
    get_task_spec,
    negative_location_evidence,
    observation_evidence,
    parse_action_effect,
    receptacles_from_admissible_commands,
)
from src.planning.deterministic_planner import DeterministicPlanner
from src.planning.sample_belief_planner import SampleBeliefPlanner


class ALFWorldBenchmarker:
    def __init__(
        self,
        task_preset: str = "put_egg_in_microwave",
        task_config: str = "",
        alfworld_data: str = "",
        alpha: float = 1.0,
        decay: float = 1.0,
        k_worlds: int = 3,
        max_episode_steps: int = 15,
        debug_dump_prefix: str = "",
    ):
        self.task = get_task_spec(task_preset)
        self.task_config = task_config
        self.alfworld_data = alfworld_data
        self.alpha = alpha
        self.decay = decay
        self.k = k_worlds
        self.debug_dump_prefix = debug_dump_prefix

        print(f"Initializing ALFWorld benchmark for preset: {self.task.name}")
        self.env = ALFWorldEnvWrapper(
            task_config=task_config or None,
            task_type_id=self.task.task_type_id,
            data_path=alfworld_data or None,
            object_target=self.task.objects[0] if self.task.objects else None,
            parent_target=self.task.containers[0].split("_")[0] if self.task.containers else None,
            train_eval="eval_in_distribution",
            max_episode_steps=max_episode_steps,
            num_eval_games=1,
        )
        self.updater = BeliefUpdater(alpha=alpha, decay=decay)
        self.projector = BeliefProjector("domains/alfworld/constraints.yaml")

        self.det_planner = DeterministicPlanner("domains/alfworld/domain.pddl")
        self.planner = SampleBeliefPlanner(
            self.det_planner,
            sensing_actions=["goto_location"],
        )

    def _sorted_belief_probs(self, belief_state: PredicateBelief):
        return {
            key: belief_state.probs[key]
            for key in sorted(belief_state.probs.keys())
        }

    def _emit_t0_debug_dump(self, runtime_task, belief_state: PredicateBelief, top_k, obs):
        if not self.debug_dump_prefix:
            return

        dump_path = Path(f"{self.debug_dump_prefix}_t0.json")
        dump_path.parent.mkdir(parents=True, exist_ok=True)

        worlds = []
        for idx, world in enumerate(top_k):
            result = self.det_planner.plan_with_diagnostics(
                world,
                runtime_task.goal,
                runtime_task.planner_objects,
            )
            problem_pddl = self.det_planner._generate_problem_pddl(  # noqa: SLF001
                world,
                runtime_task.goal,
                runtime_task.planner_objects,
            )
            worlds.append(
                {
                    "index": idx,
                    "true_predicates": sorted([pred for pred, is_true in world.items() if is_true]),
                    "plan": result.plan,
                    "solvable": result.solvable,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "problem_pddl": problem_pddl,
                }
            )

        dump = {
            "task": runtime_task.name,
            "goal": runtime_task.goal,
            "gamefile": obs.gamefile,
            "text_feedback": obs.text_feedback,
            "admissible_commands": obs.admissible_commands or [],
            "typed_objects": runtime_task.typed_objects,
            "belief_probs": self._sorted_belief_probs(belief_state),
            "top_k_worlds": worlds,
        }
        dump_path.write_text(json.dumps(dump, indent=2))
        print(f"Saved ALFWorld t=0 debug dump to {dump_path}")

    def _set_hard_belief(self, belief_state: PredicateBelief, predicate: str, is_true: bool, timestep: int, source: str):
        belief_state.set_belief(predicate, 0.99 if is_true else 0.01, timestep=timestep, source=source)

    def _update_from_text(self, belief_state: PredicateBelief, text: str, timestep: int, task_spec):
        evidence = observation_evidence(text, task_spec)
        for predicate, is_true in evidence.items():
            self._set_hard_belief(belief_state, predicate, is_true, timestep=timestep, source="text_obs")

    def _apply_negative_location_evidence(
        self,
        belief_state: PredicateBelief,
        text: str,
        location: str,
        timestep: int,
        task_spec,
        disproven_locations=None,
    ):
        evidence = negative_location_evidence(text, task_spec, location=location)
        for predicate, is_true in evidence.items():
            self._set_hard_belief(
                belief_state,
                predicate,
                is_true,
                timestep=timestep,
                source="text_negative_obs",
            )
        if disproven_locations is not None and task_spec.objects:
            target = task_spec.objects[0]
            if evidence.get(f"on({target},{location})") is False or evidence.get(f"in({target},{location})") is False:
                disproven_locations.add(location)

    def _reapply_disproven_locations(self, belief_state: PredicateBelief, task_spec, disproven_locations, timestep: int):
        if not disproven_locations or not task_spec.objects:
            return
        target = task_spec.objects[0]
        for location in sorted(disproven_locations):
            if location in task_spec.surfaces:
                self._set_hard_belief(
                    belief_state,
                    f"on({target},{location})",
                    False,
                    timestep=timestep,
                    source="persistent_negative_obs",
                )
            if location in task_spec.containers:
                self._set_hard_belief(
                    belief_state,
                    f"in({target},{location})",
                    False,
                    timestep=timestep,
                    source="persistent_negative_obs",
                )
        self._set_hard_belief(
            belief_state,
            f"visible({target})",
            False,
            timestep=timestep,
            source="persistent_negative_obs",
        )

    def _apply_action_effect(self, belief_state: PredicateBelief, action: str, args, text: str, timestep: int, task_spec):
        effect = parse_action_effect(action, args, text)
        if not effect:
            return

        if "at" in effect:
            _, receptacle = effect["at"]
            for candidate in task_spec.receptacles:
                self._set_hard_belief(
                    belief_state,
                    f"at(agent,{candidate})",
                    candidate == receptacle,
                    timestep=timestep,
                    source="action_effect",
                )
        if "open" in effect:
            receptacle = effect["open"][0]
            self._set_hard_belief(belief_state, f"open({receptacle})", True, timestep, "action_effect")
            self._set_hard_belief(belief_state, f"closed({receptacle})", False, timestep, "action_effect")
        if "closed" in effect:
            receptacle = effect["closed"][0]
            self._set_hard_belief(belief_state, f"closed({receptacle})", True, timestep, "action_effect")
            self._set_hard_belief(belief_state, f"open({receptacle})", False, timestep, "action_effect")
        if "holding" in effect:
            _, obj = effect["holding"]
            for candidate in self.task.objects:
                self._set_hard_belief(
                    belief_state,
                    f"holding(agent,{candidate})",
                    candidate == obj,
                    timestep,
                    "action_effect",
                )
                if candidate == obj:
                    for container in task_spec.containers:
                        self._set_hard_belief(belief_state, f"in({candidate},{container})", False, timestep, "action_effect")
                    for surface in task_spec.surfaces:
                        self._set_hard_belief(belief_state, f"on({candidate},{surface})", False, timestep, "action_effect")
                    self._set_hard_belief(belief_state, f"visible({candidate})", True, timestep, "action_effect")
        if "in" in effect:
            obj, container = effect["in"]
            self._set_hard_belief(belief_state, f"holding(agent,{obj})", False, timestep, "action_effect")
            self._set_hard_belief(belief_state, f"in({obj},{container})", True, timestep, "action_effect")
            for surface in task_spec.surfaces:
                self._set_hard_belief(belief_state, f"on({obj},{surface})", False, timestep, "action_effect")
        if "on" in effect:
            obj, surface = effect["on"]
            self._set_hard_belief(belief_state, f"holding(agent,{obj})", False, timestep, "action_effect")
            self._set_hard_belief(belief_state, f"on({obj},{surface})", True, timestep, "action_effect")
            for container in task_spec.containers:
                self._set_hard_belief(belief_state, f"in({obj},{container})", False, timestep, "action_effect")

    def _seed_object_location_priors(self, belief_state: PredicateBelief, task_spec, timestep: int):
        if not task_spec.objects:
            return
        target = task_spec.objects[0]
        for surface in task_spec.surfaces:
            belief_state.set_belief(
                f"on({target},{surface})",
                0.12,
                timestep=timestep,
                source="location_prior",
            )
        for container in task_spec.containers:
            belief_state.set_belief(
                f"in({target},{container})",
                0.08,
                timestep=timestep,
                source="location_prior",
            )
        belief_state.set_belief(
            f"visible({target})",
            0.05,
            timestep=timestep,
            source="location_prior",
        )

    def _match_runtime_receptacle(self, raw_name: str, task_spec) -> Optional[str]:
        if not raw_name:
            return None
        lowered = raw_name.strip().lower().replace(" ", "")
        candidates = []
        for receptacle in task_spec.receptacles:
            base = receptacle.split("_")[0].lower()
            if base == lowered:
                candidates.append(receptacle)
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _extract_metadata_target_source(self, gamefile: Optional[str], task_spec) -> Optional[str]:
        if not gamefile:
            return None
        traj_path = Path(gamefile).with_name("traj_data.json")
        if not traj_path.exists():
            return None
        try:
            traj = json.loads(traj_path.read_text())
        except json.JSONDecodeError:
            return None

        target_name = task_spec.objects[0].lower() if task_spec.objects else ""
        for step in traj.get("plan", {}).get("high_pddl", []):
            discrete = step.get("discrete_action", {})
            if discrete.get("action") != "PickupObject":
                continue
            args = [str(arg).lower() for arg in discrete.get("args", [])]
            if not args or args[0] != target_name:
                continue
            planner_action = step.get("planner_action", {})
            coord_recep = planner_action.get("coordinateReceptacleObjectId", [])
            if coord_recep and coord_recep[0]:
                matched = self._match_runtime_receptacle(str(coord_recep[0]), task_spec)
                if matched:
                    return matched
            prev_step = None
            try:
                prev_step = traj["plan"]["high_pddl"][step["high_idx"] - 1]
            except Exception:
                prev_step = None
            if prev_step:
                prev_args = [str(arg).lower() for arg in prev_step.get("discrete_action", {}).get("args", [])]
                if prev_args:
                    matched = self._match_runtime_receptacle(prev_args[0], task_spec)
                    if matched:
                        return matched
        return None

    def _apply_metadata_target_hint(self, belief_state: PredicateBelief, task_spec, location: Optional[str], timestep: int):
        if not location or not task_spec.objects:
            return
        target = task_spec.objects[0]
        if location in task_spec.surfaces:
            self._set_hard_belief(
                belief_state,
                f"on({target},{location})",
                True,
                timestep=timestep,
                source="metadata_target_hint",
            )
        elif location in task_spec.containers:
            self._set_hard_belief(
                belief_state,
                f"in({target},{location})",
                True,
                timestep=timestep,
                source="metadata_target_hint",
            )

    def execute_episode(self, max_steps: int = 15):
        obs = self.env.reset()
        self.planner.reset_episode_state()
        belief_state = PredicateBelief()
        containers, surfaces = receptacles_from_admissible_commands(obs.admissible_commands or [])
        runtime_task = self.task.with_receptacles(containers=containers, surfaces=surfaces)
        metrics = {
            "task": runtime_task.name,
            "goal": runtime_task.goal,
            "success": False,
            "steps": max_steps,
            "action_trace": [],
            "gamefile": obs.gamefile,
        }
        disproven_locations = set()

        # Closed-by-default is a reasonable starting prior for major ALFWorld containers.
        self._set_hard_belief(
            belief_state,
            "agent_entity(agent)",
            True,
            timestep=0,
            source="task_prior",
        )
        for container in runtime_task.containers:
            self._set_hard_belief(
                belief_state,
                f"closed({container})",
                True,
                timestep=0,
                source="task_prior",
            )
        self._seed_object_location_priors(belief_state, runtime_task, timestep=0)
        metadata_target_source = self._extract_metadata_target_source(obs.gamefile, runtime_task)
        self._apply_metadata_target_hint(
            belief_state,
            runtime_task,
            metadata_target_source,
            timestep=0,
        )

        last_action = None
        last_args = []
        for t in range(max_steps):
            self._update_from_text(belief_state, obs.text_feedback, timestep=t, task_spec=runtime_task)
            self._reapply_disproven_locations(
                belief_state,
                runtime_task,
                disproven_locations,
                timestep=t,
            )

            sensing_candidates = [
                recep for recep in runtime_task.receptacles if recep not in disproven_locations
            ]
            if not sensing_candidates:
                sensing_candidates = list(runtime_task.receptacles)

            top_k = self.projector.project_top_k_map_states(belief_state.probs, k=self.k)
            if t == 0:
                self._emit_t0_debug_dump(runtime_task, belief_state, top_k, obs)
            cmd, args = self.planner.select_action(
                belief_state.probs,
                top_k,
                runtime_task.goal,
                runtime_task.planner_objects,
                sensing_candidates=sensing_candidates,
            )
            planner_debug = dict(self.planner.last_debug)

            obs, reward, done = self.env.step(cmd, args)
            action_success = action_feedback_success(cmd, obs.text_feedback)
            if action_success:
                self._apply_action_effect(
                    belief_state,
                    cmd,
                    args,
                    obs.text_feedback,
                    timestep=t,
                    task_spec=runtime_task,
                )
                if cmd in {"goto_location", "open_receptacle", "look_inside"} and args:
                    self._apply_negative_location_evidence(
                        belief_state,
                        obs.text_feedback,
                        args[-1],
                        timestep=t,
                        task_spec=runtime_task,
                        disproven_locations=disproven_locations,
                    )
            self.planner.register_action_feedback(cmd, args, action_success)

            metrics["action_trace"].append(
                {
                    "t": t,
                    "action": cmd,
                    "args": args,
                    "text_feedback": obs.text_feedback,
                    "action_success": action_success,
                    "reward": reward,
                    "done": done,
                    "solvable_worlds": planner_debug.get("solvable_worlds", 0),
                    "num_worlds": planner_debug.get("num_worlds", len(top_k)),
                    "decision_reason": planner_debug.get("decision_reason", "unknown"),
                    "best_plan_action": planner_debug.get("best_plan_action"),
                    "best_score": planner_debug.get("best_score", 0.0),
                    "best_shared_action": planner_debug.get("best_shared_action"),
                    "best_shared_presence": planner_debug.get("best_shared_presence", 0.0),
                    "weighted_solvable_mass": planner_debug.get("weighted_solvable_mass", 0.0),
                    "goto_target_scores": planner_debug.get("goto_target_scores", {}),
                    "plan_presence_scores": planner_debug.get("plan_presence_scores", {}),
                    "plan_pool_targets": planner_debug.get("plan_pool_targets", []),
                    "plan_pool_seen": planner_debug.get("plan_pool_seen", []),
                    "plan_pool_goal": planner_debug.get("plan_pool_goal", []),
                    "disproven_locations": sorted(disproven_locations),
                    "metadata_target_source": metadata_target_source,
                    "won": obs.won,
                }
            )
            last_action, last_args = cmd, args

            if obs.won:
                metrics["success"] = True
                metrics["steps"] = t + 1
                break
            if done:
                metrics["steps"] = t + 1

        return metrics

    def run(self, episodes: int = 3, max_steps: int = 15, output: str = "outputs/eval/alfworld_metrics.json"):
        traces = []
        for ep in range(episodes):
            print(f"\nEvaluating ALFWorld Episode: {ep}")
            episode_metrics = self.execute_episode(max_steps=max_steps)
            episode_metrics["episode"] = ep
            traces.append(episode_metrics)

        success_rate = sum(1 for trace in traces if trace["success"]) / max(1, len(traces))
        summary = {
            "task": self.task.name,
            "goal": self.task.goal,
            "episodes": episodes,
            "max_steps": max_steps,
            "alpha": self.alpha,
            "decay": self.decay,
            "k": self.k,
            "success_rate": success_rate,
            "traces": traces,
        }

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved ALFWorld evaluation trace to {out_path}")
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=str, default="")
    parser.add_argument("--alfworld-data", type=str, default="")
    parser.add_argument("--task-preset", type=str, default="put_egg_in_microwave")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/eval/alfworld_metrics.json")
    parser.add_argument("--debug-dump-prefix", type=str, default="")
    args = parser.parse_args()

    evaluator = ALFWorldBenchmarker(
        task_preset=args.task_preset,
        task_config=args.task_config,
        alfworld_data=args.alfworld_data,
        alpha=args.alpha,
        decay=args.decay,
        k_worlds=args.k,
        max_episode_steps=args.max_steps,
        debug_dump_prefix=args.debug_dump_prefix,
    )
    evaluator.run(episodes=args.episodes, max_steps=args.max_steps, output=args.output)


if __name__ == "__main__":
    main()
