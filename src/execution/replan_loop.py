from typing import List, Dict, Any

class ReplanningLoop:
    def __init__(self, env, query_builder, crop_builder, vision_backbone, unary_head, 
                 calibrator, belief_updater, projector, planner):
        """
        Coordinates the Sense -> Update -> Project -> Plan -> Act loop.
        """
        self.env = env
        self.query_builder = query_builder
        self.crop_builder = crop_builder
        self.vision = vision_backbone
        self.unary = unary_head
        self.calibrator = calibrator
        self.updater = belief_updater
        self.projector = projector
        self.planner = planner

    def execute(self, belief_state: Any, goal_str: str, objects: List[str], max_steps: int = 10):
        """
        Runs the full Belief-PDDL execution loop until goal is reached or max_steps exceeded.
        """
        obs = self.env.reset()
        
        for t in range(max_steps):
            print(f"\n--- Timestep {t} ---")
            
            # 1. Observation Model: Get logits for visible proxies
            # (In a fully implemented loop, the crop_builder and models slice up
            # the visual outputs and assign true stochastic logs. 
            # For this loop, we represent it abstractly by feeding true masks + noise)
            
            # Since this is a test loop tracking true environment states, we mock 
            # neural outputs from `obs.gt_predicates` and `obs.visible_objects`.
            # We assume objects in GT are highly confident.
            
            for obj in objects:
                # Mock neural network processing for the sake of the execution loop demo
                is_visible = f"visible({obj})" in obs.gt_predicates
                
                obs_prob = 0.9 if is_visible else 0.1
                new_prob = self.updater.update(belief_state.get_belief(f"visible({obj})"), obs_prob=obs_prob)
                belief_state.set_belief(f"visible({obj})", new_prob, timestep=t)

                for pred in ["clear", "on_table"]:
                    # Is it actually in the GT list? (Simulate perfection for seen objects)
                    gt_label = f"{pred}({obj})" in obs.gt_predicates
                    if is_visible:
                        obs_p = 0.95 if gt_label else 0.05
                    else:
                        # Unseen -> No visual update, or decay towards prior
                        obs_p = None
                    
                    if obs_p is not None:
                        n_prob = self.updater.update(belief_state.get_belief(f"{pred}({obj})"), obs_prob=obs_p)
                        belief_state.set_belief(f"{pred}({obj})", n_prob, timestep=t)

            # 2. Add Hand Tracking
            holding_agent = [obj for obj in objects if f"holding({obj})" in obs.gt_predicates]
            for obj in objects:
                gt_h = (obj in holding_agent)
                obs_p = 0.99 if gt_h else 0.01
                n_prob = self.updater.update(belief_state.get_belief(f"holding({obj})"), obs_prob=obs_p)
                belief_state.set_belief(f"holding({obj})", n_prob, timestep=t)

            empty = len(holding_agent) == 0
            n_prob = self.updater.update(belief_state.get_belief("arm_empty()"), obs_prob=0.99 if empty else 0.01)
            belief_state.set_belief("arm_empty()", n_prob, timestep=t)
            
            # Mock `on(x,y)` to keep it brief
            for x in objects:
                for y in objects:
                    if x == y: continue
                    pred_str = f"on({x},{y})"
                    is_visible = f"visible({x})" in obs.gt_predicates
                    gt_label = pred_str in obs.gt_predicates
                    if is_visible:
                        obs_p = 0.95 if gt_label else 0.05
                        n_prob = self.updater.update(belief_state.get_belief(pred_str), obs_prob=obs_p)
                        belief_state.set_belief(pred_str, n_prob, timestep=t)

            # 3. Project Constraints via CP-SAT
            projected = self.projector.project_map_state(belief_state.probs)

            # 4. Plan Step
            cmd, args = self.planner.select_action(belief_state.probs, projected, goal_str, objects)
            
            if cmd == "noop":
                print("Planner reported unplannable state. Terminating.")
                break
                
            print(f"Executing: {cmd} {args}")
                
            # 5. Execute in Environment
            obs, reward, done = self.env.step(cmd, args)
            
            if done:
                print(f"Goal Reached at t={t+1}!")
                break
