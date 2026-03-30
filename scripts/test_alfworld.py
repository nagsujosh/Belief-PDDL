import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.alfworld_env import ALFWorldEnvWrapper
from src.belief.state import PredicateBelief
from src.belief.update import BeliefUpdater
from src.belief.projection import BeliefProjector
from src.planning.deterministic_planner import DeterministicPlanner
from src.planning.sample_belief_planner import SampleBeliefPlanner
from src.execution.replan_loop import ReplanningLoop

def run_alfworld_test():
    print("\n🏠 --- PHASE 5: ALFWORLD SCALING DEMO ---")
    
    # Target Task: Find the hidden apple inside the fridge and hold it.
    env = ALFWorldEnvWrapper("test_task")
    updater = BeliefUpdater(alpha=1.0)
    
    # Using ALFWorld defined logic constraints
    projector = BeliefProjector("domains/alfworld/constraints.yaml")
    
    det_planner = DeterministicPlanner("domains/alfworld/domain.pddl")
    # "look_inside" serves as the Information Gathering sensing action
    sample_planner = SampleBeliefPlanner(det_planner, sensing_actions=["look_inside"])
    
    objects = ["apple_1", "fridge_1", "countertop_1"]
    
    # We want to hold the apple!
    goal = "holding agent apple_1"
    belief_state = PredicateBelief()
    
    print("Agent wakes up at countertop. Goal: 'holding agent apple_1'")
    print("The fridge is closed. The apple is hidden inside.")
    
    loop = ReplanningLoop(
        env=env,
        query_builder=None, 
        crop_builder=None,
        vision_backbone=None,
        unary_head=None,
        calibrator=None,
        belief_updater=updater,
        projector=projector,
        planner=sample_planner
    )
    
    # Monkeypatch the Replan Loop to understand the ALFWorld logic schema
    # (Just an override for this quick demo test run, normally implemented abstractly)
    def alfworld_execute(belief_state, goal_str, objects, max_steps=10):
        obs = env.reset()
        for t in range(max_steps):
            print(f"\n--- Timestep {t} ---")
            
            # Simple manual observation loop to mock neural inference
            for obj in objects:
                # Is it explicitly visible?
                if f"visible({obj})" in obs.gt_predicates:
                    new_prob = updater.update(belief_state.get_belief(f"visible({obj})"), obs_prob=0.99)
                    belief_state.set_belief(f"visible({obj})", new_prob, timestep=t)
                
                # Check receptacle status
                if f"closed({obj})" in obs.gt_predicates:
                    n = updater.update(belief_state.get_belief(f"closed({obj})"), obs_prob=0.99)
                    belief_state.set_belief(f"closed({obj})", n, timestep=t)
                elif f"open({obj})" in obs.gt_predicates:
                    n = updater.update(belief_state.get_belief(f"open({obj})"), obs_prob=0.99)
                    belief_state.set_belief(f"open({obj})", n, timestep=t)
                
            projected = projector.project_map_state(belief_state.probs)

            # Plan
            cmd, args = sample_planner.select_action(belief_state.probs, projected, goal_str, objects)
            
            if cmd == "noop": break
            print(f"Agent Action: {cmd} {args}")
                
            obs, r, done = env.step(cmd, args)
            print(f"ALFWorld returns: '{obs.text_feedback}'")
            if done: 
                print(f"Goal Complete! Reward: {r}")
                break
                
    alfworld_execute(belief_state, goal, objects)

if __name__ == "__main__":
    run_alfworld_test()
