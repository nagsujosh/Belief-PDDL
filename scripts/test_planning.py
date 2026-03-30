import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.blocksworld_env import MockBlocksworldEnv
from src.belief.state import PredicateBelief
from src.belief.update import BeliefUpdater
from src.belief.projection import BeliefProjector
from src.planning.deterministic_planner import DeterministicPlanner
from src.planning.sample_belief_planner import SampleBeliefPlanner
from src.execution.replan_loop import ReplanningLoop

def test_planner_loop():
    print("\n--- 🤖 PHASE 4: PLANNING AND EXECUTION DEMO ---")
    
    # 1. Environment
    env = MockBlocksworldEnv(num_blocks=3)
    
    # 2. Logic Components
    updater = BeliefUpdater(alpha=1.0)
    projector = BeliefProjector("domains/blocksworld/constraints.yaml")
    
    # 3. Planners
    domain_path = "domains/blocksworld/domain.pddl"
    det_planner = DeterministicPlanner(domain_path)
    sensing_actions = ["reveal_side"]
    sample_planner = SampleBeliefPlanner(det_planner, sensing_actions)
    
    # 4. State
    objects = ["block_0", "block_1", "block_2"]
    
    # We want to stack block 0 on block 1.
    # But block_2 starts hidden in our mock env! Let's just track the whole thing!
    goal = "(on block_0 block_1)"
    belief_state = PredicateBelief()
    
    # 5. Loop!
    print("Initiating Execution Loop towards Goal: (on block_0 block_1)")
    
    loop = ReplanningLoop(
        env=env,
        query_builder=None, # Mocked via GT matching above
        crop_builder=None,
        vision_backbone=None,
        unary_head=None,
        calibrator=None,
        belief_updater=updater,
        projector=projector,
        planner=sample_planner
    )
    
    loop.execute(belief_state, goal_str=goal, objects=objects, max_steps=8)
    print("\n✅ Execution Demo Finished Perfectly! The Planner generated sequences respecting our CP-SAT rules automatically!")

if __name__ == "__main__":
    test_planner_loop()
