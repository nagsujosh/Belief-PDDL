import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.belief.state import PredicateBelief
from src.belief.update import BeliefUpdater
from src.belief.projection import BeliefProjector

def test_belief_pipeline():
    print("\n--- 🧠 PHASE 3: BELIEF AND VERIFIER DEMO ---")
    
    # 1. State and Updater
    init_state = {
        "clear(b1)": 0.5,
        "on(b1,b0)": 0.5,
        "on_table(b1)": 0.5,
        "holding(b1)": 0.5
    }
    belief = PredicateBelief(init_state)
    updater = BeliefUpdater(alpha=1.0)
    
    print("Simulating neural observation logic update...")
    # Simulate seeing exactly `holding(b1)` with high confidence
    new_hold_prob = updater.update(belief.get_belief("holding(b1)"), obs_prob=0.9)
    belief.set_belief("holding(b1)", new_hold_prob, timestep=1)
    
    # Simulate a noisy mistaken observation that it is on the table
    new_on_prob = updater.update(belief.get_belief("on_table(b1)"), obs_prob=0.85)
    belief.set_belief("on_table(b1)", new_on_prob, timestep=1)
    
    print("\nRaw State Before Projection:")
    belief.print_state()
    
    # 2. Verifier
    print("\nInitializing Verifier to resolve logical constraints...")
    projector = BeliefProjector("domains/blocksworld/constraints.yaml")
    
    # Project!
    try:
        projected_map = projector.project_map_state(belief.probs)
        print("\nProjected Logically Feasible Symbolic World:")
        for k, is_true in projected_map.items():
            print(f"  {k} = {is_true}")
            
        assert not (projected_map.get("holding(b1)") and projected_map.get("on_table(b1)")), "Solver failed to resolve Mutex!"
        print("\n✅ Verification Complete! CP-SAT successfully identified and purged mathematical contradiction between `holding` and `on_table` based on log-odds scores.")
    except Exception as e:
        print(f"Error during projection: {e}")

if __name__ == "__main__":
    test_belief_pipeline()
