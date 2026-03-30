from typing import Dict, Any

class PredicateBelief:
    def __init__(self, init_probs: Dict[str, float] = None):
        """
        Maintains a factorized dictionary over grounded predicate probabilities.
        """
        self.probs = init_probs if init_probs is not None else {}
        self.timestamps = {k: 0 for k in self.probs}
        self.metadata = {k: {"source": "prior", "observable": True} for k in self.probs}

    def set_belief(self, predicate: str, prob: float, timestep: int, source: str = "obs"):
        self.probs[predicate] = max(1e-4, min(1.0 - 1e-4, prob)) # Clamp away from 0 and 1
        self.timestamps[predicate] = timestep
        if predicate not in self.metadata:
            self.metadata[predicate] = {}
        self.metadata[predicate]["source"] = source

    def get_belief(self, predicate: str, default: float = 0.5) -> float:
        return self.probs.get(predicate, default)

    def print_state(self):
        print("--- Current Factorized Belief State ---")
        for k, v in self.probs.items():
            print(f"  {k}: {v:.4f} (last updated t={self.timestamps[k]} from {self.metadata[k]['source']})")
