import torch
import math

class BeliefUpdater:
    def __init__(self, alpha: float = 1.0, beta: float = 2.0):
        """
        Log-odds update model.
        alpha: Weight for observation evidence.
        beta: Weight for deterministic transition evidence.
        """
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _logit(p: float) -> float:
        p = max(1e-4, min(1.0 - 1e-4, p))
        return math.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(l: float) -> float:
        return 1.0 / (1.0 + math.exp(-l))

    def update(self, prior_prob: float, obs_prob: float = None, action_delta: float = None) -> float:
        """
        Calculates new belief using log odds:
        l_t = l_{t-1} + alpha * logit(obs_prob) + beta * logit(action_delta)
        """
        l_prior = self._logit(prior_prob)
        
        l_obs = 0.0
        if obs_prob is not None:
            l_obs = self._logit(obs_prob)
            
        l_act = 0.0
        if action_delta is not None:
            # action_delta is expected to be a target prob (e.g. 0.99 for True, 0.01 for False)
            l_act = self._logit(action_delta)
            
        new_l = l_prior + (self.alpha * l_obs) + (self.beta * l_act)
        return self._sigmoid(new_l)
