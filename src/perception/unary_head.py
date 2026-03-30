import torch
import torch.nn as nn

class UnaryPredicateHead(nn.Module):
    def __init__(self, visual_dim=256, text_dim=256, hidden_dim=128):
        super().__init__()
        # Input is concatenation of visual features of the crop + text embedding of predicate
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Predicts logit
        )

    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, visual_dim), (B, text_dim)
        Output: (B, 1) logit for the unary predicate.
        """
        x = torch.cat([visual_features, text_features], dim=-1)
        return self.mlp(x)
