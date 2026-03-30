import torch
import torch.nn as nn

class BinaryPredicateHead(nn.Module):
    def __init__(self, visual_dim=256, text_dim=256, geom_dim=4, hidden_dim=128):
        super().__init__()
        # Input: [crop_a_vis, crop_b_vis, union_vis, text_emb, relative_geometry]
        self.mlp = nn.Sequential(
            nn.Linear((visual_dim * 3) + text_dim + geom_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output logit
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor, feat_union: torch.Tensor, 
                text: torch.Tensor, rel_geom: torch.Tensor) -> torch.Tensor:
        """
        Input dims:
        feat_*: (B, visual_dim)
        text: (B, text_dim)
        rel_geom: (B, geom_dim) [dx, dy, width_ratio, height_ratio] or similar
        """
        x = torch.cat([feat_a, feat_b, feat_union, text, rel_geom], dim=-1)
        return self.mlp(x)
