import torch
import torch.nn as nn
import torchvision.models as models

class VisionBackbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=False, out_features=256):
        super().__init__()
        self.name = name
        if name == "resnet18":
            # Using basic resnet for rapid sanity checking / small Blocksworld runs.
            # Replace with torchvision.models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1) logic for Stage 1.
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_features)
        else:
            raise ValueError(f"Backbone {name} not supported yet.")
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, 224, 224) torch image batch.
        Returns: (B, out_features) raw embedding.
        """
        return self.model(x)

class DummyTextEncoder(nn.Module):
    def __init__(self, out_features=256):
        super().__init__()
        self.out_features = out_features
        # Replace completely with CLIP text embedding in real application
        self.mock_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=out_features)

    def forward(self, texts: list):
        # Simply returns a random uniform embedding to mock textual features
        b = len(texts)
        return torch.randn(b, self.out_features)
