import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple

class PerceptionCropBuilder:
    def __init__(self, target_size=(224, 224)):
        """
        Takes raw image frames and crops them centered around specific coordinates
        for passing to standard Vision backbones.
        """
        self.target_size = target_size

    def _resize_crop(self, img: Image.Image) -> torch.Tensor:
        """
        Dummies out a generic torchvision compatible transform for pure logic tracing.
        """
        img_resized = img.resize(self.target_size)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        # Convert HWC to CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.tensor(arr)

    def extract_unary_crop(self, image: Image.Image, object_id: str) -> torch.Tensor:
        """
        In a real application this uses segmentation masks or bbox inputs 
        to isolate the object_id. Returns full image placeholder for Phase 2.
        """
        return self._resize_crop(image)

    def extract_binary_crops(self, image: Image.Image, obj_a: str, obj_b: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts crop for object A, crop for object B, and the union crop 
        spanning both. Returns fixed size full image placeholders for Phase 2.
        """
        crop_a = self._resize_crop(image)
        crop_b = self._resize_crop(image)
        crop_union = self._resize_crop(image)
        return crop_a, crop_b, crop_union
