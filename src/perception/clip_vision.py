import torch
import numpy as np
from PIL import Image
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPProcessor, CLIPModel = None, None

class CLIPVisionBackbone:
    """
    A real production Vision Language Model (VLM) binding for Belief-PDDL.
    Uses OpenAI's CLIP architecture to convert RGB frames and Object classes
    into high-fidelity contrastive similarity logits, completely replacing 
    the numerical Mock generator used in procedural testing.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        if device is None:
            if torch.cuda.is_available(): device = "cuda"
            else: device = "cpu"
            
        self.device = torch.device(device)
        print(f"Loading CLIP model {model_name} onto {self.device} NVIDIA CUDA Backend...")
        
        if CLIPModel is None:
            raise ImportError("Please install `transformers` to use the CLIPVisionBackbone.")
            
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # Store a cache for text embeddings since vocabulary rarely changes per episode
        self._text_cache = {} 

    def extract_image_features(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Processes a full or cropped RGB image into visual feature vectors.
        """
        pil_img = Image.fromarray(rgb_image.astype('uint8'), 'RGB')
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        return image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    def extract_text_features(self, text_descriptions: list[str]) -> torch.Tensor:
        """
        Pre-computes text representation of objects or predicates.
        e.g., ["an apple", "a microwave", "an apple inside a microwave"]
        """
        missing = [t for t in text_descriptions if t not in self._text_cache]
        if missing:
            inputs = self.processor(text=missing, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_embeds = self.model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            for i, t in enumerate(missing):
                self._text_cache[t] = text_embeds[i].unsqueeze(0)
                
        # Stack from cache
        tensors = [self._text_cache[t] for t in text_descriptions]
        return torch.cat(tensors, dim=0)

    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> np.ndarray:
        """
        Returns raw contrastive logits (temperature scaled natively by CLIP).
        These exact logits MUST be passed through `calibrator.py` before 
        hitting the BeliefUpdater tracking matrices!
        """
        with torch.no_grad():
            # Cosine similarity scaled by logit scaling factor from CLIP
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.T
        return logits_per_image.cpu().numpy()
