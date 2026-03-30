import torch
import numpy as np
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPProcessor, CLIPModel = None, None


class CLIPVisionBackbone:
    """
    Real Vision-Language Model backbone using OpenAI CLIP.
    
    Provides zero-shot predicate scoring by comparing rendered RGB frames
    of the blocksworld environment against natural-language predicate queries.
    All similarity scores are L2-normalized cosine similarities that feed
    directly into the Bayesian BeliefUpdater.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        if CLIPModel is None:
            raise ImportError("Install `transformers` to use CLIPVisionBackbone: pip install transformers")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        print(f"[CLIP] Loading {model_name} on {self.device}...")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._text_embed_cache: dict[str, torch.Tensor] = {}
        print(f"[CLIP] Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Encode a (H, W, 3) uint8 numpy array into a unit-norm CLIP image embedding.
        Returns Tensor of shape (1, D).
        """
        pil_img = Image.fromarray(rgb_image.astype("uint8"), "RGB")
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Use vision_model directly to get a plain Tensor (pooler_output shape: [1, D])
            vision_out = self.model.vision_model(**inputs)
            embeds = self.model.visual_projection(vision_out.pooler_output)
        return embeds / embeds.norm(p=2, dim=-1, keepdim=True)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of text strings into unit-norm CLIP text embeddings.
        Returns Tensor of shape (N, D). Results are cached by string.
        """
        missing = [t for t in texts if t not in self._text_embed_cache]
        if missing:
            inputs = self.processor(text=missing, return_tensors="pt", padding=True,
                                    truncation=True, max_length=77).to(self.device)
            with torch.no_grad():
                text_out = self.model.text_model(**inputs)
                embeds = self.model.text_projection(text_out.pooler_output)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            for i, t in enumerate(missing):
                self._text_embed_cache[t] = embeds[i].unsqueeze(0).cpu()

        tensors = [self._text_embed_cache[t] for t in texts]
        return torch.cat(tensors, dim=0).to(self.device)

    def zero_shot_prob(self, image_embed: torch.Tensor,
                        positive_text: str, negative_text: str) -> float:
        """
        Compute P(positive | image) via softmax over [positive, negative] logits.
        Returns a float in (0, 1).
        """
        text_embeds = self.encode_texts([positive_text, negative_text])  # (2, D)
        with torch.no_grad():
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * (image_embed @ text_embeds.T)  # (1, 2)
        probs = torch.softmax(logits[0], dim=0)
        return float(probs[0].cpu())
