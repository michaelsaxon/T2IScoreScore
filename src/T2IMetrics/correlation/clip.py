from transformers import CLIPProcessor, CLIPModel
from .base import CorrelationMetric

class CLIPScore(CorrelationMetric):
    """CLIPScore implementation using OpenAI's CLIP model."""
    
    def __init__(self, device=None):
        super().__init__(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device) 