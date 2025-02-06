from transformers import AutoProcessor, AutoModel
from .base import CorrelationMetric

class SiGLIPScore(CorrelationMetric):
    """SiGLIPScore implementation using Google's SiGLIP model."""
    
    def __init__(self, device=None, **kwargs):
        super().__init__(device)
        model_id = "google/siglip-so400m-patch14-384"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device) 