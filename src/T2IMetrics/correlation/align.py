from transformers import AlignProcessor, AlignModel
from .base import CorrelationMetric

class ALIGNScore(CorrelationMetric):
    """ALIGNScore implementation using Kakao's ALIGN model."""
    
    def __init__(self, device=None):
        super().__init__(device)
        self.processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model = AlignModel.from_pretrained("kakaobrain/align-base").to(self.device) 