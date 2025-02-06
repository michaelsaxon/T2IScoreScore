import logging
from typing import Optional

from .base import LikelihoodMetric
from ..models.vlm import AVAILABLE_MODELS as VLM_MODELS, VisionLanguageModel

logger = logging.getLogger(__name__)

class VQAScore(LikelihoodMetric):
    """VQAScore using token likelihood from VLM models."""
    
    def __init__(self, 
                 vlm_type: str = "qwen2-mlx",
                 device: Optional[str] = None, 
                 **kwargs):
        super().__init__(device)
        if vlm_type not in VLM_MODELS:
            raise ValueError(f"Unknown VLM type: {vlm_type}. Available models: {list(VLM_MODELS.keys())}")
        
        # Initialize VLM model
        self.vlm = VLM_MODELS[vlm_type](device=device, **kwargs)
    
    def calculate_score(self, image, text: str) -> float:
        """Calculate VQA score based on 'Yes' token probability."""
        question = f'Does this figure show "{text}"? Please answer yes or no.'
        
        # Get probability of "Yes" token
        score = self.vlm.get_string_probability(question, "Yes", image)
        return score 