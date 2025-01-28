from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

class LanguageModel(ABC):
    """Base class for language models used in question generation."""
    
    def __init__(self, model_key: str, device: Optional[str] = None, **kwargs):
        """
        Initialize language model.
        
        Args:
            model_key: Identifier for specific model/checkpoint
            device: Device to run model on (if applicable)
            **kwargs: Additional model-specific initialization args
        """
        self.model_key = model_key
        self.device = device
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion for a prompt."""
        pass
    