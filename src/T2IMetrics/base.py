from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path
import torch
from PIL import Image

class T2IMetric(ABC):
    """Base class for text-to-image metrics."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize metric.
        
        Args:
            device: Device to run the model on ('cuda', 'cpu', etc.)
                   If None, will use CUDA if available
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def calculate_score(self, image: Union[str, Path, Image.Image], prompt: str) -> float:
        """
        Calculate metric score for an image-prompt pair.
        
        Args:
            image: Path to image file or PIL Image object
            prompt: Text prompt to evaluate against
            
        Returns:
            Float score indicating prompt-image alignment
        """
        pass
    
    def __call__(self, image: Union[str, Path, Image.Image], prompt: str) -> float:
        """Convenience method to call calculate_score."""
        return self.calculate_score(image, prompt)
    
    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Helper to load image from path if needed."""
        if isinstance(image, (str, Path)):
            return Image.open(image)
        return image
