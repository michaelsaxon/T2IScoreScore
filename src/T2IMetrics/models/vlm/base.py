from abc import ABC, abstractmethod
from typing import Optional, Union, List
from pathlib import Path
from PIL import Image

class VisionLanguageModel(ABC):
    """Base class for vision-language models used for question answering."""
    
    def __init__(self, model_key: str, device: Optional[str] = None, **kwargs):
        self.model_key = model_key
        self.device = device
        
    @abstractmethod
    def get_answer(self, question: str, image: Union[str, Path, Image.Image]) -> str:
        """Get answer for a question about an image."""
        pass
    
    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Helper to load image from path if needed."""
        if isinstance(image, (str, Path)):
            return Image.open(image)
        return image 
    
    def get_model_identifier(self) -> str:
        """Get a unique identifier for this model instance."""
        return f"{self.__class__.__name__.lower()}-{self.model_key}"
    