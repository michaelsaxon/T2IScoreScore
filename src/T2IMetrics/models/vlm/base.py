from abc import ABC, abstractmethod
from typing import Optional, Union, List
from pathlib import Path
import torch
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
    
    def get_string_probability(self, prompt: str, target_str: str, image: Union[str, Path, Image.Image]) -> float:
        """Get probability of target string given prompt and image."""
        raise NotImplementedError("String probability calculation not implemented for this model.")
    
    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Helper to load image from path if needed."""
        if isinstance(image, (str, Path)):
            return Image.open(image)
        return image 
    
    def get_model_identifier(self) -> str:
        """Get a unique identifier for this model instance."""
        return f"{self.__class__.__name__.lower()}-{self.model_key}"


class HuggingFaceVLM(VisionLanguageModel):
    """Base class for HuggingFace vision-language models."""
    
    def get_string_probability(self, prompt: str, target_str: str, image: Union[str, Path, Image.Image]) -> float:
        """Get probability of target string given prompt and image."""
        image = self._load_image(image)
        
        # Get the token(s) for target string
        target_tokens = self.tokenizer.encode(f" {target_str}", add_special_tokens=False)
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get logits for the last token
        last_token_logits = outputs.logits[0, -1, :]
        
        # Convert to probabilities
        probabilities = torch.softmax(last_token_logits, dim=-1)
        
        # Get probability of target token(s)
        # For now, just using first token if multiple
        target_prob = probabilities[target_tokens[0]].item()
        
        return target_prob
    