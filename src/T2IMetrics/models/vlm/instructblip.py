import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor
)
from typing import Union, Optional
from pathlib import Path

from .base import VisionLanguageModel

class InstructBlipModel(VisionLanguageModel):
    """Wrapper for Salesforce's InstructBLIP model."""
    
    def __init__(self, 
                 model_key: str = "Salesforce/instructblip-vicuna-7b",
                 device: Optional[str] = None,
                 load_in_4bit: bool = True,
                 **kwargs):
        """
        Initialize InstructBLIP model.
        
        Args:
            model_key: HuggingFace model identifier
            device: Device to run model on
            load_in_4bit: Whether to use 4-bit quantization
            **kwargs: Additional initialization parameters
        """
        super().__init__(model_key, device)
        
        # Configure quantization if requested
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
            
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_key,
            quantization_config=bnb_config,
            **kwargs
        ).to(self.device)
        
        self.processor = InstructBlipProcessor.from_pretrained(model_key)
        
    def get_answer(self, 
                  question: str, 
                  image: Union[str, Path, Image.Image],
                  max_length: int = 256,
                  min_length: int = 1,
                  num_beams: int = 5,
                  temperature: float = 1.0,
                  **kwargs) -> str:
        """
        Get answer for a question about an image.
        
        Args:
            question: Question text
            image: Image to analyze
            max_length: Maximum length of generated answer
            min_length: Minimum length of generated answer
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
        
        Returns:
            Model's answer to the question
        """
        image = self._load_image(image)
        image = image.resize((512, 512))  # Standard size for InstructBLIP
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate answer
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=temperature,
            **kwargs
        )
        
        # Decode and clean answer
        answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return answer.strip() 