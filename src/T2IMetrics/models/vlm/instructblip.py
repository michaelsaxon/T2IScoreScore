import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor
)
from typing import Union, Optional
from pathlib import Path
import logging

from .base import HuggingFaceVLM
from ..utils.generation_streamer import GenerationStreamer

logger = logging.getLogger(__name__)

class InstructBlipModel(HuggingFaceVLM):
    """Wrapper for Salesforce's InstructBLIP model."""
    
    def __init__(self, 
                 model_key: str = "Salesforce/instructblip-vicuna-7b",
                 device: Optional[str] = None,
                 load_in_4bit: bool = False,
                 **kwargs):
        """Initialize InstructBLIP model."""
        super().__init__(model_key, device)
        
        logger.debug(f"Initializing InstructBLIP with model: {model_key}")
        logger.debug(f"Device: {device}, 4-bit: {load_in_4bit}")
        
        # Configure quantization if requested and on CUDA
        if load_in_4bit:
            if not self.device.startswith('cuda'):
                raise ValueError("4-bit quantization only supported on CUDA devices")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
            
        logger.debug("Loading model and processor...")
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_key,
            quantization_config=bnb_config,
            device_map=self.device if self.device.startswith('cuda') else None,
            **kwargs
        )
        
        # Remove BetterTransformer conversion
        if not self.device.startswith('cuda'):
            self.model = self.model.to(self.device)
        
        self.processor = InstructBlipProcessor.from_pretrained(model_key)
        logger.debug("Model initialization complete")
        
    def get_answer(self, 
                  question: str, 
                  image: Union[str, Path, Image.Image],
                  max_length: int = 256,
                  min_length: int = 1,
                  num_beams: int = 1,
                  temperature: float = 1.0,
                  **kwargs) -> str:
        """Get answer for a question about an image."""
        logger.debug(f"Processing question: {question}")
        
        image = self._load_image(image)
        image = image.resize((512, 512))  # Standard size for InstructBLIP
        
        logger.debug("Preparing inputs...")
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.device)
        
        logger.debug("Generating answer...")
        streamer = GenerationStreamer(self.processor)
        
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
            streamer=streamer,
            **kwargs
        )
        
        answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        logger.debug(f"Final answer: {answer}")
        
        return answer 