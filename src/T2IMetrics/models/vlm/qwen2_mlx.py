from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from typing import Union, Optional
from pathlib import Path
import logging

from .base import VisionLanguageModel

logger = logging.getLogger(__name__)

class Qwen2MLXModel(VisionLanguageModel):
    """Wrapper for Qwen2-VL model using MLX backend."""
    
    def __init__(self, 
                 model_key: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit",
                 device: Optional[str] = None,  # Not used for MLX
                 **kwargs):
        """Initialize Qwen2-VL MLX model."""
        super().__init__(model_key, device)
        
        logger.debug(f"Initializing Qwen2-VL MLX with model: {model_key}")
        
        # Load model, processor, and config
        logger.debug("Loading model and processor...")
        self.model, self.processor = load(model_key)
        self.config = load_config(model_key)
        logger.debug("Model initialization complete")
    
    def get_answer(self, 
                  question: str, 
                  image: Union[str, Path, Image.Image],
                  max_length: int = 256,
                  min_length: int = 1,
                  temperature: float = 1.0,
                  **kwargs) -> str:
        """Get answer for a question about an image."""
        logger.debug(f"Processing question: {question}")
        
        # Handle image input and resizing
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Resize image to 512x512
        image = image.resize((512, 512))
        
        # Save resized image to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name
        
        logger.debug("Preparing inputs...")
        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            question,
            num_images=1
        )
        
        logger.debug("Generating answer...")
        answer = generate(
            self.model,
            self.processor,
            formatted_prompt,
            [image_path],  # MLX-VLM expects a list of image paths
            verbose=logger.isEnabledFor(logging.DEBUG),
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            **kwargs
        )
        
        logger.debug(f"Generated answer: {answer}")
        
        # Clean up temporary file
        import os
        os.unlink(image_path)
        
        return answer.strip() 