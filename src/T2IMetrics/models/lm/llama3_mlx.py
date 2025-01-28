from mlx_lm import load, generate
from typing import Optional
import logging

from .base import LanguageModel

logger = logging.getLogger(__name__)

class Llama3MLXModel(LanguageModel):
    """Wrapper for Llama 3 model using MLX backend."""
    
    def __init__(self, 
                 model_key: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
                 device: Optional[str] = None,  # Not used for MLX
                 **kwargs):
        """Initialize Llama 3 MLX model."""
        super().__init__(model_key, device)
        
        logger.debug(f"Initializing Llama 3 MLX with model: {model_key}")
        
        # Load model and tokenizer
        logger.debug("Loading model and tokenizer...")
        self.model, self.tokenizer = load(model_key)
        logger.debug("Model initialization complete")
    
    def generate(self, 
                prompt: str, 
                max_tokens: int = 700,
                temperature: float = 0,
                stop: Optional[list] = None,
                **kwargs) -> str:
        """
        Generate text using Llama 3 MLX model.
        
        Args:
            prompt: Input text to complete
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0 = deterministic)
            stop: Optional list of stop sequences (not used in MLX-LM)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text completion
        """
        logger.debug(f"MLX-LM Request:")
        logger.debug(f"Model: {self.model_key}")
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Parameters: max_tokens={max_tokens}, temperature={temperature}")
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            #temp=temperature,
            verbose=logger.isEnabledFor(logging.DEBUG),
            **kwargs
        )
        
        logger.debug(f"Generated response: {response}")
        
        return response.strip() 