from openai import OpenAI
from typing import Optional
import logging

from .base import LanguageModel

logger = logging.getLogger(__name__)

class OpenAIModel(LanguageModel):
    """Wrapper for OpenAI's API-based language models."""
    
    def __init__(self, 
                 model_key: str = "gpt-4o-mini",
                 device: Optional[str] = None,  # Added but unused for API-based models
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize OpenAI model.
        
        Args:
            model_key: OpenAI model identifier (e.g., "gpt-3.5-turbo", "gpt-4")
            device: Unused for API-based models
            api_key: OpenAI API key. If None, will try to use environment variable
            **kwargs: Additional initialization parameters
        """
        super().__init__(model_key, device, **kwargs)
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, 
                prompt: str, 
                max_tokens: int = 700,
                temperature: float = 0,
                stop: Optional[list] = None,
                **kwargs) -> str:
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt: Input text to complete
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0 = deterministic)
            stop: Optional list of stop sequences
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text completion
        """
        # Only use <|endoftext|> as stop sequence by default
        stop = stop or ["<|endoftext|>"]
        
        logger.debug(f"OpenAI Request:")
        logger.debug(f"Model: {self.model_key}")
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Parameters: max_tokens={max_tokens}, temperature={temperature}, stop={stop}")
        
        response = self.client.chat.completions.create(
            model=self.model_key,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs
        )
        
        logger.debug("OpenAI Response:")
        logger.debug(f"Full response: {response}")
        logger.debug(f"Content: {response.choices[0].message.content}")
        logger.debug(f"Finish reason: {response.choices[0].finish_reason}")
        
        return response.choices[0].message.content