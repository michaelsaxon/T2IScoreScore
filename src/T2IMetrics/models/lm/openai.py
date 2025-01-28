import openai
from typing import Optional
from .base import LanguageModel

class OpenAIModel(LanguageModel):
    """Wrapper for OpenAI's API-based language models."""
    
    def __init__(self, 
                 model_key: str = "gpt-3.5-turbo",
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
        
        if api_key:
            openai.api_key = api_key
        elif not openai.api_key:
            raise ValueError(
                "OpenAI API key must be provided either through api_key parameter "
                "or OPENAI_API_KEY environment variable"
            )
    
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
        stop = stop or ["\n\n", "<|endoftext|>"]
        
        response = openai.ChatCompletion.create(
            model=self.model_key,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs
        )
        
        return response['choices'][0]['message']['content']