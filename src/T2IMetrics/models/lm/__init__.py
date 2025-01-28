from .base import LanguageModel
from .openai import OpenAIModel
# Add other LM implementations as they're created

__all__ = [
    'LanguageModel',
    'OpenAIModel',
]

# Dictionary mapping friendly names to model classes
AVAILABLE_MODELS = {
    'openai': OpenAIModel,
    # Add other models as they're implemented
} 