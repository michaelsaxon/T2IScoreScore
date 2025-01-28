from .base import VisionLanguageModel
from .instructblip import InstructBlipModel
# Add other VLM implementations as they're created

__all__ = [
    'VisionLanguageModel',
    'InstructBlipModel',
]

# Dictionary mapping friendly names to model classes
AVAILABLE_MODELS = {
    'instructblip': InstructBlipModel,
    # Add other models as they're implemented
} 