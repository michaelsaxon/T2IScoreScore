"""Vision-Language Models for T2IMetrics."""

from .base import VisionLanguageModel
from .instructblip import InstructBlipModel
from .blip2 import Blip2Model
from .qwen2_mlx import Qwen2MLXModel
from .smolvlm import SmolVLMModel
from .moondream import MoondreamModel
# Add other VLM implementations as they're created

__all__ = [
    'VisionLanguageModel',
    'InstructBlipModel',
    'Blip2Model',
    'Qwen2MLXModel',
    'SmolVLMModel',
    'MoondreamModel',
]

# Registry of available VLM models
AVAILABLE_MODELS = {
    'instructblip': InstructBlipModel,
    'blip2': Blip2Model,
    'qwen2-mlx': Qwen2MLXModel,
    'smolvlm': SmolVLMModel,
    'moondream': MoondreamModel,
    # Add other models as they're implemented
}


