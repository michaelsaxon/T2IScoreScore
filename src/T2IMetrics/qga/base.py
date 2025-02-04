from typing import List, Dict, Optional, Union
from pathlib import Path
from PIL import Image
from abc import abstractmethod
import json
import hashlib
import logging

from ..base import T2IMetric
from ..models.lm import AVAILABLE_MODELS as LM_MODELS, LanguageModel
from ..models.vlm import AVAILABLE_MODELS as VLM_MODELS, VisionLanguageModel

logger = logging.getLogger(__name__)

class QAMetric(T2IMetric):
    """Base class for question-answering based metrics (TIFA, DSG etc).
    
    These metrics work by:
    1. Generating questions from the prompt
    2. Answering those questions using a VLM
    3. Computing a final score based on the answers
    """
    
    def __init__(self, 
                 lm_type: str,
                 vlm_type: str,
                 device: Optional[str] = None,
                 cache_dir: Optional[Union[str, Path]] = None,
                 lm_kwargs: Optional[Dict] = None,
                 vlm_kwargs: Optional[Dict] = None):
        """
        Initialize QA metric.
        
        Args:
            lm_type: Type of language model to use
            vlm_type: Type of vision-language model to use
            device: Device to run models on
            cache_dir: Directory to cache intermediate results
            lm_kwargs: Additional kwargs for language model initialization
            vlm_kwargs: Additional kwargs for vision-language model initialization
        """
        super().__init__(device)
        
        # Validate model choices
        if lm_type not in LM_MODELS:
            raise ValueError(f"Invalid lm_type: {lm_type}. Must be one of: {list(LM_MODELS.keys())}")
        if vlm_type not in VLM_MODELS:
            raise ValueError(f"Invalid vlm_type: {vlm_type}. Must be one of: {list(VLM_MODELS.keys())}")
            
        # Initialize models with their specific kwargs
        self.lm = self._init_lm(lm_type, **(lm_kwargs or {}))
        self.vlm = self._init_vlm(vlm_type, **(vlm_kwargs or {}))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Create cache directories if needed
        if self.cache_dir:
            self.questions_cache_dir = self.cache_dir / "questions"
            self.questions_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, *args, model_type: str = None) -> str:
        """
        Generate a unique cache key from arguments.
        
        Args:
            *args: Arguments to hash
            model_type: Either 'lm' or 'vlm' to include appropriate model name
        """
        # Start with the metric class name
        components = [self.__class__.__name__]
        
        # Add model identifier if specified
        if model_type == 'lm':
            components.append(f"lm_{self.lm.get_model_identifier()}")
        elif model_type == 'vlm':
            components.append(f"vlm_{self.vlm.get_model_identifier()}")
            
        # Add all other arguments
        components.extend(str(arg) for arg in args)
        
        # Combine and hash
        combined = "_".join(components)
        # Get first 8 chars of positive hex hash
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    def cache_questions(func):
        """Decorator to cache question generation results."""
        def wrapper(self, prompt: str, *args, **kwargs):
            if not self.cache_dir:
                return func(self, prompt, *args, **kwargs)
                
            # Try to load from cache first
            cache_key = self._get_cache_key(prompt, model_type='lm')
            cache_file = self.questions_cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                logger.debug(f"Loading cached questions for prompt: {prompt}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Generate new questions if not cached
            questions = func(self, prompt, *args, **kwargs)
            
            # Cache the new questions
            logger.debug(f"Caching questions for prompt: {prompt}")
            with open(cache_file, 'w') as f:
                json.dump(questions, f)
                
            return questions
        return wrapper
    
    def _init_lm(self, lm_type: str, **kwargs) -> LanguageModel:
        """Initialize language model from registry."""
        model_class = LM_MODELS[lm_type]
        return model_class(**kwargs)
    
    def _init_vlm(self, vlm_type: str, **kwargs) -> VisionLanguageModel:
        """Initialize vision-language model from registry."""
        model_class = VLM_MODELS[vlm_type]
        kwargs['device'] = self.device
        return model_class(**kwargs)
    
    @abstractmethod
    def generate_questions(self, prompt: str) -> List[Dict]:
        """Generate questions from prompt."""
        pass
    
    @abstractmethod
    def compute_score(self, answers: List[str], questions: List[Dict]) -> float:
        """Compute final score from answers."""
        pass
    
    def calculate_score(self, image: Union[str, Path, Image.Image], prompt: str) -> float:
        """Calculate metric score for image-prompt pair."""
        questions = self.generate_questions(prompt)
        answers = []
        for q in questions:
            answer = self.vlm.get_answer(q['question'], image)
            answers.append(answer)
        return self.compute_score(answers, questions) 