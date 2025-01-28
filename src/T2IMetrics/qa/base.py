from typing import List, Dict, Optional, Union
from pathlib import Path
from PIL import Image
from abc import abstractmethod
import json

from ..base import T2IMetric
from ..models.lm import AVAILABLE_MODELS as LM_MODELS, LanguageModel
from ..models.vlm import AVAILABLE_MODELS as VLM_MODELS, VisionLanguageModel

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
                 cache_dir: Optional[str] = None,
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
        self.current_questions = None
        
    @abstractmethod
    def generate_questions(self, prompt: str) -> List[Dict]:
        """Generate questions from prompt.
        
        Args:
            prompt: Text prompt to generate questions from
            
        Returns:
            List of question dictionaries containing at minimum:
            - question: str
            - answer_type: str (e.g., 'yes/no', 'multiple_choice', etc)
            Additional fields may be added by specific implementations
        """
        pass
        
    def answer_question(self, question: str, image: Union[str, Path, Image.Image]) -> str:
        """Answer a single question about an image.
        
        Default implementation passes directly to VLM.
        Can be overridden for custom answering logic.
        """
        image = self._load_image(image)
        return self.vlm.get_answer(question, image)
    
    @abstractmethod
    def compute_score(self, answers: List[str], questions: List[Dict]) -> float:
        """Compute final score from answers.
        
        Args:
            answers: List of answers from VLM
            questions: Original questions with expected answers/scoring info
            
        Returns:
            Float score between 0 and 1
        """
        pass
        
    def calculate_score(self, image: Union[str, Path, Image.Image], prompt: str) -> float:
        """Main entry point - generate questions, get answers, compute score."""
        # Generate or load cached questions
        if not self.current_questions:
            self.current_questions = self.generate_questions(prompt)
            
        # Get answers for all questions
        answers = []
        for q in self.current_questions:
            answer = self.answer_question(q['question'], image)
            answers.append(answer)
            
        # Cache intermediate results if requested
        if self.cache_dir:
            self._cache_results(prompt, self.current_questions, answers)
            
        return self.compute_score(answers, self.current_questions)
    
    def _cache_results(self, prompt: str, questions: List[Dict], answers: List[str]):
        """Cache intermediate results to JSON."""
        results = {
            'prompt': prompt,
            'questions': questions,
            'answers': answers
        }
        cache_file = self.cache_dir / f"{hash(prompt)}.json"
        with open(cache_file, 'w') as f:
            json.dump(results, f)
            
    def _init_lm(self, lm_type: str, **kwargs) -> LanguageModel:
        """Initialize language model from registry."""
        model_class = LM_MODELS[lm_type]
        return model_class(**kwargs)
    
    def _init_vlm(self, vlm_type: str, **kwargs) -> VisionLanguageModel:
        """Initialize vision-language model from registry."""
        model_class = VLM_MODELS[vlm_type]
        # Ensure device is passed to VLM
        kwargs['device'] = self.device
        return model_class(**kwargs) 