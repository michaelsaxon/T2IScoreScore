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
        self.current_questions = None
        
        # Create cache directories if needed
        if self.cache_dir:
            self.questions_cache_dir = self.cache_dir / "questions"
            self.answers_cache_dir = self.cache_dir / "answers"
            self.questions_cache_dir.mkdir(parents=True, exist_ok=True)
            self.answers_cache_dir.mkdir(parents=True, exist_ok=True)
        
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
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cached_questions(self, prompt: str) -> Optional[List[Dict]]:
        """Try to load cached questions for a prompt."""
        if not self.cache_dir:
            return None
            
        cache_key = self._get_cache_key(prompt, model_type='lm')
        cache_file = self.questions_cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            logger.debug(f"Loading cached questions for prompt: {prompt}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def _cache_questions(self, prompt: str, questions: List[Dict]):
        """Cache generated questions."""
        if not self.cache_dir:
            return
            
        cache_key = self._get_cache_key(prompt, model_type='lm')
        cache_file = self.questions_cache_dir / f"{cache_key}.json"
        
        logger.debug(f"Caching questions for prompt: {prompt}")
        with open(cache_file, 'w') as f:
            json.dump(questions, f)
    
    def _get_cached_answer(self, question: str, image_hash: str) -> Optional[str]:
        """Try to load cached answer for a question-image pair."""
        if not self.cache_dir:
            return None
            
        cache_key = self._get_cache_key(question, image_hash, model_type='vlm')
        cache_file = self.answers_cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            logger.debug(f"Loading cached answer for question: {question}")
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data['answer']
        return None
    
    def _cache_answer(self, question: str, image_hash: str, answer: str):
        """Cache generated answer."""
        if not self.cache_dir:
            return
            
        cache_key = self._get_cache_key(question, image_hash, model_type='vlm')
        cache_file = self.answers_cache_dir / f"{cache_key}.json"
        
        logger.debug(f"Caching answer for question: {question}")
        with open(cache_file, 'w') as f:
            json.dump({'question': question, 'answer': answer}, f)
    
    def generate_questions(self, prompt: str) -> List[Dict]:
        """Generate questions from prompt, using cache if available."""
        # Try to load from cache first
        cached_questions = self._get_cached_questions(prompt)
        if cached_questions is not None:
            logger.info("Using cached questions")
            return cached_questions
            
        # Generate new questions if not cached
        questions = self._generate_questions(prompt)
        
        # Cache the new questions
        self._cache_questions(prompt, questions)
        return questions
    
    def answer_question(self, question: str, image: Union[str, Path, Image.Image]) -> str:
        """Answer a question about an image, using cache if available."""
        # Generate image hash for cache key
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image_data = image.tobytes()
        image_hash = hashlib.md5(image_data).hexdigest()
        
        # Try to load from cache first
        cached_answer = self._get_cached_answer(question, image_hash)
        if cached_answer is not None:
            logger.debug("Using cached answer")
            return cached_answer
            
        # Generate new answer if not cached
        answer = self.vlm.get_answer(question, image)
        
        # Cache the new answer
        self._cache_answer(question, image_hash, answer)
        return answer
    
    @abstractmethod
    def _generate_questions(self, prompt: str) -> List[Dict]:
        """Internal method to generate questions. To be implemented by subclasses."""
        pass
    
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