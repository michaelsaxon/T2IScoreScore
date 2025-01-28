import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from importlib import resources
from tqdm import tqdm

from .base import QAMetric
from .answer_processor import AnswerProcessor

logger = logging.getLogger(__name__)

class TIFAMetric(QAMetric):
    """TIFA (Text-to-Image Faithfulness Assessment) metric."""
    
    def __init__(self,
                 lm_type: str = "openai",
                 vlm_type: str = "instructblip",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 lm_kwargs: Optional[Dict] = None,
                 vlm_kwargs: Optional[Dict] = None,
                 similarity_threshold: float = 0.8):
        """
        Initialize TIFA metric.
        
        Args:
            lm_type: Type of language model for question generation
            vlm_type: Type of vision-language model for answering
            device: Device to run on
            cache_dir: Directory to cache intermediate results
            lm_kwargs: Additional kwargs for language model
            vlm_kwargs: Additional kwargs for vision-language model
            similarity_threshold: Similarity threshold for answer validation
        """
        super().__init__(lm_type, vlm_type, device, cache_dir, lm_kwargs, vlm_kwargs)
        self.answer_processor = AnswerProcessor(device=device)
        self.similarity_threshold = similarity_threshold
        
        # Load TIFA prompt template using importlib.resources
        with resources.files('T2IMetrics.resources.prompts').joinpath('tifa_prompt.txt').open('r') as f:
            self.prompt_template = f.read()
            
        # Define valid question categories
        self.categories = [
            'object', 'human', 'animal', 'food', 'activity',
            'attribute', 'counting', 'color', 'material',
            'spatial', 'location', 'shape', 'other'
        ]
    
    def _generate_questions(self, prompt: str) -> List[Dict]:
        """Internal method to generate questions."""
        # Format prompt for question generation
        full_prompt = self.prompt_template + prompt + "\nEntities"
        
        # Generate questions using LM
        response = self.lm.generate(full_prompt)
        
        # Parse response into question instances
        return self._parse_questions(response)
    
    def _parse_questions(self, response: str) -> List[Dict]:
        """Parse LM response into structured question format."""
        lines = response.split('\n')
        questions = []
        
        current = {}
        for line in lines[6:]:  # Skip header lines
            if line.startswith('About '):
                if current:
                    questions.append(current)
                current = {}
                # Parse entity and type
                content = line[6:-1]  # Remove 'About ' and final period
                entity, type_info = content.split(' (')
                current['element'] = entity
                current['element_type'] = type_info[:-1]  # Remove closing paren
                
            elif line.startswith('Q: '):
                current['question'] = line[3:]
            elif line.startswith('Choices: '):
                current['choices'] = [c.strip() for c in line[9:].split(',')]
            elif line.startswith('A: '):
                current['answer'] = line[3:]
                
        if current:  # Add final question if exists
            questions.append(current)
            
        # Filter invalid categories
        return [q for q in questions if q['element_type'] in self.categories]
    
    def calculate_score(self, image, prompt: str) -> float:
        """Calculate TIFA score for image-prompt pair."""
        logger.info(f"Generating questions for prompt: {prompt}")
        
        questions = self._generate_questions(prompt)
        logger.info(f"Generated {len(questions)} questions")
        
        answers = []
        # Only show progress bar if debug logging is enabled
        question_iterator = tqdm(
            questions,
            desc="Answering questions",
            disable=not logger.isEnabledFor(logging.DEBUG)
        )
        
        for i, q in enumerate(question_iterator):
            logger.debug(f"Q{i+1}: {q['question']}")
            logger.debug(f"Expected: {q['answer']}")
            
            answer = self.vlm.get_answer(q['question'], image)
            answers.append(answer)
            logger.debug(f"Generated: {answer}\n")
            
            # Cache results if requested
            if self.cache_dir:
                self._cache_qa_pair(prompt, q, answer)
        
        score = self.compute_score(answers, questions)
        logger.info(f"Final TIFA score: {score:.4f}")
        return score
    
    def compute_score(self, answers: List[str], questions: List[Dict]) -> float:
        """Compute final score from answers."""
        if not questions:
            return 0.0
        
        correct = 0
        for i, (answer, question) in enumerate(zip(answers, questions)):
            is_correct = self.answer_processor.validate_answer(
                answer,
                question['answer'],
                choices=question.get('choices'),
                threshold=self.similarity_threshold
            )
            correct += is_correct
            logger.debug(f"Q{i+1} correct: {is_correct}")
            
        score = correct / len(questions)
        logger.debug(f"Total correct: {correct}/{len(questions)}")
        return score
    
    def _cache_qa_pair(self, prompt: str, question: Dict, answer: str):
        """Cache a question-answer pair."""
        cache_file = self.cache_dir / f"{hash(prompt)}_{hash(question['question'])}.json"
        result = {
            'prompt': prompt,
            'question': question,
            'generated_answer': answer
        }
        with open(cache_file, 'w') as f:
            json.dump(result, f) 