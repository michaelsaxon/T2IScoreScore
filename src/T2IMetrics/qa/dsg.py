import logging
from typing import List, Dict, Optional
from importlib import resources
from tqdm import tqdm

from .base import QAMetric
from .answer_processor import AnswerProcessor

logger = logging.getLogger(__name__)

class DSGMetric(QAMetric):
    """DSG (Detailed Scene Graph) metric for image-text alignment."""
    
    def __init__(self,
                 lm_type: str = "llama3-mlx",
                 vlm_type: str = "qwen2-mlx",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 lm_kwargs: Optional[Dict] = None,
                 vlm_kwargs: Optional[Dict] = None,
                 similarity_threshold: float = 0.8):
        """
        Initialize DSG metric.
        
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
        
        # Load DSG prompt templates using importlib.resources
        with resources.files('T2IMetrics.resources.prompts').joinpath('dsg_tuple_prompt.txt').open('r') as f:
            self.tuple_prompt = f.read()
        with resources.files('T2IMetrics.resources.prompts').joinpath('dsg_question_prompt.txt').open('r') as f:
            self.question_prompt = f.read()
        with resources.files('T2IMetrics.resources.prompts').joinpath('dsg_dependency_prompt.txt').open('r') as f:
            self.dependency_prompt = f.read()
    
    def _generate_questions(self, prompt: str) -> List[Dict]:
        """Internal method to generate questions using DSG approach."""
        logger.debug("Generating tuples...")
        tuple_response = self.lm.generate(self.tuple_prompt + prompt)
        
        logger.debug("Generating questions...")
        question_response = self.lm.generate(self.question_prompt + prompt)
        
        logger.debug("Generating dependencies...")
        dependency_response = self.lm.generate(self.dependency_prompt + prompt)
        
        # Parse responses into structured format
        questions = self._parse_questions(question_response)
        dependencies = self._parse_dependencies(dependency_response)
        
        # Add dependency information to questions
        for i, question in enumerate(questions):
            question['parent_id'] = dependencies.get(i + 1, [])
        
        return questions
    
    def _parse_questions(self, response: str) -> List[Dict]:
        """Parse question generation response into structured format."""
        questions = []
        for line in response.strip().split('\n'):
            if not line.strip():
                continue
            # Extract question from the line (format: "ID|Question")
            question = line.split('|')[-1].strip()
            questions.append({
                'question': question,
                'choices': ['yes', 'no'],
                'answer': 'yes'  # DSG assumes positive examples
            })
        return questions
    
    def _parse_dependencies(self, response: str) -> Dict[int, List[int]]:
        """Parse dependency information into question ID mapping."""
        dependencies = {}
        # Implementation will depend on exact format of dependency output
        # For now, returning empty dependencies
        return dependencies
    
    def calculate_score(self, image, prompt: str) -> float:
        """Calculate DSG score for image-prompt pair."""
        logger.info(f"Generating questions for prompt: {prompt}")
        
        questions = self._generate_questions(prompt)
        logger.info(f"Generated {len(questions)} questions")
        
        answers = []
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
            
            if self.cache_dir:
                self._cache_qa_pair(prompt, q, answer)
        
        score = self.compute_score(answers, questions)
        logger.info(f"Final DSG score: {score:.4f}")
        return score
    
    def compute_score(self, answers: List[str], questions: List[Dict]) -> float:
        """Compute final score from answers, considering dependencies."""
        if not questions:
            return 0.0
        
        correct = 0
        for i, (answer, question) in enumerate(zip(answers, questions)):
            # Check if parent questions were answered correctly
            parents_correct = True
            for parent_id in question.get('parent_id', []):
                if parent_id < i:  # Only check previous questions
                    parent_correct = self.answer_processor.validate_answer(
                        answers[parent_id],
                        questions[parent_id]['answer'],
                        choices=['yes', 'no'],
                        threshold=self.similarity_threshold
                    )
                    if not parent_correct:
                        parents_correct = False
                        break
            
            # Only count this question if all parents were correct
            if parents_correct:
                is_correct = self.answer_processor.validate_answer(
                    answer,
                    question['answer'],
                    choices=['yes', 'no'],
                    threshold=self.similarity_threshold
                )
                correct += is_correct
                logger.debug(f"Q{i+1} correct: {is_correct}")
        
        score = correct / len(questions)
        logger.debug(f"Total correct: {correct}/{len(questions)}")
        return score 