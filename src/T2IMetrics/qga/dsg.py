import logging
from typing import List, Dict, Optional
from importlib import resources
from tqdm import tqdm

from .base import QAMetric
from .answer_processor import AnswerProcessor

# Create module-level logger
logger = logging.getLogger(__name__)

class DSGScore(QAMetric):
    """DSG (Davidsonian Scene Graph) metric for image-text alignment."""
    
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
        
        logger.debug("Initializing DSG metric...")
        
        # Load DSG prompt templates using importlib.resources
        with resources.files('T2IMetrics.resources.prompts').joinpath('dsg_tuple_prompt.txt').open('r') as f:
            self.tuple_prompt = f.read()
        with resources.files('T2IMetrics.resources.prompts').joinpath('dsg_question_prompt.txt').open('r') as f:
            self.question_prompt = f.read()
        with resources.files('T2IMetrics.resources.prompts').joinpath('dsg_dependency_prompt.txt').open('r') as f:
            self.dependency_prompt = f.read()
    
    def _deduplicate_tuples(self, tuple_response: str) -> str:
        """Remove duplicate tuples, keeping only the first occurrence."""
        # Split into lines and filter empty ones
        lines = [line.strip() for line in tuple_response.split('\n') if line.strip()]
        
        # Extract tuple content without IDs for comparison
        seen_tuples = {}  # tuple_content -> original_line
        unique_lines = []

        for line in lines:
            logger.debug(f"line: {line}")
            try:
                # Split by | and get the tuple part
                tuple_id, tuple_content = line.split('|')
                logger.debug(f"tuple_content: {tuple_content}")
                tuple_content = tuple_content.strip()
                
                # If we haven't seen this tuple before, keep it
                if tuple_content not in seen_tuples:
                    seen_tuples[tuple_content] = line
                    unique_lines.append(line)
                else:
                    logger.debug(f"Removing duplicate tuple: {line}")
            except ValueError:
                # If line doesn't match expected format, keep it
                unique_lines.append(line)
        
        # Renumber the remaining tuples
        final_lines = []
        for i, line in enumerate(unique_lines, 1):
            try:
                _, tuple_content = line.split('|')
                final_lines.append(f"{i} | {tuple_content.strip()}")
            except ValueError:
                final_lines.append(line)
        
        return '\n'.join(final_lines)

    @QAMetric.cache_questions
    def generate_questions(self, prompt: str) -> List[Dict]:
        """Internal method to generate questions using DSG approach."""
        logger.debug("Generating tuples...")
        tuple_prompt_text = self.tuple_prompt.format(prompt=prompt)
        tuple_response = self.lm.generate(tuple_prompt_text)
        
        # Deduplicate tuples before continuing
        logger.debug("Deduplicating tuples...")
        tuple_response = self._deduplicate_tuples(tuple_response)
        logger.debug(f"Deduplicated tuples:\n{tuple_response}")
        
        logger.debug("Generating questions...")
        question_prompt_text = self.question_prompt.format(
            prompt=prompt,
            tuples=tuple_response
        )
        question_response = self.lm.generate(question_prompt_text)
        
        logger.debug("Generating dependencies...")
        dependency_prompt_text = self.dependency_prompt.format(
            prompt=prompt,
            tuples=tuple_response
        )
        dependency_response = self.lm.generate(dependency_prompt_text)
        
        # Parse responses into structured format
        id2tuple = self._parse_tuple_output(tuple_response)
        id2question = self._parse_question_output(question_response)
        id2dep = self._parse_dependency_output(dependency_response)
        
        # Add dependency information to questions
        questions = []
        for i, question in id2question.items():
            questions.append({
                'question': question,
                'choices': ['yes', 'no'],
                'answer': 'yes',  # DSG assumes positive examples
                'parent_id': id2dep.get(i, [])
            })
        
        return questions
    
    def _parse_tuple_output(self, output_str: str) -> Dict[int, str]:
        """Parse tuple generation output into structured format."""
        id2tuple = {}
        for line in output_str.strip().split('\n'):
            if not line.strip():
                continue
            try:
                tup_id, tup = line.split('|')
                tup_id = int(tup_id.strip())
                tup = tup.strip()
                id2tuple[tup_id] = tup
            except ValueError:
                continue
        return id2tuple
    
    def _parse_dependency_output(self, output_str: str) -> Dict[int, List[int]]:
        """Parse dependency information into question ID mapping."""
        id2dep = {}
        for line in output_str.strip().split('\n'):
            if not line.strip():
                continue
            try:
                question_id, dep = line.split('|')
                question_id = int(question_id.strip())
                deps = [int(d.strip()) for d in dep.strip().split(',')]
                id2dep[question_id] = deps
            except ValueError:
                continue
        return id2dep
    
    def _parse_question_output(self, output_str: str) -> Dict[int, str]:
        """Parse question generation output into structured format."""
        id2question = {}
        for line in output_str.strip().split('\n'):
            if not line.strip():
                continue
            try:
                question_id, question = line.split('|')
                question_id = int(question_id.strip())
                question = question.strip()
                id2question[question_id] = question
            except ValueError:
                continue
        return id2question
    
    def calculate_score(self, image, prompt: str) -> float:
        """Calculate DSG score for image-prompt pair."""
        logger.info(f"Generating questions for prompt: {prompt}")
        
        questions = self.generate_questions(prompt)
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
        
        score = self.compute_score(answers, questions)
        logger.info(f"Final DSG score: {score:.4f}")
        return score
    
    def compute_score(self, answers: List[str], questions: List[Dict]) -> float:
        """Compute final score from answers, considering dependencies."""
        if not questions:
            return 0.0
        
        logger.debug("Computing DSG score...")
        
        # Step 1: First pass - validate all answers independently
        initial_correctness = []
        for i, (answer, question) in enumerate(zip(answers, questions)):
            is_correct = self.answer_processor.validate_answer(
                answer,
                question['answer'],
                choices=['yes', 'no'],
                threshold=self.similarity_threshold
            )
            initial_correctness.append(is_correct)
            logger.debug(f"Q{i+1} initial validation: {is_correct}")
        
        # Step 2: Second pass - apply dependency rules
        final_correctness = initial_correctness.copy()
        for i, question in enumerate(questions):
            parent_ids = question.get('parent_id', [])
            # Skip if no parents
            if not parent_ids:
                continue
            
            # Check if all parents were correct
            parents_correct = all(
                final_correctness[parent_id - 1]  # -1 because IDs are 1-based
                for parent_id in parent_ids
                if parent_id > 0  # Skip 0 which indicates no dependency
            )
            
            # If any parent was wrong, mark this question as wrong
            if not parents_correct:
                if final_correctness[i]:  # Only log if we're changing from correct to incorrect
                    logger.debug(f"Q{i+1} marked incorrect due to parent dependencies {parent_ids}")
                final_correctness[i] = False
        
        # Step 3: Calculate final score
        correct = sum(final_correctness)
        total = len(questions)
        score = correct / total
        
        logger.debug(f"Initial correct answers: {sum(initial_correctness)}/{total}")
        logger.debug(f"Final correct answers after dependencies: {correct}/{total}")
        logger.debug(f"Final score: {score:.4f}")
        
        return score