import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class AnswerProcessor:
    """Processor for validating VQA answers using semantic similarity."""
    
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    def __init__(self, 
                 model_name: str = DEFAULT_MODEL,
                 device: Optional[str] = None):
        """
        Initialize answer processor.
        
        Args:
            model_name: SBERT model to use for similarity
            device: Device to run model on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def validate_answer(self, 
                       generated_answer: str,
                       expected_answer: str,
                       choices: Optional[List[str]] = None,
                       threshold: float = 0.8) -> bool:
        """
        Validate if generated answer matches expected answer.
        
        Args:
            generated_answer: Answer from VLM
            expected_answer: Ground truth answer
            choices: Optional list of valid choices
            threshold: Similarity threshold for non-choice answers
            
        Returns:
            Boolean indicating if answer is correct
        """
        # Handle yes/no questions
        if expected_answer.lower() in ['yes', 'no']:
            return self._validate_binary(generated_answer, expected_answer)
            
        # Handle multiple choice if choices provided
        if choices:
            best_choice = self.get_best_choice(generated_answer, choices)
            return best_choice.lower() == expected_answer.lower()
            
        # Otherwise use semantic similarity
        return self.compute_similarity(generated_answer, expected_answer) >= threshold
    
    def get_best_choice(self, answer: str, choices: List[str]) -> str:
        """Get most similar choice to given answer."""
        answer_embedding = self.embed_text([answer])
        choice_embeddings = self.embed_text(choices)
        
        similarities = torch.matmul(choice_embeddings, answer_embedding.T)
        best_idx = torch.argmax(similarities).item()
        
        return choices[best_idx]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        embeddings = self.embed_text([text1, text2])
        return F.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
    
    def embed_text(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for list of texts."""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            
        # Mean pooling
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        embeddings = outputs.last_hidden_state * attention_mask
        embeddings = embeddings.sum(1) / attention_mask.sum(1)
        
        # Normalize
        return F.normalize(embeddings, p=2, dim=1)
    
    def _validate_binary(self, generated: str, expected: str) -> bool:
        """Special handling for yes/no questions with semantic fallback."""
        generated = generated.lower().strip()
        expected = expected.lower().strip()
        
        # First try direct matching patterns
        if 'yes' in generated or any(pos in generated for pos in ['correct', 'true', 'right']):
            return expected == 'yes'
        if 'no' in generated or any(neg in generated for neg in ['incorrect', 'false', 'wrong']):
            return expected == 'no'

        # If no direct match found, try semantic similarity
        logger.debug(f"No direct yes/no match found in answer: '{generated}'. Using semantic similarity.")
        
        # Get the question from the answer if possible
        question = generated.split('?')[0] if '?' in generated else generated
        
        # Create yes/no variants
        yes_variant = f"{question}? Yes."
        no_variant = f"{question}? No."
        
        # Get embeddings and calculate similarities
        embeddings = self.embed_text([generated, yes_variant, no_variant])
        yes_sim = F.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
        no_sim = F.cosine_similarity(embeddings[0:1], embeddings[2:3]).item()
        
        logger.debug(f"Yes similarity: {yes_sim:.4f}")
        logger.debug(f"No similarity: {no_sim:.4f}")
        
        # Return true if the correct answer has higher similarity
        if expected == 'yes':
            return yes_sim > no_sim and yes_sim > 0.8  # Using same threshold as parent class
        else:
            return no_sim > yes_sim and no_sim > 0.8
        
        return False 