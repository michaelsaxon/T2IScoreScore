from typing import List
import numpy as np
from .base import NodePairMetricEvaluator

class DeltaEvaluator(NodePairMetricEvaluator):
    """Evaluates mean difference between node distributions."""

    def evaluate_node_pair(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """
        Calculate mean difference between two score distributions.
        
        Args:
            scores1: Array of scores from first node
            scores2: Array of scores from second node
            
        Returns:
            Mean difference between distributions
        """
        return np.mean(scores1) - np.mean(scores2) 