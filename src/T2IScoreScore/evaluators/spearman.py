from typing import List
from scipy.stats import spearmanr
import numpy as np
from .base import WalkMetricEvaluator

class SpearmanEvaluator(WalkMetricEvaluator):
    """Evaluates Spearman rank correlation along walks through the graph."""

    def evaluate_walk(self, error_counts: List[int], scores: List[float]) -> float:
        """
        Calculate Spearman correlation between error counts and scores.
        
        Args:
            error_counts: List of error counts for each node
            scores: List of metric scores for each node
            
        Returns:
            Spearman correlation coefficient
        """
        if len(error_counts) <= 1:
            return float('nan')
        
        correlation = spearmanr(np.array(error_counts), np.array(scores)).correlation
        return 0.0 if np.isnan(correlation) else correlation 