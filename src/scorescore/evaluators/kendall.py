from typing import List
from scipy.stats import kendalltau
import numpy as np
from .base import WalkMetricEvaluator

class KendallEvaluator(WalkMetricEvaluator):
    """Evaluates Kendall's tau correlation along walks through the graph."""

    def evaluate_walk(self, error_counts: List[int], scores: List[float]) -> float:
        """
        Calculate Kendall's tau between error counts and scores.
        
        Args:
            error_counts: List of error counts for each node
            scores: List of metric scores for each node
            
        Returns:
            Kendall's tau correlation coefficient
        """
        if len(walk_x) <= 1:
            return float('nan')
        
        correlation = kendalltau(np.array(walk_x), np.array(walk_y)).correlation
        return 0.0 if np.isnan(correlation) else correlation 