from typing import List
from scipy.stats import ks_2samp
import numpy as np
from .base import NodePairMetricEvaluator

import logging
logger = logging.getLogger(__name__)

class KSTestEvaluator(NodePairMetricEvaluator):
    """Evaluates Kolmogorov-Smirnov test between node distributions."""

    def evaluate_node_pair(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """
        Calculate KS test statistic between two score distributions.
        
        Args:
            scores1: Array of scores from first node
            scores2: Array of scores from second node
            
        Returns:
            KS test statistic
        """
        return ks_2samp(scores1, scores2)[0]