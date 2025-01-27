from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Set
import pandas as pd
import numpy as np
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricEvaluator(ABC):
    """Base class for evaluating text-to-image metrics."""
    
    def __init__(self, scaled_avg: bool = True, invert_scores: bool = False, adjacent_only: bool = False):
        """
        Initialize evaluator.
        
        Args:
            scaled_avg: Whether to use weighted averaging based on sample counts
            invert_scores: Whether to invert scores (for metrics where higher is worse)
            adjacent_only: Whether to only compare nodes with no intermediate error counts
        """
        self.scaled_avg = scaled_avg
        self.invert_scores = invert_scores
        self.adjacent_only = adjacent_only

    @staticmethod
    def get_error_count(node_id: str) -> int:
        """Extract error count from node ID (e.g., '1a' -> 1)."""
        return int(''.join(filter(str.isdigit, node_id)))

    @abstractmethod
    def process_dataframe(self, 
                         df: pd.DataFrame,
                         score_column: str,
                         node_id_column: str = "node_id",
                         id_column: str = "id") -> Dict[int, Tuple[float, int]]:
        """Process multiple graphs from a DataFrame."""
        pass

class WalkMetricEvaluator(MetricEvaluator):
    """Base class for walk-based metrics like Spearman that evaluate ordered paths through the graph."""
    
    def get_walks(self, node_ids: List[str]) -> List[List[str]]:
        """
        Generate all valid descending walks through the graph.
        
        Args:
            node_ids: List of all node IDs in the graph
            
        Returns:
            List of walks, where each walk is a list of node IDs
        """
        # Group nodes by error count, using sets to ensure uniqueness
        nodes_by_level = {}
        for node in set(node_ids):  # Use set to get unique node IDs
            level = self.get_error_count(node)
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)
        
        # Start with root nodes (error count 0)
        walks = [[node] for node in nodes_by_level.get(0, [])]
        
        # Build walks level by level
        max_level = max(nodes_by_level.keys())
        for level in range(1, max_level + 1):
            if level not in nodes_by_level:
                continue
            new_walks = []
            for walk in walks:
                for node in nodes_by_level[level]:
                    new_walks.append(walk + [node])
            walks = new_walks
        
        # Log all walks with their values
        logger.debug("\nGenerated walks with all values:")
        for walk in walks:
            values = [f"{node}:[{', '.join(str(x) for x in node_ids.count(node) * [1])}]" for node in walk]
            logger.debug(f"  {' -> '.join(values)}")
            
        return walks

    def process_graph(self, node_ids: List[str], scores: List[float]) -> Tuple[float, int]:
        """Process a graph structure and return evaluation score."""
        # Convert to dict for easy lookup
        score_dict = defaultdict(list)
        for node, score in zip(node_ids, scores):
            score_dict[node].append(score)
        
        # Get all valid walks
        walks = self.get_walks(node_ids)
        logger.debug("\nProcessing walks:")
        
        # Process each walk
        walk_scores = []
        walk_counts = []
        
        for walk in walks:
            logger.debug(f"Evaluating walk: {' -> '.join(walk)}")
            # Get scores and error counts for this walk
            node_scores = [score_dict[node] for node in walk]
            logger.debug(f"  All values in walk: {' -> '.join(f'{node}:{scores}' for node, scores in zip(walk, node_scores))}")
            
            # Flatten scores for evaluation
            walk_scores_list = [score for node_scores_list in node_scores for score in node_scores_list]
            error_counts = [self.get_error_count(node) for node in walk for _ in score_dict[node]]
            
            # Filter NaN values
            valid_indices = [i for i, score in enumerate(walk_scores_list) 
                           if not np.isnan(score)]
            if len(valid_indices) <= 1:
                logger.debug("  Skipping walk (insufficient valid scores)")
                continue
                
            error_counts = [error_counts[i] for i in valid_indices]
            walk_scores_list = [walk_scores_list[i] for i in valid_indices]
            
            logger.debug(f"  Final values for correlation: error_counts={error_counts}, scores={walk_scores_list}")
            
            # Evaluate walk
            score = self.evaluate_walk(error_counts, walk_scores_list)
            if not np.isnan(score):
                score = -score if self.invert_scores else score
                walk_scores.append(score)
                walk_counts.append(len(valid_indices))
        
        if not walk_scores:
            return float('nan'), 0
            
        if self.scaled_avg:
            final_score = sum(s * c for s, c in zip(walk_scores, walk_counts)) / sum(walk_counts)
        else:
            final_score = sum(walk_scores) / len(walk_scores)
            
        return final_score, len(walk_scores)

    @abstractmethod
    def evaluate_walk(self, error_counts: List[int], scores: List[float]) -> float:
        """Evaluate metric scores along a single graph walk."""
        pass

    def process_dataframe(self, 
                         df: pd.DataFrame,
                         score_column: str,
                         node_id_column: str = "node_id",
                         id_column: str = "id") -> Dict[int, Tuple[float, int]]:
        """
        Process multiple graphs from a DataFrame.
        
        Args:
            df: DataFrame containing scores and node IDs
            score_column: Name of column containing scores
            node_id_column: Name of column containing node IDs
            id_column: Name of column containing graph IDs
            
        Returns:
            Dictionary mapping graph IDs to (score, walk_count) tuples
        """
        results = {}
        for graph_id in df[id_column].unique():
            graph_df = df[df[id_column] == graph_id]
            score, count = self.process_graph(
                graph_df[node_id_column].tolist(),
                graph_df[score_column].tolist()
            )
            results[graph_id] = (score, count)
        return results

class NodePairMetricEvaluator(MetricEvaluator):
    """Base class for node-pair based metrics like KS test that compare node distributions."""
    
    def evaluate_node_pairs(self, node_ids: List[str], scores: List[float]) -> float:
        """
        Calculate test statistic between node distributions.
        Uses full node IDs but only compares nodes with different error counts.
        """
        # Group scores by their node IDs
        node_to_scores = defaultdict(list)
        for node, score in zip(node_ids, scores):
            node_to_scores[node].append(score)
        
        # Convert lists to numpy arrays
        node_to_scores = {k: np.array(v) for k, v in node_to_scores.items()}
        
        # Sort nodes by error count for ordered comparisons
        nodes = sorted(node_to_scores.keys(), key=self.get_error_count)
        logger.debug("\nPairwise Test Comparisons:")
        
        # Calculate statistic for each valid pair of nodes
        pair_scores = []
        for i, node1 in enumerate(nodes):
            error1 = self.get_error_count(node1)
            for node2 in nodes[i+1:]:
                error2 = self.get_error_count(node2)
                if error1 != error2:  # Only compare nodes with different error counts
                    scores1 = node_to_scores[node1]
                    scores2 = node_to_scores[node2]
                    if len(scores1) > 0 and len(scores2) > 0:
                        pair_score = self.evaluate_node_pair(scores1, scores2)
                        if not np.isnan(pair_score):
                            pair_scores.append(pair_score)
                            logger.debug(f"Nodes {node1}->{node2} score: {pair_score:.3f}")
                            logger.debug(f"  Node {node1}: {scores1.tolist()}")
                            logger.debug(f"  Node {node2}: {scores2.tolist()}\n")
        
        if not pair_scores:
            return 0.0
        
        final_score = sum(pair_scores) / len(pair_scores)
        logger.debug(f"Final average score: {final_score:.3f}")
        return final_score

    @abstractmethod
    def evaluate_node_pair(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """Evaluate a single pair of node distributions."""
        pass

    def process_dataframe(self, 
                         df: pd.DataFrame,
                         score_column: str,
                         node_id_column: str = "node_id",
                         id_column: str = "id") -> Dict[int, Tuple[float, int]]:
        """
        Process multiple sets of node pairs from a DataFrame.
        
        Args:
            df: DataFrame containing scores and node IDs
            score_column: Name of column containing scores
            node_id_column: Name of column containing node IDs
            id_column: Name of column containing graph IDs
            
        Returns:
            Dictionary mapping graph IDs to (score, pair_count) tuples
        """
        results = {}
        for graph_id in df[id_column].unique():
            graph_df = df[df[id_column] == graph_id]
            score = self.evaluate_node_pairs(
                graph_df[node_id_column].tolist(),
                graph_df[score_column].tolist()
            )
            # For now, using 1 as count since we don't track pairs
            results[graph_id] = (score, 1)
        return results