import numpy as np
import pandas as pd
import logging
from T2IScoreScore.evaluators import SpearmanEvaluator, KSTestEvaluator, DeltaEvaluator

# Configure logging - make sure this is at the very start
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(message)s',  # Simple format for readability
    force=True  # This ensures our configuration takes precedence
)
logger = logging.getLogger(__name__)

def test_dataframe_api():
    """Test DataFrame API for all evaluators using known examples."""
    
    # Test with real example from paper (blip1 dsg)
    df = pd.DataFrame({
        "id": [0] * 12,
        "rank": ['0', '1a', '1a', '1a', '1a', '1a', '2', '2', '2', '1b', '1b', '1b'],
        "score": [1, 0.6, 0.8, 0.6, 0.6, 0.6, 0.2, 0.2, 0.6, 0.8, 1, 0.8]
    })
    
    # Test KS evaluator
    ks_test = KSTestEvaluator()
    ks_results = ks_test.process_dataframe(df, score_column='score', node_id_column='rank')
    
    # Known target from paper
    ks_target = 0.8667  # Target includes all valid node pairs (0->1a, 0->1b, 0->2, 1a->2, 1b->2)
    logger.debug(f"KS test score (paper example): {ks_results[0][0]:.3f}")
    assert np.abs(ks_results[0][0] - ks_target) < 0.001, \
        f"KS score ({ks_results[0][0]:.3f}) should be close to target ({ks_target:.3f})"
    
    # Test Spearman with same example, both weighted and unweighted
    spearman_unweighted = SpearmanEvaluator(scaled_avg=False, invert_scores=True)
    spearman_weighted = SpearmanEvaluator(scaled_avg=True, invert_scores=True)
    
    # Process using DataFrame API
    unweighted_results = spearman_unweighted.process_dataframe(df, score_column='score', node_id_column='rank')
    weighted_results = spearman_weighted.process_dataframe(df, score_column='score', node_id_column='rank')
    
    # Known targets from paper
    spearman_unweighted_target = 0.8663  # Unweighted average of walk correlations
    spearman_weighted_target = 0.8606    # Weighted by number of samples in each walk
    
    logger.debug(f"Spearman score (unweighted): {unweighted_results[0][0]:.4f}")
    logger.debug(f"Spearman score (weighted): {weighted_results[0][0]:.4f}")
    
    assert np.abs(unweighted_results[0][0] - spearman_unweighted_target) < 0.001, \
        f"Unweighted Spearman ({unweighted_results[0][0]:.4f}) should be close to target ({spearman_unweighted_target:.4f})"
    assert np.abs(weighted_results[0][0] - spearman_weighted_target) < 0.001, \
        f"Weighted Spearman ({weighted_results[0][0]:.4f}) should be close to target ({spearman_weighted_target:.4f})"
    
    # Test Delta with simple example
    delta = DeltaEvaluator()
    delta_df = pd.DataFrame({
        "id": [0] * 4,
        "rank": ['0', '0', '1a', '1a'],
        "score": [1.0, 1.0, 0.0, 0.0]
    })
    delta_results = delta.process_dataframe(delta_df, score_column='score', node_id_column='rank')
    
    # Delta should be mean(scores_0) - mean(scores_1a) = 1.0 - 0.0 = 1.0
    assert delta_results[0][0] == 1.0, \
        f"Delta score ({delta_results[0][0]}) should be 1.0"

if __name__ == "__main__":
    test_dataframe_api() 