# Evaluators

This package contains the evaluator classes used to compute metametrics for text-to-image model scores.

## Available Evaluators

- `SpearmanEvaluator`: Computes Spearman rank correlation along walks through the error graph
- `KSTestEvaluator`: Applies Kolmogorov-Smirnov test between node distributions
- `DeltaEvaluator`: Measures separation between error level distributions
- `KendallEvaluator`: Computes Kendall's Tau correlation coefficient

## Base Classes

- `MetricEvaluator`: Abstract base class for all evaluators
- `WalkMetricEvaluator`: Base class for walk-based metrics (Spearman, Kendall)
- `NodePairMetricEvaluator`: Base class for distribution-based metrics (KS test, Delta)

## Adding New Evaluators

To add a new evaluator:

1. Inherit from appropriate base class (`WalkMetricEvaluator` or `NodePairMetricEvaluator`)
2. Implement required abstract methods:
   - For walk-based: `evaluate_walk(error_counts, scores)`
   - For node-pair: `evaluate_node_pair(scores1, scores2)`
3. Register in `__init__.py` by adding to `AVAILABLE_EVALUATORS`

Example:

```python
python
from .base import WalkMetricEvaluator
class MyEvaluator(WalkMetricEvaluator):
def evaluate_walk(self, error_counts, scores):
# Implement evaluation logic
return score
```