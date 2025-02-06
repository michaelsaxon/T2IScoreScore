from .base import MetricEvaluator
from .spearman import SpearmanEvaluator
from .ks_test import KSTestEvaluator
from .kendall import KendallEvaluator
from .delta import DeltaEvaluator

AVAILABLE_EVALUATORS = {
    'spearman': SpearmanEvaluator,
    'kstest': KSTestEvaluator,
    'delta': DeltaEvaluator,
    'kendall': KendallEvaluator,
}

__all__ = [
    'MetricEvaluator',
    'SpearmanEvaluator',
    'KSTestEvaluator',
    'DeltaEvaluator',
    'KendallEvaluator',
    'AVAILABLE_EVALUATORS'
] 