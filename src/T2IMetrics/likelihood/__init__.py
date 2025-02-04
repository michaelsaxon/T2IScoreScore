from .vqascore import VQAScore

AVAILABLE_METRICS = {
    'vqascore': VQAScore,
}

__all__ = list(AVAILABLE_METRICS.keys()) + ['AVAILABLE_METRICS'] 