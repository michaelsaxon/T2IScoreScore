from .tifa import TIFAScore
from .dsg import DSGScore

AVAILABLE_METRICS = {
    'tifascore': TIFAScore,
    'dsgscore': DSGScore,
}

__all__ = list(AVAILABLE_METRICS.keys()) + ['AVAILABLE_METRICS'] 