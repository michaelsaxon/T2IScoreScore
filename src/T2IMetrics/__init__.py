# Import all metric classes
from .correlation.clip import CLIPScore
from .correlation.siglip import SiGLIPScore
from .correlation.align import ALIGNScore
from .likelihood.vqascore import VQAScore
from .qga.tifa import TIFAScore
from .qga.dsg import DSGScore

# Import registries
from .correlation import AVAILABLE_METRICS as CORRELATION_METRICS
from .likelihood import AVAILABLE_METRICS as LIKELIHOOD_METRICS
from .qga import AVAILABLE_METRICS as QGA_METRICS

# Combine all metrics
AVAILABLE_METRICS = {
    **CORRELATION_METRICS,
    **LIKELIHOOD_METRICS,
    **QGA_METRICS,
}

__all__ = [
    'CLIPScore',
    'SiGLIPScore',
    'ALIGNScore',
    'VQAScore',
    'TIFAScore',
    'DSGScore',
    'AVAILABLE_METRICS'
] 