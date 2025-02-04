from .clip import CLIPScore
from .siglip import SiGLIPScore
from .align import ALIGNScore

AVAILABLE_METRICS = {
    'clipscore': CLIPScore,
    'siglipscore': SiGLIPScore,
    'alignscore': ALIGNScore,
}

__all__ = list(AVAILABLE_METRICS.keys()) + ['AVAILABLE_METRICS'] 