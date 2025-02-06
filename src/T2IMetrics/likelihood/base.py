from typing import Optional
import logging
from ..base import T2IMetric

logger = logging.getLogger(__name__)

class LikelihoodMetric(T2IMetric):
    """Base class for likelihood-based metrics."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device) 
        logger.warning("Likelihood metric implementation is not complete and bug tested. Beware.")