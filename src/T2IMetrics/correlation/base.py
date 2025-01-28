from .base import T2IMetric

class CorrelationMetric(T2IMetric):
    """Base class for correlation-based metrics (CLIPScore, ALIGNScore etc).
    
    These metrics compute similarity scores between image-text pairs using
    pre-trained vision-language models.
    """
    
    def __init__(self, device=None):
        super().__init__(device)
        self.processor = None
        self.model = None
    
    def calculate_score(self, image, prompt: str) -> float:
        """Calculate similarity score between image and text."""
        image = self._load_image(image)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.logits_per_image.item() 