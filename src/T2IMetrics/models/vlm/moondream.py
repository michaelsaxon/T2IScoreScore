from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import HuggingFaceVLM

class MoondreamModel(HuggingFaceVLM):
    """Moondream model implementation."""
    
    def __init__(self, model_key="vikhyatk/moondream2", device=None, **kwargs):
        super().__init__(model_key, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_key,
            device_map=self.device,
            trust_remote_code=True,
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_key)
    
    def get_answer(self, question: str, image):
        """Get answer for a question about an image."""
        answer = self.model.query(image, question)["answer"]
        return answer.strip() 