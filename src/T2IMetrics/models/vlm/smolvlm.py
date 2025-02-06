from transformers import AutoProcessor, AutoModelForVision2Seq
from .base import HuggingFaceVLM

class SmolVLMModel(HuggingFaceVLM):
    """SmolVLM model implementation."""
    
    def __init__(self, model_key="HuggingFaceTB/SmolVLM-Instruct", device=None, **kwargs):
        super().__init__(model_key, device)
        self.processor = AutoProcessor.from_pretrained(model_key, **kwargs)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_key,
            device_map=self.device,
            **kwargs
        )
        self.tokenizer = self.processor.tokenizer
    
    def get_answer(self, question: str, image):
        """Get answer for a question about an image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return answer.strip() 