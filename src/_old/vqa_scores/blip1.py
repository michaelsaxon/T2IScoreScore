import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

class BlipVQAScorer():
    def __init__(self, model_path):

        self.model = BlipForQuestionAnswering.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def get_answer(self, question, image_path):
        image = Image.open(image_path, "r").resize((512, 512))

        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)


        generation_output = self.model.generate(**inputs)
        answer = self.processor.decode(generation_output[0], skip_special_tokens=True)

        return answer



