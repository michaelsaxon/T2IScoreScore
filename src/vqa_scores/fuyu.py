import torch
from PIL import Image
from transformers import FuyuForCausalLM, FuyuProcessor

from vqa_scores.vqa_score import VQAScorer


class FuyuVQAScorer(VQAScorer):
    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = FuyuProcessor.from_pretrained(model_path)
        self.model = FuyuForCausalLM.from_pretrained(model_path, device_map=self.device)

    def get_answer(self, question, image_path):
        image = Image.open(image_path, "r").resize((512, 512))
        model_inputs = self.processor(text=question, images=[image], device=self.device, return_tensors="pt").to(self.device)

        generation_output = self.model.generate(**model_inputs, max_new_tokens=512)
        generation_text = self.processor.batch_decode(generation_output[:, -512:], skip_special_tokens=True)

        answer = str(generation_text[0].strip().split("<s>")[-1].strip().replace("\n", " ").split("?")[-1]).strip('\x04').strip()

        return answer