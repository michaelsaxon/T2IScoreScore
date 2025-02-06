import torch
from PIL import Image
from transformers import (BitsAndBytesConfig,
                          InstructBlipForConditionalGeneration,
                          InstructBlipProcessor)


class InstructBlipVQAScorer():
    def __init__(self, model_path):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_path, quantization_config=bnb_config)
        self.processor = InstructBlipProcessor.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def get_answer(self, question, image_path):
        image = Image.open(image_path, "r").resize((512, 512))

        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)

        generation_output = self.model.generate(
                **inputs,
                do_sample=True,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        answer = self.processor.batch_decode(generation_output, skip_special_tokens=True)[0].replace(',', '').strip()
        return answer

