import torch
from mplug_owl2.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mplug_owl2.conversation import SeparatorStyle, conv_templates
from mplug_owl2.mm_utils import (KeywordsStoppingCriteria,
                                 get_model_name_from_path, process_images,
                                 tokenizer_image_token)
from mplug_owl2.model.builder import load_pretrained_model
from PIL import Image

from vqa_scores.vqa_score import VQAScorer


class MPlugVQAScorer(VQAScorer):
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            None,
            model_name,
            load_8bit=False,
            load_4bit=True,
            device=self.device
        )

    def get_answer(self, question, image_path):

        #conv_conv_templates : mplug_owl2 = "A chat between a curious human and an artificial intelligence assistant. "
        # "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        conv = conv_templates["mplug_owl2"].copy()

        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        temperature = 0.7
        max_new_tokens = 50

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        return answer
