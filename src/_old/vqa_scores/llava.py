import argparse

import torch
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image

from vqa_scores.vqa_score import VQAScorer


class LLavaVQAScorer(VQAScorer):
    def __init__(self, model_path):

        parser = argparse.ArgumentParser()
        parser.model_base=None
        parser.device="cuda"
        parser.conv_mode=None
        parser.temperature=0.2
        parser.max_new_tokens=512
        parser.load_8bit=True
        parser.load_4bit=False
        parser.debug=False

        args = parser

        disable_torch_init()

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

        if 'llama-2' in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        if args.conv_mode is not None and self.conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(self.conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = self.conv_mode
        self.conv_mode = args.conv_mode

        return self.model, self.conv_mode, self.image_processor, self.tokenizer, self.context_len, args

    def get_answer(self, question, image_path):
        image = self.load_image(image_path)
        conv = conv_templates[self.conv_mode].copy()
        alternate_prompt = False

        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        if not alternate_prompt:
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv.append_message(conv.roles[0], inp)
            prompt = conv.get_prompt()
        else:
            prompt = f"A chat between a user and an AI assistant. The assistant gives accurate and brief answers to the human's questions. \n USER: <image> \n {question} \n ASSISTANT: "

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        return answer