import torch
from mplug_owl2.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mplug_owl2.conversation import SeparatorStyle, conv_templates
from mplug_owl2.mm_utils import (KeywordsStoppingCriteria,
                                get_model_name_from_path, process_images,
                                tokenizer_image_token)
from mplug_owl2.model.builder import load_pretrained_model
from PIL import Image
from typing import Union, Optional
from pathlib import Path
import logging

from .base import VisionLanguageModel, HuggingFaceVLM
from ..utils.generation_streamer import GenerationStreamer

logger = logging.getLogger(__name__)

class MPlugOwlModel(HuggingFaceVLM):
    """Wrapper for mPLUG-Owl model."""
    
    def __init__(self, 
                 model_key: str = "MAGAer13/mplug-owl2-llama2-7b",
                 device: Optional[str] = None,
                 load_in_4bit: bool = False,
                 **kwargs):
        """Initialize mPLUG-Owl model."""
        super().__init__(model_key, device)
        
        logger.debug(f"Initializing mPLUG-Owl with model: {model_key}")
        logger.debug(f"Device: {device}, 4-bit: {load_in_4bit}")
            
        logger.debug("Loading model and processor...")
        model_name = get_model_name_from_path(model_key)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_key,
            None,
            model_name,
            load_8bit=False,
            load_4bit=load_in_4bit,
            device=self.device
        )
        logger.debug("Model initialization complete")
        
    def get_answer(self, 
                  question: str, 
                  image: Union[str, Path, Image.Image],
                  max_length: int = 256,
                  min_length: int = 1,
                  num_beams: int = 1,
                  temperature: float = 0.7,
                  **kwargs) -> str:
        """Get answer for a question about an image."""
        logger.debug(f"Processing question: {question}")
        
        image = self._load_image(image)
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        
        logger.debug("Preparing inputs...")
        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        conv = conv_templates["mplug_owl2"].copy()
        inp = DEFAULT_IMAGE_TOKEN + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        logger.debug("Generating answer...")
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_length,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **kwargs
            )
        
        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        logger.debug(f"Final answer: {answer}")
        
        return answer 