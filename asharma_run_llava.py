import requests
import textwrap
import torch

from io import BytesIO
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import(
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image, UnidentifiedImageError


disable_torch_init()

MODEL = "4bit/llava-v1.5-13b-3GB"
model_name = get_model_name_from_path(MODEL)
CONV_MODE = "llava_v0"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True
)

def load_image(image_file):
  if image_file.startswith("http://") or image_file.startswith("https://"):
    response = requests.get(image_file)
    # print(response.headers['content-type'])
    image = Image.open(BytesIO(response.content)).convert("RGB")
  else:
    image = Image.open(image_file).convert("RGB")
  return image


def process_image(image):
  args = {"image_aspect_ratio": "pad"}
  image_tensor = process_images([image], image_processor, args)
  return image_tensor.to(model.device, dtype=torch.float16)


def create_prompt(prompt: str):
  conv = conv_templates[CONV_MODE].copy()
  roles = conv.roles
  prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
  conv.append_message(roles[0], prompt)
  conv.append_message(roles[1], None)
  return conv.get_prompt(), conv

def run_llava(img: str, prompt: str):
  image = load_image(img)
  image_tensor = process_image(image)
  prompt, conv = create_prompt(prompt)
  input_ids = (
      tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
      .unsqueeze(0)
      .to(model.device)
  )

  stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
  stopping_criteria = KeywordsStoppingCriteria(
      keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids
  )

  with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.01,
        max_new_tokens=512,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )

  return tokenizer.decode(
      output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
  ).strip()


def run_batch_llava(links):
  from PIL import Image
  res = {}
  prompt = "Describe the image"

  for i in links:
    try:
      res[i] = [run_llava(i, prompt)]
    except Image.UnidentifiedImageError as e:
      print("Failed to read this image: " + i)
  return res