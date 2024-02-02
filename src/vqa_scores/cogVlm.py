import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

class CogVLMVQAScorer():
    def __init__(self, model_path):

        model_id = "THUDM/cogvlm-chat-hf"
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()


    def get_answer(self, question, image_path):
        image = Image.open(image_path, "r").resize((512, 512))

        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=question, history=[], images=[image], template_version='vqa')

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False, 'no_repeat_ngram_size': 3, 'early_stopping': True}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer




