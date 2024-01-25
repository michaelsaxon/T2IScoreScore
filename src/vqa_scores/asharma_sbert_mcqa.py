# originally score.py on nlp

"""Score VQA models on HalluVision dataset."""

import json
import torch
import torch.nn.functional as F

from absl import app, flags
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import load_json, store_json

FLAGS = flags.FLAGS

flags.DEFINE_string("score", "", "Example json file with VQA output.")
flags.DEFINE_string("mode", "", "DSG if specificed otherwise assume TIFA.")
flags.DEFINE_string("output", "dummy.json", "json file to store output.")


def main(unused_argv):
    data = load_json(FLAGS.score)
    out_dict = {}
    for image, item in tqdm(data.items()):
        out_qs = []
        for q in item:
            out_q = q
            out_q["score"] = 0
            if FLAGS.mode == "DSG" or len(q["choices"]) < 3:
                if "Yes" in q["vqa_answer"]:
                    out_q["score"] = 1
            else:
                sbert_model = SBERTModel("sentence-transformers/all-mpnet-base-v2")
                mc_answer = sbert_model.multiple_choice(q["vqa_answer"], q["choices"])
                if q["answer"] == mc_answer:
                    out_q["score"] = 1
            out_qs.append(out_q)
        out_dict[image] = out_qs
    store_json(FLAGS.output, out_dict)


class SBERTModel:
    def __init__(self, ckpt="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_sentences(self, sentences):
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.model.device))

        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.detach().cpu()

    def multiple_choice(self, answer, choices):
        answer_embedding = self.embed_sentences([answer])
        choices_embedding = self.embed_sentences(choices)
        top_choice_index = torch.argmax(
            torch.matmul(choices_embedding, answer_embedding.T)
        ).item()
        return choices[top_choice_index]


if __name__ == "__main__":
    app.run(main)