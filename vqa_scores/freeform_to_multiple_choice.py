"""Score VQA models on HalluVision dataset."""

import torch
import torch.nn.functional as F

#from absl import app, flags
import click
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import pandas as pd


# excertped from the TIFA source code  
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


def fname(model, score):
    return f"output_csvs_correct/a_{model}_{score}.csv_headline.csv", f"HalluVisionAllFinal/Q_{score.upper()}_final.csv"

def get_mc_answer(sbert_model, correct_answer, vqa_answer, choices, mode = "DSG"):
    if mode == "DSG" or len(choices) < 3:
        if "yes" in vqa_answer.lower():
            return 1, "yes"
        mc_answer = "no"
    else:
        mc_answer = sbert_model.multiple_choice(vqa_answer, choices)
        if correct_answer.lower().strip() == mc_answer.lower().strip():
            return 1, mc_answer
    return 0, mc_answer


'''
fields for the output files:
Q_DSG_final.csv: id,prompt,question_id,question,choices,answer
a_fuyu_dsg.csv: id,image,question_id,vqa_answer
'''

# convert the multiple choice outputs into a single choice using sentencebert from a csv output in Michael's style
@click.command()
@click.option('--model')
@click.option('--score')
@click.option('--debug', is_flag=True)
def main(model, score, debug):
    def debug_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    answer_file, question_file = fname(model, score)
    answer_df = pd.read_csv(answer_file)
    question_df = pd.read_csv(question_file)

    mc_answer_file = answer_file.replace(".csv", "_mc.csv")

    sbert_model = SBERTModel("sentence-transformers/all-mpnet-base-v2")

    mc_answer_lines = ['id,question_id,vqa_answer,mc_answer,correct\n']

    # for every image (row in answer file) retrieve the options and answer for each question (just by reading in order)
    # then use get_mc_answer to score that image, and write it back out to a new copy of the answer file
    for idx, image_row in enumerate(tqdm(list(answer_df.iterrows()))):
        # WHY THE FUCK IS PANDAS SO STUPID
        image_row = image_row[1]

        id, question_id, vqa_answer = image_row[['id', 'question_id', 'vqa_answer']]

        debug_print(f"\n{idx}")

        debug_print(id)
        debug_print(question_id)
        debug_print(vqa_answer)

        # Maybe there's a reason simple stuff is so unintuitive in pandas :(
        answer_row = question_df.loc[question_df['id'] == id].loc[question_df['question_id'] == question_id].iloc[0]
        choices, correct_answer = answer_row[['choices', 'answer']]

        debug_print(answer_row)
        choices = choices.split('|')

        debug_print(id)
        debug_print(question_id)
        debug_print(vqa_answer)
        debug_print(choices)
        debug_print(correct_answer)
        
        correct, mc_answer = get_mc_answer(sbert_model, correct_answer, vqa_answer, choices, mode = score.upper())
        mc_answer_lines.append(f"{id},{question_id},{vqa_answer},{mc_answer},{correct}\n")

    with open(mc_answer_file, 'w') as f:
        f.writelines(mc_answer_lines)


if __name__ == "__main__":
    main()
