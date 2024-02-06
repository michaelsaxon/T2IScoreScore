import argparse
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

"""
To run this script, use the following command in the terminal:

python postprocess.py --raw_answer_file blip1_dsg.csv --score_file blip1_dsg_score.csv --question_gen_method dsg --question_file TS2_DSG_Q.csv

Command-line arguments:
  --raw_answer_file         Path to the raw answer file (CSV format).
  --score_file              Path to the output score file (CSV format).
  --question_gen_method     Question generation method (e.g., 'dsg').
  --question_file           Path to the question file (CSV format).
"""

class SBERTModel:

    DEFAULT_MODEL_CHECKPOINT = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, ckpt=None):
        if ckpt is None:
            ckpt = self.DEFAULT_MODEL_CHECKPOINT
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

class AnswerProcessor:
    def __init__(self, sbert_model, question_gen_method):
        self.sbert_model = sbert_model
        self.question_gen_method = question_gen_method

    def process_answers(self, answer_df, question_df):
        processed_lines = ['id,question_id,vqa_answer,mc_answer,correct\n']

        for _, image_row in enumerate(tqdm(list(answer_df.iterrows()))):
            id, question_id, vqa_answer = image_row[1][['id', 'question_id', 'vqa_answer']]

            if isinstance(vqa_answer, float):
                vqa_answer = " " if math.isnan(vqa_answer) else str(vqa_answer)

            answer_row = question_df.loc[question_df['id'] == id].loc[self.question_df['question_id'] == question_id].iloc[0]
            choices, correct_answer = answer_row[['choices', 'answer']]
            choices = choices.split('|')

            correct, mc_answer = self.get_mc_answer(correct_answer, vqa_answer, choices)
            processed_lines.append(f"{id},{question_id},{vqa_answer},{mc_answer},{correct}\n")

        return processed_lines

    def get_mc_answer(self, correct_answer, vqa_answer, choices):
        if self.question_gen_method == "dsg" or len(choices) < 3:
            if "yes" in vqa_answer.lower():
                return 1, "yes"
            mc_answer = "no"
        else:
            mc_answer = self.sbert_model.multiple_choice(vqa_answer, choices)
            if correct_answer.lower().strip() == mc_answer.lower().strip():
                return 1, mc_answer
        return 0, mc_answer


def gen_image_fname(id, sequence_number):
    id_str = str(id).zfill(3) if id < 100 else str(id)
    sequence_str = str(sequence_number).zfill(2)
    return f"{id_str}.{sequence_str}.jpg"

def get_avg_scores(df):
    for id_value in df['id'].unique():
        id_rows = df[df['id'] == id_value]

        sequence_number = 0
        start_index = 0
        current_set = []

        for i in range(len(id_rows)):
            if id_rows.iloc[i]['question_id'] == 0 and len(current_set) > 0:
                avg_score = np.mean(current_set)
                current_id = id_rows.iloc[start_index]['id']
                image_filename = gen_image_fname(current_id, sequence_number)

                yield {'id': current_id, 'image_id': image_filename, 'score': avg_score}

                start_index = i + 1
                sequence_number += 1
                current_set = []
                current_set.append(float(id_rows.iloc[i]['correct']))
            else:
                current_set.append(float(id_rows.iloc[i]['correct']))

        avg_score = np.mean(current_set)
        current_id = id_rows.iloc[start_index]['id']
        image_filename = gen_image_fname(current_id, sequence_number)

        yield {'id': current_id, 'image_id': image_filename, 'score': avg_score}

def main(score_file, question_gen_method, question_file, raw_answer_file):
    answer_df = pd.read_csv(raw_answer_file, header=None, names=['id', 'image_path', 'question_id', 'vqa_answer'], delimiter=',')

    sbert_model = SBERTModel("sentence-transformers/all-mpnet-base-v2")
    answer_processor = AnswerProcessor(sbert_model, question_gen_method)

    question_df = pd.read_csv(question_file)

    mc_answer_file = raw_answer_file.replace(".csv", "_mc.csv")

    print("Converting answers to score per image for file:", raw_answer_file)

    processed_lines = answer_processor.process_answers(answer_df, question_df)

    with open(mc_answer_file, 'w') as f:
        f.writelines(processed_lines)

    print("Getting average score through each image for file:", mc_answer_file)

    processed_df = pd.read_csv(mc_answer_file)

    result_generator = get_avg_scores(processed_df)

    result_data = list(result_generator)

    result_df = pd.DataFrame(result_data)

    result_df.to_csv(score_file, index=False, columns=['id', 'image_id', 'score'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate output format with id, image_id, and score.')
    parser.add_argument('-s', '--score_file', required=True, help='Output CSV file for scores')
    parser.add_argument('-q', '--question_file', required=True, help='Question CSV file')
    parser.add_argument('-m', '--question_gen_method', default='dsg', required=True, help='Question generation method (tifa, dsg)')
    parser.add_argument('-a', '--raw_answer_file', required=True, help='Input CSV file for raw answers')

    args = parser.parse_args()
    main(args.score_file, args.question_gen_method, args.question_file, args.raw_answer_file)
