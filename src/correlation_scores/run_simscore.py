import argparse
import csv
import os
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from transformers import (AlignModel, AlignProcessor, AutoProcessor, BlipModel,
                          CLIPModel, CLIPProcessor)

from sim_score import SimScorer

class CLIPScorer(SimScorer):
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    def calculate_score(self, image, prompt):
        image = Image.open(image)
        input = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(self.model.device)
        output = self.model(**input)
        logits_per_image = output.logits_per_image
        return logits_per_image.item()

class BlipScorer(SimScorer):
    def __init__(self):
        self.model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def calculate_score(self, image, prompt):
        image = Image.open(image)
        input = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(self.model.device)
        output = self.model(**input)
        logits_per_image = output.logits_per_image
        return logits_per_image.item()


class ALIGNScore(SimScorer):
    def __init__(self):
        self.processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model = AlignModel.from_pretrained("kakaobrain/align-base")

    def calculate_score(self, image, prompt):
        image = Image.open(image)
        input = self.processor(text=prompt, images=image, return_tensors="pt", padding=True, padding='max_length').to(self.model.device)
        output = self.model(**input)
        logits_per_image = output.logits_per_image
        return logits_per_image.item()


def score(config):
    image_folder = Path(config['image_folder'])
    scores = []
    csv_records = []

    df = pd.read_csv(config['csv_file'])
    skipped = []


    for _, row in tqdm(df.iterrows(), total=len(df)):

        image_file_path = os.path.join(image_folder, row['file_name'])
        prompt = row['target_prompt']

        if os.path.exists(image_file_path):
            score = config['score_method'].calculate_score(image_file_path, prompt)

            formatted_image_id = row['file_name'].split('/')[-1]

            record = {
                "id": row['id'],
                "image_id": formatted_image_id,
                "score": score
            }
            csv_records.append(record)
        else:
            print(f"Image not found: {row['Image']}")
            skipped.append(f"{index},{row}/n")

    df_normalized = normalize(pd.DataFrame(csv_records))

    with open(config['result_file_path'], mode="w", newline="") as file:
        fieldnames = ["id", "image_id", "score"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in df_normalized.iterrows():
            writer.writerow(row.to_dict())

    if len(skipped) > 0:
        print(f"{len(skipped)} total images skipped! Writing list...")
        with open(config['result_file_path'] + ".skipped.csv", "w") as f:
            f.writelines(skipped)

def normalize(df):
    df_scores = df['score']
    scaler = MinMaxScaler()
    df_normalized_scores = pd.DataFrame(scaler.fit_transform(df_scores.values.reshape(-1, 1)), columns=['score'])
    df['score'] = df_normalized_scores['score']
    return df


def main():

    parser = argparse.ArgumentParser(description='Calculate scores for images in a meta-data file.')
    parser.add_argument("-m", '--model', required=True, default="clip", help="Choose sim score model (clip, blip, align)")
    parser.add_argument("-o", '--output', required=True, default="output/clipscore.csv", help="Path to the output CSV file")
    parser.add_argument("-i", '--image_folder', required=True, default="data/T2IScoreScore/", help="Base path for image files")
    parser.add_argument("-md", '--metadata_file', required=True, default="data/metadata.csv", help="Path to meta-data csv file")

    args = parser.parse_args()

    if args.model == "clip":
            scorer = CLIPScorer()
    elif args.model == "blip":
            scorer = BlipScorer()
    elif args.model == "align":
            scorer = ALIGNScore()

    config = {
        'image_folder': args.image_folder,
        'csv_file': args.metadata_file,
        'result_file_path': args.output,
        'score_method': scorer
    }

    score(config)

if __name__ == "__main__":
    main()