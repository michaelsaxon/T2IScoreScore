import argparse
import csv
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from transformers import (AlignModel, AlignProcessor, AutoProcessor, BlipModel,
                          CLIPModel, CLIPProcessor)

from correlation_scores.sim_score import SimScorer


# Define a base class for scoring methods
class ScoreMethod(SimScorer):
    def __init__(self, model_path, processor_path):
        self.model = self.load_model(model_path).to("cuda")
        self.processor = self.load_processor(processor_path)

    def calculate_score(self, image, prompt):
        image = Image.open(image)
        input = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(self.model.device)
        output = self.model(**input)
        logits_per_image = output.logits_per_image
        return logits_per_image.item()

    def load_model(self, model_path):
        raise NotImplementedError("Subclasses must implement this method.")

    def load_processor(self, processor_path):
        raise NotImplementedError("Subclasses must implement this method.")

# Subclass for CLIPScore
class CLIPScorer(ScoreMethod):
    def load_model(self, model_path):
        return CLIPModel.from_pretrained(model_path)

    def load_processor(self, processor_path):
        return CLIPProcessor.from_pretrained(processor_path)

# Subclass for BLIPScore
class BlipScorer(ScoreMethod):
    def load_model(self, model_path):
        return BlipModel.from_pretrained(model_path)

    def load_processor(self, processor_path):
        return AutoProcessor.from_pretrained(processor_path)

# Subclass for ALIGNScore
class ALIGNScorer(ScoreMethod):
    def load_model(self, model_path):
        return AlignModel.from_pretrained(model_path)

    def load_processor(self, processor_path):
        return AlignProcessor.from_pretrained(processor_path)

def score(config):
    image_folder = Path(config['image_folder'])
    scores = []
    csv_records = []

    df = pd.read_csv(config['csv_file'])
    skipped = []

    for index, row in df.iterrows():
        image_file_path = os.path.join(image_folder, row['Image'])
        prompt = row['Prompt']

        if os.path.exists(image_file_path):
            print(f"Scoring {row['Image']}")
            score = config['score_method'].calculate_score(image_file_path, prompt)
            scores.append(score)
            file_path_obj = Path(image_file_path)
            filename_without_extension = file_path_obj.stem

            record = {"id": filename_without_extension, "prompt": prompt.replace(",", ""), config['score_method_name']: score}
            record[config['score_method_name']] = "{:.2f}".format(record[config['score_method_name']])
            csv_records.append(record)
        else:
            print(f"Image not found: {row['Image']}")
            skipped.append(f"{index},{row}/n")

    with open(config['result_file_path'], mode="w", newline="") as file:
        fieldnames = ["id", "prompt", config['score_method_name']]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in csv_records:
            writer.writerow(record)

    if len(skipped) > 0:
        print(f"{len(skipped)} total images skipped! Writing list...")
        with open(config['result_file_path'] + ".skipped.csv", "w") as f:
            f.writelines(skipped)

def main():
    parser = argparse.ArgumentParser(description='Calculate scores for images in a meta-data file.')
    parser.add_argument('-i', '--image-folder', default='HalluVisionFull/Final-HalluVision/', help='Path to the folder containing images')
    parser.add_argument('-m', '--metadata-file', default='data/HalluVision.csv', help='Path to the meta-data file')
    parser.add_argument('-o', '--output-dir', default='output/clipscore.csv', help='Path to the output directory')
    parser.add_argument('-s', '--score-method-name', default='CLIPScore', help='Name of the score method you want to test')

    args = parser.parse_args()

    # Configure the scoring method based on the provided argument
    if args.score_method_name == 'CLIPScore':
        config = {
            'image_folder': args.image_folder,
            'csv_file': args.metadata_file,
            'result_file_path': args.output_dir,
            'score_method_name': args.score_method_name,
            'score_method': CLIPScorer()
        }
    elif args.score_method_name == 'BLIPScore':
        config = {
            'image_folder': args.image_folder,
            'csv_file': args.metadata_file,
            'result_file_path': args.output_dir,
            'score_method_name': args.score_method_name,
            'score_method': BlipScorer()
        }
    elif args.score_method_name == 'ALIGNScorer':
        config = {
            'image_folder': args.image_folder,
            'csv_file': args.metadata_file,
            'result_file_path': args.output_dir,
            'score_method_name': args.score_method_name,
            'score_method': ALIGNScorer()
        }
    else:
        print('Score method is not defined.')
        return

    # Call the score function with the configuration, this function produce score csv file
    score(config)

if __name__ == "__main__":
    main()
