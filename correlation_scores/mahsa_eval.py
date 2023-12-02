import os
from transformers import CLIPProcessor, CLIPModel, BlipModel, AutoProcessor
from transformers import AlignProcessor, AlignModel
import argparse

from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path
import csv
import pandas as pd


class ScoreMethod(ABC):
    """
    Abstract base class for scoring methods.

    This abstract class defines the structure for custom scoring methods used to calculate similarity scores between an image and a prompt.
    Any scoring method should inherit from this class and implement the 'calculate_score' method.

    Attributes:
    None

    Methods:
    calculate_score(image, prompt):
    Calculate a similarity score between an image and a prompt using the custom scoring method.

    Parameters:
    image (str): The image file name for scoring. ex. "1-0.jpg".
    prompt (str): The prompt data for scoring.

    Returns:
    float: The calculated similarity score.

    Example:
    >>> class CustomScorer(ScoreMethod):
    >>>     def calculate_score(self, image, prompt):
    >>>         # Implement your custom scoring logic here
    >>>         pass
    >>>
    >>> custom_scorer = CustomScorer()
    >>> image_data = "your_image_data"
    >>> prompt_data = "your_prompt_data"
    >>> score = custom_scorer.calculate_score(image_data, prompt_data)
    """
    @abstractmethod
    def calculate_score(self, image, prompt):
        """
        Calculate a similarity score between an image and a prompt using the custom scoring method.

        Parameters:
        image (str): The image file name for scoring. ex. "1-0.jpg".
        prompt (str): The prompt data for scoring.

        Returns:
        float: The calculated similarity score.
        """
        pass

    @abstractmethod
    def calculate_score_rank(self, image_list, prompt):
        pass

# scoring method (CLIPScore)
class CLIPScorer(ScoreMethod):
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to("cuda")

    def calculate_score(self, image, prompt):
        image = Image.open(image)
        input = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(self.model.device)
        output = self.model(**input)
        logits_per_image = output.logits_per_image

        return logits_per_image.item()

    def calculate_score_rank(self, image_list, prompt):
        pass

# scoring method (BlipScore)
class BlipScorer(ScoreMethod):
    def __init__(self):
        self.model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def calculate_score(self, image, prompt):
        image = Image.open(image)
        input = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(self.model.device)
        output = self.model(**input)
        logits_per_image = output.logits_per_image

        return logits_per_image.item()

    def calculate_score_rank(self, image_list, prompt):
        pass


# scoring method (ALIGNScore)
class ALIGNScore(ScoreMethod):
    def __init__(self):
        self.processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model = AlignModel.from_pretrained("kakaobrain/align-base").to("cuda")

    def calculate_score(self, image, prompt):

      image = Image.open(image)
      input = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
      output = self.model(**input)
      logits_per_image = output.logits_per_image

      return logits_per_image.item()

    def calculate_score_rank(self, image_list, prompt):
        pass

def score(score_method, score_method_name, image_folder, csv_file, result_file_path):
    """
    Process dataset to generate a score file

    Parameters:
    score_method (ScoreMethod): An instance of a custom scoring method that should inherit from the 'ScoreMethod' abstract class.
    score_method_name (str): The name or identifier of the custom scoring method.

    Returns:
    list of float: A list of similarity scores between the image and corresponding prompt, calculated using the specified scoring method.

    This function takes an instance of a custom scoring method (score_method), which should inherit from the abstract base class 'Scorer'. The provided 'score_method_name' is used to identify the specific scoring method. The function calculates a similarity score for the given image and prompt using the specified scoring method and returns the score as a float.

    Example:
    >>> class CustomScorer(ScoreMethod):
    >>>     def calculate_score(self, image, prompt):
    >>>         # Implement your custom scoring logic here
    >>>         pass
    >>>
    """

    image_folder = Path(image_folder)
    scores = []
    prompts = []
    csv_records = []

    df = pd.read_csv(csv_file)
    skipped = []
    for index, row in df.iterrows():

        image_file_path = os.path.join(image_folder, row['Image'])
        prompt = row['Prompt']

        if os.path.exists(image_file_path):
            print(f"Scoring {row['Image']}")
            score = score_method.calculate_score(image_file_path, prompt)
            scores.append(score)
            file_path_obj = Path(image_file_path)
            filename_without_extension = file_path_obj.stem

            record = {"id": filename_without_extension, "prompt": prompt.replace(",", ""), str(score_method_name): score}
            record[str(score_method_name)] = "{:.2f}".format(record[str(score_method_name)])
            csv_records.append(record)
        else:
            print(f"Image not found: {row['Image']}")
            skipped.append(f"{index},{row}/n")

    with open(result_file_path, mode="w", newline="") as file:
        fieldnames = ["id", "prompt", score_method_name]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for record in csv_records:
            writer.writerow(record)
    if len(skipped) > 0:
        print(f"{len(skipped)} total images skipped! Writing list...")
        with open(result_file_path + ".skipped.csv", "w") as f:
            f.writelines(skipped)
    return scores


def main():
    parser = argparse.ArgumentParser(description='Calculate scores for images in a CSV file.')
    parser.add_argument('image_folder', help='Path to the folder containing images')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('output_dir', help='Path to the output directory')
    parser.add_argument('score_method_name', help='name of the score method you want to test')

    args = parser.parse_args()

    if args.score_method_name == 'CLIPScore':
        clip_scorer = CLIPScorer()
        clip_scores = score(
            clip_scorer,
            'CLIPScore',
            args.image_folder,
            args.csv_file,
            args.output_dir
        )
    elif args.score_method_name == 'BLIPScore':
        blip_scorer = BlipScorer()
        blip_scorers = score(
            blip_scorer,
            'BLIPScore',
            args.image_folder,
            args.csv_file,
            args.output_dir
        )
    elif args.score_method_name == 'ALIGNSCore':
        align_score = BlipScorer()
        align_scores = score(
            align_score,
            'ALIGNSCore',
            args.image_folder,
            args.csv_file,
            args.output_dir
        )
    else:
        print('score method is not defined.')

if __name__ == "__main__":
    main()