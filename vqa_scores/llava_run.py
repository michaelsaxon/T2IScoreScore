# originally called script.py

"""Computes end-to-end LLaVA 1.5 on image dataset."""

from fuyu.run_llava import run_llava

import csv
import json
import pandas as pd
import argparse

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def process_image(image_path, vqa_data, output_dict, run_llava_func):
    failed_imgs = []

    for item in vqa_data:
        prompt = "Question: " + item["question"]
        try:
            output_dict["questions"].append({
                "question_id": item["question_id"],
                "input": item["input"],
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],
                # "vqa_answer": run_llava_func(image_path, prompt)
            })
        except FileNotFoundError as e:
            print(f"Image: {image_path} cannot be found")
            failed_imgs.append(image_path)

    return output_dict, failed_imgs

def main(args):
    vqa_data = load_json(args.evaluation)
    data = pd.read_csv(args.dataset)

    out_dict = {"questions": []}
    failed_imgs = []

    for _, row in data.iterrows():
        image_path = args.images + row["Image"]
        print(image_path)

        out_dict, failed_imgs = process_image(image_path, vqa_data[str(row["ID"])], out_dict, run_llava)  # Replace 'run_llava' with your actual function

        if _ % 10 == 0:
            out_dict["failed_imgs"] = list(set(failed_imgs))
            with open(args.output, "w") as outfile:
                json.dump(out_dict, outfile)

    out_dict["failed_imgs"] = list(set(failed_imgs))
    with open(args.output, "w") as outfile:
        json.dump(out_dict, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes end-to-end LLaVA 1.5 on image dataset.')
    parser.add_argument('--dataset', required=True, help='Example csv file with images.')
    parser.add_argument('--images', required=True, help='Path to dataset images.')
    parser.add_argument('--evaluation', required=True, help='Example json file with VQA eval inputs.')
    parser.add_argument('--output', required=True, help='Save model output to json file.')

    args = parser.parse_args()
    main(args)
