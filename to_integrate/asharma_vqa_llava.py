# originally called script.py

"""Computes end-to-end LLaVA 1.5 on image dataset."""

import csv
import json

import pandas as pd
from absl import app, flags
# from fuyu.run_llava import run_llava

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", "", "Example csv file with images.")
flags.DEFINE_string("images", "", "Path to dataset images.")
flags.DEFINE_string("evaluation", "", "Example json file with VQA eval inputs.")
flags.DEFINE_string("output", "", "Save model output to json file.")

def main(unused_argv):
    vqa_data = load_json(FLAGS.evaluation)
    data = pd.read_csv(FLAGS.dataset)

    failed_imgs = []
    out_dict = {}

    for i, row in data.iterrows():
        out_img_res = []
        image = FLAGS.images + row["Image"]
        print(image)
        for item in vqa_data[str(row["ID"])]:
            prompt = "Question: " + item["question"]
            try:
                out_img_res.append(
                    {
                        "question_id": item["question_id"],
                        "input": item["input"],
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": item["answer"],
                        # "vqa_answer": run_llava(image, prompt)
                    }
                )
            except FileNotFoundError as e:
                print("Image: " + image + " cannot be found")
                failed_imgs.append(image)
        out_dict[image] = out_img_res

        if i % 10 == 0:
            out_dict["failed_imgs"] = list(set(failed_imgs))
            with open(FLAGS.output, "w") as outfile:
                json.dump(out_dict, outfile)

    out_dict["failed_imgs"] = list(set(failed_imgs))
    with open(FLAGS.output, "w") as outfile:
        json.dump(out_dict, outfile)


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    app.run(main)