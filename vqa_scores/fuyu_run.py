import argparse

import click
import io

import torch
from PIL import Image
from transformers import AutoTokenizer, FuyuConfig, FuyuForCausalLM
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor
from transformers.models.fuyu.processing_fuyu import FuyuProcessor


def run_fuyu(text_prompt, image, processor, model, out_len=30):
    # big images make memory run out :(
    image = Image.open(image, "r").resize((512,512))
    model_inputs = processor(text=text_prompt, images=[image], device=model.device, return_tensors="pt").to(model.device)

    generation_output = model.generate(**model_inputs, max_new_tokens=out_len)
    generation_text = processor.batch_decode(
        generation_output[:, -out_len:], skip_special_tokens=True
    )
    output = str(generation_text[0].strip().split("<s>")[-1].strip().replace("\n"," ").split("?")[-1]).strip('\x04').strip()
    return output

def csv_line_map(line):
    return line.strip().split(",")

def get_answers(q, o, b, s, e):
    if s != "0" or e != ":":
        o = o + f".{s}-{e}.csv"

    questions = list(map(csv_line_map, open(q, "r").readlines()))[1:]
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading...")
    print("Debug 1!")
    processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
    print("Debug 2!")
    model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", device_map=d)
    print("Debug 3!")

    print("Model loaded!")
    all_images_list = list(map(csv_line_map, open("HalluVisionFull/HalluVisionAll.csv", "r").readlines()))
    print(len(all_images_list))
    s = max(int(s), 1)

    if e == ":":
        all_images_list = all_images_list[s:]
    else:
        all_images_list = all_images_list[s:int(e)]

    print(len(all_images_list))

    # iterate over all images
    out_lines = []
    fail_imgs = []

    for all_img_line_no in range(len(all_images_list)):
        image_line = all_images_list[all_img_line_no]
        this_id = image_line[0]
        this_fname = image_line[2]
        question_set = filter(lambda x: int(x[0]) == int(this_id), questions)

        for question_line in question_set:
            question_id = question_line[2]
            question = question_line[3]

            try:
                answer = run_fuyu(question, b + this_fname, processor, model)
                out_line = f"{str(this_id)},{this_fname},{str(question_id)},{str(answer).replace(',', '').strip()}"
                print(out_line)
                out_lines.append(out_line + "\n")
            except FileNotFoundError:
                print(f"File {this_fname} not found")
                fail_imgs.append(f"{str(this_id)},{this_fname}\n")

    with open(o, "w") as f:
        f.writelines(out_lines)

    with open(o + ".fail.csv", "w") as f:
        f.writelines(fail_imgs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get answers using FUYU model for a set of questions and images.')
    parser.add_argument('-q', default="HalluVisionFull/HalluVision_TIFA_Q.csv")
    parser.add_argument('-o', default="output_csvs/a_fuyu_tifa.csv")
    parser.add_argument('-b', default="HalluVisionFull/Final-HalluVision/")
    parser.add_argument('-s', default="0")
    parser.add_argument('-e', default=":")

    args = parser.parse_args()
    get_answers(args.q, args.o, args.b, args.s, args.e)