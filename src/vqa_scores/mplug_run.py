import argparse

import click
import torch
from mplug_owl2.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mplug_owl2.conversation import SeparatorStyle, conv_templates
from mplug_owl2.mm_utils import (KeywordsStoppingCriteria,
                                 get_model_name_from_path, process_images,
                                 tokenizer_image_token)
from mplug_owl2.model.builder import load_pretrained_model
from PIL import Image


def get_mplug_answer(query, image_file, image_processor, model, tokenizer):
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temperature = 0.7
    max_new_tokens = 50

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs

def csv_line_map(line):
    return line.strip().split(",")

def main():
    parser = argparse.ArgumentParser(description='Get answers using MPlug model for a set of questions and images.')
    parser.add_argument("-q", '--questions', default="HalluVisionFull/HalluVision_TIFA_Q.csv", help="Path to the questions CSV file")
    parser.add_argument("-o", '--output', default="output_csvs/a_mplug_tifa.csv", help="Path to the output CSV file")
    parser.add_argument("-b", '--image_folder', default="HalluVisionFull/Final-HalluVision/", help="Base path for image files")
    parser.add_argument("-s", '--start', default="0", help="Start index for image processing")
    parser.add_argument("-e", '--end', default="HalluVisionFull/HalluVisionAll.csv", help="End index for image processing")
    parser.add_argument("-m", '--metadata_file', default=":", help="Path to meta-data csv file")


    args = parser.parse_args()

    if args.start != "0" or args.end != ":":
        args.output = args.output + f".{args.start}-{args.end}.csv"

    questions = list(map(csv_line_map, open(args.questions, "r").readlines()))[1:]

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading...")
    print("Debug 1!")
    model_path = 'MAGAer13/mplug-owl2-llama2-7b'
    query = "Answer the following question in short: Are there any hotdogs in the image?"

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
                        model_path,
                        None,
                        model_name,
                        load_8bit=False,
                        load_4bit=True,
                        device="cuda")

    print("Model loaded!")

    all_images_list = list(map(csv_line_map, open(args.metadata_file, "r").readlines()))
    print(len(all_images_list))

    start_idx = max(int(args.start), 1)

    if args.end == ":":
        all_images_list = all_images_list[start_idx:]
    else:
        all_images_list = all_images_list[start_idx:int(args.end)]

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
            question = question_line[3]
            question_id = question_line[2]

            try:
                answer = get_mplug_answer(question, args.base_path + this_fname, image_processor, model, tokenizer)
                out_line = f"{str(this_id)},{this_fname},{str(question_id)},{str(answer).replace(',', '').strip()}"
                print(out_line)
                out_lines.append(out_line + "\n")
            except FileNotFoundError:
                print(f"File {this_fname} not found")
                fail_imgs.append(f"{str(this_id)},{this_fname}\n")

    with open(args.output, "w") as f:
        f.writelines(out_lines)

    with open(args.output + ".fail.csv", "w") as f:
        f.writelines(fail_imgs)

if __name__ == "__main__":
    main()
