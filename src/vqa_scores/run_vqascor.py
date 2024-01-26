import argparse
from mplug import MPlugVQAScorer
from llava import LLavaVQAScorer
from fuyu import FuyuVQAScorer

class VQAProcessor:
    def __init__(self, model_type, model_path):
        if model_type == "mplug":
            self.vqa_scorer = MPlugVQAScorer(model_path=model_path)
        elif model_type == "fuyu":
            self.vqa_scorer = FuyuVQAScorer(model_path=model_path)
        elif model_type == "llava":
            self.vqa_scorer = LLavaVQAScorer(model_path=model_path)
        else:
            raise ValueError("Invalid model type")

    def process_images_and_questions(self, metadata_file, image_folder, questions, start, end, output_file):
        all_images_list = list(map(csv_line_map, open(metadata_file, "r").readlines()))
        print(len(all_images_list))
        start_idx = max(int(start), 1)

        if end == ":":
            all_images_list = all_images_list[start_idx:]
        else:
            all_images_list = all_images_list[start_idx:int(end)]

        print(len(all_images_list))

        out_lines = []
        fail_imgs = []

        for all_img_line_no in range(len(all_images_list)):
            image_line = all_images_list[all_img_line_no]
            this_id = image_line[0]
            this_fname = image_line[9]
            question_set = filter(lambda x: int(x[0]) == int(this_id), questions)

            for question_line in question_set:
                question = question_line[3]
                question_id = question_line[2]

                try:
                    answer = self.vqa_scorer.get_answer(question, image_folder + this_fname)
                    out_line = f"{str(this_id)},{this_fname},{str(question_id)},{str(answer).replace(',', '').strip()}"
                    print(out_line)
                    out_lines.append(out_line + "\n")
                except FileNotFoundError:
                    print(f"File {this_fname} not found")
                    fail_imgs.append(f"{str(this_id)},{this_fname}\n")

        with open(output_file, "w") as f:
            f.writelines(out_lines)

        with open(output_file + ".fail.csv", "w") as f:
            f.writelines(fail_imgs)

def csv_line_map(line):
    return line.strip().split(",")

def main():
    parser = argparse.ArgumentParser(description='Get answers using a VQA model for a set of questions and images.')
    parser.add_argument("-m", '--model', default="mplug", help="Choose the VQA model (mplug, fuyu, llava)")
    parser.add_argument("-q", '--questions', default="data/HalluVision_TIFA_Q.csv", help="Path to the questions CSV file (tifa, dsg)")
    parser.add_argument("-o", '--output', default="output/a_mplug_tifa.csv", help="Path to the output CSV file")
    parser.add_argument("-b", '--image_folder', default="HalluVisionFull/Final-HalluVision/", help="Base path for image files")
    parser.add_argument("-s", '--start', default="0", help="Start index for image processing")
    parser.add_argument("-e", '--end', default="HalluVisionFull/HalluVisionAll.csv", help="End index for image processing")
    parser.add_argument("-md", '--metadata_file', default="data/HalluVision.csv", help="Path to meta-data csv file")

    args = parser.parse_args()

    vqa_processor = VQAProcessor(args.model, model_path='MAGAer13/mplug-owl2-llama2-7b')

    if args.start != "0" or args.end != ":":
        args.output = args.output + f".{args.start}-{args.end}.csv"

    vqa_processor.process_images_and_questions(args.metadata_file, args.image_folder, args.questions, args.start, args.end, args.output)

if __name__ == "__main__":
    main()
