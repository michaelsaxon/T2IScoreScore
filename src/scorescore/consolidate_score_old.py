import argparse
import pandas as pd
from tqdm import tqdm

def gen_image_fname(id, image_id):
    id_str = str(id).zfill(3) if id < 100 else str(id)
    image_id_str = str(image_id).zfill(2) if image_id < 10 else str(image_id)
    return f"{id_str}_{image_id_str}.jpg"

def question_set_iterator(answer_df):
    current_set = []
    image_id = 0
    id = 0
    for _, image_row in tqdm(answer_df.iterrows()):
        current_id, question_id, correct = image_row[['id', 'question_id', 'correct']]
        if question_id == 0 and len(current_set) > 0:
            score = sum(current_set) / len(current_set)
            yield id, gen_image_fname(id, image_id), score
            current_set = []
            image_id += 1
            if current_id != id:
                id = current_id
                image_id = 0
        current_set.append(correct)
    yield id, gen_image_fname(id, image_id), score

def main(infile, outfile, debug):
    inlines = pd.read_csv(infile)
    outlines = ["id,image_id,score\n"]
    for id, image_id, score in question_set_iterator(inlines):
        outlines.append(f"{id},{image_id},{score}\n")

    with open(outfile, "w") as f:
        f.writelines(outlines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate output format with id, image_id, and score.')
    parser.add_argument('--infile', required=True, help='Input CSV file')
    parser.add_argument('--outfile', required=True, help='Output CSV file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    main(args.infile, args.outfile, args.debug)
