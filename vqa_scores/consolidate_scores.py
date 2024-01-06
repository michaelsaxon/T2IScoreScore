import click

import pandas as pd

def gen_image_fname(id, image_id):
    if id < 10:
        id = "00" + str(id)
    elif id < 100:  
        id = "0" + str(id)
    else:
        id = str(id)
    if image_id < 10:
        image_id = "0" + str(image_id)
    else:
        image_id = str(image_id)
    return f"{id}_{image_id}.jpg"

# output format: id,image_id (missing from inputs),score
# input format id,question_id,...,correct
# loop through the questions (start on qid 0, accumulate until loops back and start new question)
def question_set_iterator(answer_df):
    current_set = []
    image_id = 0
    id = 0
    for image_row in tqdm(enumerate(list(answer_df.iterrows()))):
        #print(image_row)
        image_row = image_row[1][1]
        current_id, question_id, correct = image_row[['id', 'question_id', 'correct']]
        if question_id == 0 and len(current_set) > 0:
            score = sum(current_set) / len(current_set)
            print(f"{id},{score}")
            yield id, gen_image_fname(id, image_id), score
            current_set = []
            image_id += 1
            id = current_id
        current_set.append(correct)
    

@click.command()
@click.option('--infile')
@click.option('--outfile')
@click.option('--debug', is_flag=True)
def main(infile, outfile, debug):
    inlines = pd.read_csv(infile)
    outlines = ["id,image_id,score\n"]
    for id, image_id, score in question_set_iterator(inlines):
        outlines.append(f"{id},{image_id},{score}\n")
    
    with open(outfile, "w") as f:
        f.writelines(outlines)


if __name__ == "__main__":
    main()
