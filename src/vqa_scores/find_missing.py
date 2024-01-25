from typing import Dict

import click

import pandas as pd

def fname_convert(fname):
    fname = fname.strip(".jpg").strip("images/")
    if "-" in fname:
        fname_split = fname.split("-")
    else:
        fname_split = fname.split("_")
    return fname_split[0], fname_split[1]

@click.command()
@click.option('--infile')
@click.option('--reffile')
@click.option('--field', default="image_id")
def main(infile, reffile, field):
    input_df = pd.read_csv(infile)
    ref_df = pd.read_csv(reffile)

    # get list of ids from each df
    input_ids = list(map(fname_convert, input_df[field].unique()))
    ref_ids = list(map(fname_convert, ref_df["file_name"].unique()))

    # print all ids in ref that are missing from input
    print(set(ref_ids) - set(input_ids))
    print(set(input_ids) - set(ref_ids))

if __name__ == "__main__":
    main()