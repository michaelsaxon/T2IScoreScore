import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

def corr_plot(df, title, output_path):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0, vmax=0.7)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(output_path + f"corr_plot.pdf")

def bar_plot(df, title, output_path):
    df = df.mean(axis=0)
    df.plot.bar()
    plt.tight_layout()
    plt.title(title)
    plt.savefig(output_path + f"bar_plot.pdf")

def scatter_plot(df, title, x_column, y_column, output_path):
    sns.scatterplot(x=df[x_column], y=df[y_column])
    plt.tight_layout()
    plt.title(title)
    plt.savefig(output_path + f"scatter_plot.pdf")

def line_plot(df, id_to_plot, metrics_to_show, output_path):

    selected_data = df[df['id'] == id_to_plot]
    selected_data = selected_data.sort_values(by='rank')

    x_axis = range(len(selected_data))
    y_columns = metrics_to_show

    sns.set(style="darkgrid")
    plt.figure(figsize=(15, 5))

    for column in y_columns:
        sns.lineplot(x=x_axis, y=selected_data[column], label=column, marker='o')

    plt.xlabel(f"Images ID (0 to {len(x_axis) - 1})", fontname="Times New Roman", fontsize=14)
    plt.ylabel("Score", fontname="Times New Roman", fontsize=14)
    plt.title(f"Scores for ID = {id_to_plot}", fontname="Times New Roman", fontsize=16)
    plt.xticks(ticks=x_axis[::5], labels=x_axis[::5], fontname="Times New Roman", fontsize=12)
    plt.yticks(fontname="Times New Roman", fontsize=12)

    legend = plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1), fontsize='x-large')
    for text in legend.get_texts():
        text.set_fontname("Times New Roman")
        text.set_fontsize(15)

    plt.tight_layout()
    plt.savefig(output_path + f"line_plot_{id_to_plot}.pdf")

def prepare_date_for_line_plot():
    ''' Add normalized rank as a column to scores file
    '''
    df2 = pd.read_csv("output/scores-final-all.csv")
    df1 = pd.read_csv("data/metadata.csv")

    def extract_numerical_rank(rank):
        numerical_part = re.findall(r'\d+', rank)
        if numerical_part:
            return int(numerical_part[0])
        else:
            return None

    # Apply the function to the rank column in the first dataframe
    df1['rank'] = df1['rank'].apply(extract_numerical_rank)
    df1['rank'] = (df1['rank']).astype(int)


    rank_column = df1['rank']
    df2['rank'] = rank_column

    # Group by "id" and normalize the "rank" column within each group
    df2['output/normalized_rank'] = df2.groupby('id')['rank'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0
    )
    # Write the updated dataframe to a new CSV file
    df2.to_csv("output/normalized_file.csv", index=False)


def main():

    # Example usage for line plot:

    id_to_plot = 148  # Replace with the desired id
    metrics_to_show = ['alignscore', 'blipscore', 'clipscore']
    prepare_date_for_line_plot()
    df = pd.read_csv('output/normalized_file.csv')
    line_plot(df, id_to_plot, metrics_to_show, args.output_path)

    # Example usage for corr plot:

    # # load in the csv including the correlation scores and types (we will add these types to the final ver)
    df = pd.read_csv(args.csv_path)
    df.set_index("id", inplace=True)

    # # HACK current version as of Jan-29 is missing final row; manually add in the type of set each id is
    # # synthetic: 0-110; natural img: 111-135, natural err: 136-163/4
    ranges = {"synth_err": [0,110], "nat_img": [111,135], "nat_err": [136,163]}
    # #set_type = ["synthetic_error"] * (110 + 1) + ["natural_image"] * (135 - 111 + 1) + ["natural_error"] * (163 - 136 + 1)
    # #df = df.assign(set_type=set_type)

    # hack, adding in the inversion factor for scores where higher = better vs lower = better
    invert_columns = ["mplug_dsg", "mplug_tifa", "fuyu_dsg", "blipscore_norm", "llava_dsg", "fuyu_tifa", "alignscore_norm", "llava-alt_tifa", "clipscore_norm", "llava_tifa", "llava-alt_dsg", "viescore", "blip1_tifa","blip1_dsg","instructblip_tifa","instructblip_dsg"]

    if args.invert:
        for column in invert_columns:
            df[column] = -df[column]

    df = df.reindex(sorted(df.columns), axis=1)
    corr_plot(df, "")


    # Example usage for bar plot:
    for type in ranges.keys():
        df_tmp = df.iloc[ranges[type][0]:ranges[type][1]+1]
        bar_plot(df_tmp, f"{type} Average Correlation Scores", args.output_path)

    bar_plot(df, "Average Correlation Scores", args.output_path)

    # Example usage for scatter plot:
    df = pd.read_csv(args.csv_path)
    scatter_plot(df, "BLIPScore vs LLava-alt TIFA", "blipscore_norm", "llava-alt_tifa", args.output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate plots.')
    parser.add_argument('--csv_path', type=str, default='../../output/spearman_corrs_weighted.csv', help='csv input with correlation scores.')
    parser.add_argument('--output_path', type=str, default='../../output/fig/', help='DSG if specified otherwise assume TIFA.')
    parser.add_argument('--invert', type=bool, action=argparse.BooleanOptionalAction, help='Invert the scores if higher is better (e.g. BLIPScore).')
    args = parser.parse_args()
    main()