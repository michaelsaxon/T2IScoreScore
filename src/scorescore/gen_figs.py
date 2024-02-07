import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def corr_plot(df, title, output_path):
    # generate the correlation plot
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0, vmax=0.7)
    plt.tight_layout()
    plt.title(title)
    plt.show()
    #plt.savefig(output_path + f"corr_{row}_{column}.png"

def bar_plot(df, title):
    # average by column
    df = df.mean(axis=0)
    # generate the bar plot
    #sns.barplot(x=df.index, y=df.values)
    #df.plot.bar()
    #plt.tight_layout()
    #plt.title(title)
    #plt.show()
    print(title)
    print(df)

def scatter_plot(df, title, x_column, y_column):
    sns.scatterplot(x=df[x_column], y=df[y_column])
    plt.tight_layout()
    plt.title(title)
    plt.show()

def line_plot(df, id_to_plot, metrics_to_show, output_path):

    selected_data = df[df['id'] == id_to_plot]
    selected_data = selected_data.sort_values(by='rank')

    x_axis = range(len(selected_data))
    y_columns = metrics_to_show

    sns.set(style="darkgrid")
    plt.figure(figsize=(15, 7))

    for column in y_columns:
        sns.lineplot(x=x_axis, y=selected_data[column], label=column, marker='o')

    plt.xlabel(f"Images ID (0 to {len(x_axis) - 1})")
    plt.ylabel("Score")
    plt.title(f"Scores for ID {id_to_plot}")
    plt.xticks(ticks=x_axis[::5], labels=x_axis[::5])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(output_path + f"line_plot_{id_to_plot}.png")


def main():
    parser = argparse.ArgumentParser(description='Generate correlation plots.')
    parser.add_argument('--csv_path', type=str, default='../../output/spearman_corrs_weighted.csv', help='csv input with correlation scores.')
    parser.add_argument('--output_path', type=str, default='../../output/fig/', help='DSG if specified otherwise assume TIFA.')
    parser.add_argument('--invert', type=bool, action=argparse.BooleanOptionalAction, help='Invert the scores if higher is better (e.g. BLIPScore).')
    args = parser.parse_args()


    # # load in the csv including the correlation scores and types (we will add these types to the final ver)
    # df = pd.read_csv(args.csv_path)
    # df.set_index("id", inplace=True)
    # print(df)
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

    # df = df.reindex(sorted(df.columns), axis=1)

    # generate the correlation plots
    #corr_plot(df, "")

    
    for type in ranges.keys():
        df_tmp = df.iloc[ranges[type][0]:ranges[type][1]+1]
        #corr_plot(df_tmp, f"{type} Correlation Scores", args.output_path)
        bar_plot(df_tmp, f"{type} Average Correlation Scores")

    #corr_plot(df, "Correlation Scores", args.output_path)
    bar_plot(df, "Average Correlation Scores")
    

    # scatter_plot(df, "BLIPScore vs LLava-alt TIFA", "blipscore_norm", "llava-alt_tifa")

    # Example usage for line plot:
    '''
        id_to_plot = 148  # Replace with the desired id
        metrics_to_show = ['alignscore', 'blipscore', 'clipscore']
        df = pd.read_csv(args.csv_path)
        line_plot(df, id_to_plot, metrics_to_show, args.output_path)
    '''
    id_to_plot = 148  # Replace with the desired id
    metrics_to_show = ['alignscore', 'blipscore', 'clipscore']
    df = pd.read_csv(args.csv_path)
    line_plot(df, id_to_plot, metrics_to_show, args.output_path)
    scatter_plot(df, "BLIPScore vs LLava-alt TIFA", "blipscore_norm", "llava-alt_tifa")

if __name__ == "__main__":
    main()