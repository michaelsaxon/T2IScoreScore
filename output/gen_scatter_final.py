import pandas as pd

file_path = 'kstest_average_weighted.csv'
#file_path = 'spearman_corrs_weighted.csv'
df = pd.read_csv(file_path)

invert_columns = ["mplug_dsg", "mplug_tifa", "fuyu_dsg", "blipscore_norm", "llava_dsg", "fuyu_tifa", "alignscore_norm", "llava-alt_tifa", "clipscore_norm", "llava_tifa", "llava-alt_dsg", "viescore", "blip1_tifa","blip1_dsg","instructblip_tifa","instructblip_dsg"]
column = df.columns

#for column in invert_columns:
#    df[column] = -df[column]


df = df.rename(columns={'fuyu_dsg': 'DSG-Fuyu', 'mplug_dsg': 'DSG-mPLUG', 'llava_dsg': 'DSG-LLaVA',
                        'fuyu_tifa': 'TIFA-Fuyu', 'mplug_tifa': 'TIFA-mPLUG', 'llava_tifa': 'TIFA-LLaVA',
                        'instructblip_tifa': 'TIFA-instructBLIP', 'instructblip_dsg': 'DSG-instructBLIP',
                        'blip1_tifa': 'TIFA-BLIP1', 'blip1_dsg': 'DSG-BLIP1',
                        'llava-alt_tifa': 'TIFA-LLaVA (alt)', 'llava-alt_dsg': 'DSG-LLaVA (alt)',
                        'llmscore_over': 'LLMScore Over', 'llmscore_ec': 'LLMScore EC',
                        'blipscore_norm': 'BLIPScore', 'clipscore_norm': 'CLIPScore',
                        'alignscore_norm': 'ALIGNScore', 'viescore': 'VIEScore'})


import numpy as np

def find_correlation_pairs(df):
    df_corr = df.drop(columns=['id'])
    corr_matrix = df_corr.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_flat = corr_matrix.where(mask).stack().sort_values(ascending=False)

    top_2_pairs = corr_flat.head(2)
    bottom_2_pairs = corr_flat.tail(2)

    return top_2_pairs, bottom_2_pairs


import matplotlib.pyplot as plt

def scatter_plot(df, x_column, y_column, output_path):

    df['category'] = df.index.map(lambda x: 'Synthetic Error, Synthetic Image (Synth)' if x <= 110 else ('Synthetic Error, Natural Image (Nat)' if x <= 135 else 'Natural Error, Synthetic Image (Real)'))

    colors = {
        'Synthetic Error, Synthetic Image (Synth)': 'orange',
        'Synthetic Error, Natural Image (Nat)': 'brown',
        'Natural Error, Synthetic Image (Real)': 'skyblue'
    }

    fig, ax = plt.subplots(figsize=(5,4))

    for category, color in colors.items():
        subset = df[df['category'] == category]
        ax.scatter(subset[x_column], subset[y_column], label=category, color=color, alpha=0.7)

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    ax.legend(loc='lower right')

    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(output_path + f"_scatterplot_spearman.pdf")

    plt.show()


# Function to automatically plot the top and bottom correlation pairs
def plot_correlation_pairs(df, output_path):
    top_2_pairs, bottom_2_pairs = find_correlation_pairs(df)

    # Plotting the top 2 pairs
    m=1
    for pair in top_2_pairs.index:
        x_column, y_column = pair
        scatter_plot(df, x_column, y_column, output_path + f"top_{m}_{x_column}_{y_column}")
        m+=1

    # Plotting the bottom 2 pairs
    n=1
    for pair in bottom_2_pairs.index:
        x_column, y_column = pair
        scatter_plot(df, x_column, y_column, output_path + f"bottom_{n}_{x_column}_{y_column}")
        n+=1




plot_correlation_pairs(df, "ks-")

