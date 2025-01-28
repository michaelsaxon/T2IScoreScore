import argparse

import pandas as pd
import seaborn as sns

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

METRIC_LABEL = {
    "clipscore": "CLIPScore",
    "alignscore": "ALIGNScore",
    "mplug_tifa": "TIFA-MPlug",
    "llava_tifa": "TIFA-LLaVA",
    "llava-alt_tifa": "TIFA-LLaVA (alt)",
    "instructblip_tifa": "TIFA-InstructBLIP",
    "blip1_tifa": "TIFA-BLIP1",
    "fuyu_tifa": "TIFA-Fuyu",
    "mplug_dsg": "DSG-MPlug",
    "llava_dsg": "DSG-LLaVA",
    "llava-alt_dsg": "DSG-LLaVA (alt)",
    "instructblip_dsg": "DSG-InstructBLIP",
    "blip1_dsg": "DSG-BLIP1",
    "fuyu_dsg": "DSG-Fuyu",
    "llmscore_ec": "LLMScore EC",
    "llmscore_over": "LLMScore Over",
    "viescore": "VIEScore",
    "gpt4v_dsg": "DSG-GPT4V",
    "gpt4v_tifa": "TIFA-GPT4V",
}

SCORE_LABEL = {
    "spearman" : (r"Ordering Score (\texttt{rank}_m)", [0,1]),
    "delta" : (r"Separation Score (\texttt{delta}_m)", [0,1.5]),
    "ks" : (r"Separation Score (\texttt{sep}_m)", [0,1.1]),
    "cost" : ("Estimated per-image cost (FLOPs)", [1e7, 1e17])
}

PARTITION_LABEL = {
    "all": "(All)",
    "synth": "(Synth)",
    "real": "(Real)",
    "nat": "(Nat)"
}

def get_score_label_range(column):
    if "-" in column:
        return SCORE_LABEL[column.split("-")[0]][0], SCORE_LABEL[column.split("-")[0]][1]
    return SCORE_LABEL[column]

def generate_pareto_line(df, x_column, y_column, x_scale, y_scale, x_ascending = True, y_comparator = max):
    df = df.sort_values(by=x_column, ascending=x_ascending)
    x = list(df[x_column])
    print(x)
    y = list(df[y_column])
    x_pareto = [x[0]]
    y_pareto = [y[0]]
    for i in range(len(x)):
        if y_comparator(y_pareto + [y[i]]) == y[i] and y[i] not in y_pareto:
            x_pareto.append(x[i])
            y_pareto.append(y[i])
    if x_ascending:
        x_pareto = [x_pareto[0]] + x_pareto + [x_scale[1]]
        y_pareto = [y_scale[0]] + y_pareto + [y_pareto[-1]]
    else:
        x_pareto = [x_pareto[0]] + x_pareto + [x_scale[0]]
        y_pareto = [y_scale[0]] + y_pareto + [y_pareto[-1]]
    return x_pareto, y_pareto

def scatter_plot_pareto(df, title, title_text, x_column, y_column, points_to_label = [], point_color="darkred", line_color="lightgrey", x_ascending=True, y_comparator=max):
    x_label, x_range = get_score_label_range(x_column)
    y_label, y_range = get_score_label_range(y_column)
    x_pareto, y_pareto = generate_pareto_line(df, x_column, y_column, x_range, y_range, x_ascending=x_ascending, y_comparator=y_comparator)
    plt.figure(figsize=(3,3))
    plt.title(title_text)
    plt.subplots_adjust(top=0.93)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if sum(x_range) > 2:
        plt.xscale("log")
    if sum(y_range) > 2:
        plt.yscale("log")
    plt.plot(x_pareto, y_pareto, '--', color=line_color, zorder=-1.0)
    plt.fill_between(x_pareto, y_pareto, y_range[0], alpha=0.1, zorder=-1.0, hatch='//', edgecolor=line_color, facecolor='none')
    plt.xlim(x_range)
    plt.ylim(y_range)
    if type(y_column) == list:
        for y in y_column:
            plt.plot(x=df[x_column], y=df[y], color=point_color)
    else:
        sns.scatterplot(x=df[x_column], y=df[y_column], color=point_color)
    df = df.set_index("metric")
    plt.tight_layout()
    for point_to_label in points_to_label:
        print(x_column)
        print(df[x_column])
        plt.text(df[x_column][point_to_label], 
                 df[y_column][point_to_label] + 0.05*(y_range[1] - y_range[0]), 
                 METRIC_LABEL[point_to_label], 
                 fontsize=9, 
                 weight="bold",
                 ha='center',
                 backgroundcolor=point_color,
                 color='white',
                 style='italic')
    plt.savefig(f"src/output/pareto/{title}.pdf")

if __name__ == "__main__":
    df = pd.read_csv("src/output/pareto_information.csv")
    scatter_plot_pareto(df, "spearman-all-cost", "(All)", "cost", "spearman-all", ["alignscore", "gpt4v_dsg", "fuyu_tifa", "viescore"])
    scatter_plot_pareto(df, "spearman-real-cost", "(Real)", "cost", "spearman-real", ["clipscore", "gpt4v_dsg", "fuyu_tifa", "viescore"], point_color="indigo")
    scatter_plot_pareto(df, "spearman-nat-cost", "(Nat)", "cost", "spearman-nat", ["alignscore", "gpt4v_dsg", "fuyu_tifa", "viescore"], point_color="darkgreen")
    scatter_plot_pareto(df, "spearman-all-ks-all", "(All)", "ks-all", "spearman-all", ["gpt4v_dsg", "viescore"], x_ascending = False)
    scatter_plot_pareto(df, "spearman-real-ks-real", "(Real)", "ks-real", "spearman-real", ["clipscore", "fuyu_tifa"], point_color="indigo", x_ascending = False)
    scatter_plot_pareto(df, "spearman-nat-ks-nat", "(Nat)", "ks-nat", "spearman-nat", ["alignscore", "fuyu_tifa"], point_color="darkgreen", x_ascending = False)
    scatter_plot_pareto(df, "delta-all-cost", "(All)", "cost", "delta-all", ["alignscore", "gpt4v_dsg", "fuyu_tifa", "viescore"])
    scatter_plot_pareto(df, "delta-real-cost", "(Real)", "cost", "delta-real", ["clipscore", "gpt4v_dsg", "fuyu_tifa", "viescore"], point_color="indigo")
    scatter_plot_pareto(df, "delta-nat-cost", "(Nat)", "cost", "delta-nat", ["alignscore", "gpt4v_dsg", "fuyu_tifa", "viescore"], point_color="darkgreen")
