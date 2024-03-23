import pandas as pd
import re
from scoring import *

def compute():

    dataframe = pd.read_csv("output/scores_final_all.csv")
    ranks = pd.read_csv("data/metadata.csv")


    dataframe['rank'] = pd.Series(list(ranks["rank"]), index=dataframe.index)
    dataframe['rank'] = dataframe['rank'].apply(lambda x: int(re.sub(r'\D', '', x)))


    id_range = list(range(max(dataframe["id"])))

    print(id_range)

    metrics = list(set(dataframe.columns) - set(["id","image_id","rank"]))
    print(metrics)
    print(dataframe)

    tree_spearman_avg, tree_counts = analysis_tree_score(dataframe,
                                                         metrics,
                                                         id_range,
                                                         spearman_corr,
                                                         scaled_avg=True)

    output_dataframe = pd.DataFrame(tree_spearman_avg)
    output_treecounts = pd.DataFrame(tree_counts)

    output_dataframe.to_csv("output/spearman_corrs_weighted.csv")
    output_treecounts.to_csv("output/spearman_counts_weighted.csv")

    avg_ks_stat, ks_counts = analysis_tree_score(dataframe,
                                                metrics, id_range,
                                                between_nodepair_ks_score,
                                                scaled_avg=True)

    output_dataframe = pd.DataFrame(avg_ks_stat)
    output_kscounts = pd.DataFrame(ks_counts)

    output_dataframe.to_csv("output/kstest_average_weighted.csv")
    output_kscounts.to_csv("output/kstest_counts_weighted.csv")


    node_variances, node_counts = within_node_score(dataframe, metrics, id_range, variance)
    nv_dataframe = pd.DataFrame(node_variances)
    nv_counts = pd.DataFrame(node_counts)
    nv_dataframe.to_csv("output/node_variances.csv")
    nv_counts.to_csv("output/node_variances_counts.csv")

if __name__ == "__main__":
    compute()

