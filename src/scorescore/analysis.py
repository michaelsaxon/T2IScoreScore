import pandas as pd
from tqdm import tqdm
from scoring import analysis_tree_score, spearman_corr

if __name__ == "__main__":

    id_range = list(range(164))
    metrics = ['alignscore','blipscore', 'clipscore','fuyu_dsg' ,
            'fuyu_tifa', 'instruct_blip_dsg', 'instruct_blip_tifa',
                'llava-alt_dsg','llava-alt_tifa','llava_dsg',
                'llava_tifa','mplug_dsg', 'mplug_tifa']


    tree_spearman_avg, tree_counts = analysis_tree_score("src/output/integrated_image_scores.csv",
                        metrics,
                        id_range,
                        spearman_corr,
                        node_id_col="rank",
                        scaled_avg=False)

    output_dataframe = pd.DataFrame(tree_spearman_avg)
    output_treecounts = pd.DataFrame(tree_counts)

    output_dataframe.to_csv("src/output/spearman_corrs_weighted.csv")
    output_treecounts.to_csv("src/output/spearman_counts_weighted.csv")