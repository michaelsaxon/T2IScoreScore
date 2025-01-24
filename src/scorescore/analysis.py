import pandas as pd
from tqdm import tqdm
from scoring import analysis_tree_score, spearman_corr, between_nodepair_ks_score, kendall_tau, between_nodepair_avg_delta

if __name__ == "__main__":

    id_range = list(range(164))
    metrics = ['alignscore', 'clipscore','fuyu_dsg' ,
            'fuyu_tifa', 'instruct_blip_dsg', 'instruct_blip_tifa',
                'llava-alt_dsg','llava-alt_tifa','llava_dsg',
                'llava_tifa','mplug_dsg', 'mplug_tifa', 'blip1_dsg', 'blip1_tifa', 'gpt4v_tifa', 'gpt4v_dsg',
                'viescore', 'llmscore_over', 'llmscore_ec']


    #dataframe = pd.read_csv("output/scores_final_all_2.csv")
    dataframe = pd.read_csv("output/scores_final_all_3.csv")
    ranks = pd.read_csv("data/metadata.csv")

    dataframe['rank'] = pd.Series(list(ranks["rank"]), index=dataframe.index)

    '''
    tree_spearman_avg, tree_counts = analysis_tree_score(dataframe,
                        metrics,
                        id_range,
                        spearman_corr,
                        node_id_col="rank",
                        scaled_avg=True,
                        debug=False)

    output_dataframe = pd.DataFrame(tree_spearman_avg)
    output_treecounts = pd.DataFrame(tree_counts)

    output_dataframe.to_csv("src/output/spearman_corrs_weighted_3.csv")
    output_treecounts.to_csv("src/output/counts_weighted_3.csv")

    tree_ks_avg, _ = analysis_tree_score(dataframe,
                        metrics,
                        id_range,
                        between_nodepair_ks_score,
                        node_id_col="rank",
                        scaled_avg=True,
                        debug=False)

    output_dataframe = pd.DataFrame(tree_ks_avg)

    output_dataframe.to_csv("src/output/kstest_weighted_3.csv")

    tree_kendall_avg, _ = analysis_tree_score(dataframe,
                        metrics,
                        id_range,
                        kendall_tau,
                        node_id_col="rank",
                        scaled_avg=True,
                        debug=False)

    output_dataframe = pd.DataFrame(tree_kendall_avg)

    output_dataframe.to_csv("src/output/kendall_weighted_3.csv")

    '''
    
    tree_delta_avg, _ = analysis_tree_score(dataframe,
                        metrics,
                        id_range,
                        between_nodepair_avg_delta,
                        node_id_col="rank",
                        scaled_avg=True,
                        debug=False)

    output_dataframe = pd.DataFrame(tree_delta_avg)

    output_dataframe.to_csv("src/output/delta_weighted_3.csv")