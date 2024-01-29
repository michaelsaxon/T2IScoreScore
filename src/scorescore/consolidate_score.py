from utils.read_csv_utils import load_kvp_multiple_files, load_kvp_single_file, load_col_source_csv
from scoring import tree_correlation_score, spearman_corr
import pandas as pd

def load_data():
    output_dict = load_kvp_multiple_files("../output/HalluVision_scores.csv")

    output_dict["clipscore"] = load_kvp_single_file("../output/clipscore.csv")
    output_dict["blipscore"] = load_kvp_single_file("../output/blipscore.csv")
    output_dict["alignscore"] = load_kvp_single_file("../output/alignscore.csv")
    output_dict["dsg_mplug1"] = load_kvp_single_file("../output/mplug1_dsg.csv", score_idx=1, do_fname_slug=True)
    output_dict["tifa_mplug1"] = load_kvp_single_file("../output/mplug1_tifa.csv", score_idx=1, do_fname_slug=True)
    output_dict["rank"] = load_col_source_csv("../data/metadata.csv", col_idx=4)
    output_dict["id"] = load_col_source_csv("../data/metadata.csv", col_idx=0)

    output_df = pd.DataFrame(output_dict)

    # Save the DataFrame to a CSV file
    output_df.to_csv("output_dataframe.csv", index=False)
    print(output_dict["rank"])

    return output_dict

def main():
    output_dict = load_data()
    id_range = list(range(164))

    metrics = ['dsg_fuyu', 'dsg_llava', 'dsg_mplug', 'tifa_fuyu', 'tifa_llava', 'tifa_mplug', 'clipscore', 'blipscore', 'alignscore', 'dsg_mplug1', 'tifa_mplug1']

    #print(pd.DataFrame(output_dict))

    print("running tree corr for all samples")
    #tree_spearman_avg, tree_counts = tree_correlation_score(pd.DataFrame(output_dict), metrics, [20], spearman_corr, scaled_avg=True)

    # output_dataframe = pd.DataFrame(tree_spearman_avg)
    # output_treecounts = pd.DataFrame(tree_counts)

    # output_dataframe.to_csv("spearman_corrs_weighted.csv")
    # output_treecounts.to_csv("spearman_counts_weighted.csv")

if __name__ == "__main__":
    main()
























