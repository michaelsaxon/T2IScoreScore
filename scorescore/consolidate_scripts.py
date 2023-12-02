from read_csv_utils import *

output_dict = load_kvp_multiple_files("HalluVision_scores.csv")

print(output_dict.keys())

output_dict["clipscore"] = load_kvp_single_file("output_csvs/clipscore.csv")
output_dict["blipscore"] = load_kvp_single_file("output_csvs/blipscore.csv")
output_dict["alignscore"] = load_kvp_single_file("output_csvs/alignscore.csv")
output_dict["dsg_mplug1"] = load_kvp_single_file("mplug1_dsg.csv",score_idx=1,do_fname_slug=True)
output_dict["tifa_mplug1"] = load_kvp_single_file("mplug1_tifa.csv",score_idx=1,do_fname_slug=True)
output_dict["rank"] = load_col_source_csv("HalluVisionAll.csv", 5)
output_dict["id"] = load_col_source_csv("HalluVisionAll.csv", 0)

print(output_dict.keys())

get_row(output_dict, "235_a tree with yellow leaves on a snow covered mountain, night view_0")
get_row(output_dict, "5_A woman on a scooter._0")

dataframe = pd.DataFrame(output_dict)
print(dataframe)

# dataframe.to_csv("test.csv")

id_range = list(range(259))
print(max(id_range))

from scoring import *
import pandas as pd

metrics = ['dsg_fuyu', 'dsg_llava', 'dsg_mplug', 'tifa_fuyu', 'tifa_llava', 'tifa_mplug', 'clipscore', 'blipscore', 'alignscore', 'dsg_mplug1', 'tifa_mplug1']

#tree_correlation_score(dataframe, "clipscore", [0,1,2,3,4,5,6,7,8,9,10], within_delta)
#print(tree_correlation_score(dataframe, metrics, [0,1,2,3,4,5,6,7,8,9,10], spearman_corr))


print("running tree corr for all samples")
tree_spearman_avg, tree_counts = tree_correlation_score(dataframe, metrics, id_range, spearman_corr, scaled_avg=True)
#node_variances, node_counts = within_node_score(dataframe, metrics, id_range, variance)

#print(tree_spearman_avg)

output_dataframe = pd.DataFrame(tree_spearman_avg)
output_treecounts = pd.DataFrame(tree_counts)
#nv_dataframe = pd.DataFrame(node_variances)
#nv_counts = pd.DataFrame(node_counts)

output_dataframe.to_csv("spearman_corrs_weighted.csv")
output_treecounts.to_csv("spearman_counts_weighted.csv")
#nv_dataframe.to_csv("node_variances.csv")
#nv_counts.to_csv("node_variances_counts.csv")
























