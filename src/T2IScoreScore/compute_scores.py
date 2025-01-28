import pandas as pd

import os

import pandas as pd
import re

import os
import pandas as pd

'''Merge all score files into a single combined file.
'''
def clean_and_merge(folder_path='output/scores_per_image', score_file_suffix='_score.csv', metadata_file='data/metadata.csv', output_file='output/scores_final_all.csv'):
    combined_df = pd.DataFrame(columns=['id', 'image_id'])

    for subdir in ['DSG', 'TIFA']:
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            score_files = [filename for filename in os.listdir(subdir_path) if filename.endswith(score_file_suffix)]
            score_files.sort()

            for filename in score_files:
                score_file_path = os.path.join(subdir_path, filename)
                model_name = os.path.splitext(filename)[0]
                model_name = model_name.replace('_score', '')
                current_df = pd.read_csv(score_file_path)
                current_df.rename(columns={'score': model_name}, inplace=True)

                combined_df = pd.merge(combined_df, current_df[['id', 'image_id', model_name]], on=['id', 'image_id'], how='outer')

    original_score_files = [filename for filename in os.listdir(folder_path) if filename.endswith(score_file_suffix) and filename not in combined_df.columns]
    original_score_files.sort()

    for filename in original_score_files:
        score_file_path = os.path.join(folder_path, filename)
        model_name = os.path.splitext(filename)[0]
        model_name = model_name.replace('_score', '')
        current_df = pd.read_csv(score_file_path)
        current_df.rename(columns={'score': model_name}, inplace=True)

        combined_df = pd.merge(combined_df, current_df[['id', 'image_id', model_name]], on=['id', 'image_id'], how='outer')

    combined_df['image_id'] = combined_df['image_id'].apply(lambda x: '0' if x.startswith('000') else re.search(r'\.(\d+)\.', x).group(1).lstrip('0') if re.search(r'\.(\d+)\.', x) else '')
    combined_df.to_csv(output_file, index=False)

# ,erged in utils

from scoring import *

"""
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
"""

# cleans a string containing some node number like "2a" to just the int part "2"
def clean_int_string(instr):
    for char in instr:
        if char not in "0123456789":
            instr = instr.replace(char,"")
    return instr

dataframe = pd.read_csv("output/scores_final_all_2 copy.csv")

ranks = pd.read_csv("data/metadata.csv")

dataframe['rank'] = pd.Series(list(ranks["rank"]), index=dataframe.index)

id_range = list(range(max(dataframe["id"])))
#id_range = [137, 139, 142, 144]
print(max(id_range))

from scoring import *

metrics = list(set(dataframe.columns) - set(["id","image_id","rank"]))

#tree_correlation_score(dataframe, "clipscore", [0,1,2,3,4,5,6,7,8,9,10], within_delta)
#print(tree_correlation_score(dataframe, metrics, [0,1,2,3,4,5,6,7,8,9,10], spearman_corr))

print(dataframe)


print("running tree corr for all samples")

# 

#tree_spearman_avg, tree_counts = tree_correlation_score(dataframe, metrics, id_range, spearman_corr, scaled_avg=True)
#node_variances, node_counts = within_node_score(dataframe, metrics, id_range, variance)

avg_ks_stat, ks_counts = analysis_tree_score(dataframe, ["blip1_dsg"], [0], between_nodepair_ks_score, scaled_avg=True)

print(avg_ks_stat)

#print(tree_spearman_avg)

#print(tree_spearman_avg)

#output_dataframe = pd.DataFrame(tree_spearman_avg)
#output_treecounts = pd.DataFrame(tree_counts)
#nv_dataframe = pd.DataFrame(node_variances)
#nv_counts = pd.DataFrame(node_counts)

output_dataframe = pd.DataFrame(avg_ks_stat)

output_dataframe.to_csv("kstest_average_weighted.csv")
#output_treecounts.to_csv("spearman_counts_weighted.csv")
#nv_dataframe.to_csv("node_variances.csv")
#nv_counts.to_csv("node_variances_counts.csv")

























