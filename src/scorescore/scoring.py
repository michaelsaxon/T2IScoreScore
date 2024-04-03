from tqdm import tqdm
import itertools

import pandas as pd
import numpy as np
import math

from scipy.stats import spearmanr, ks_2samp

# produce an output dict indexed by id then node of the function eval'd over each node
def within_node_score(dataframe, metric_col_idces, id_range, score_function, node_id_idx = "rank"):
    output_dict = {metric_col_idx : {} for metric_col_idx in metric_col_idces}
    counts_dict = {metric_col_idx : {} for metric_col_idx in metric_col_idces}
    for metric_col_idx in metric_col_idces:
        for id_idx in id_range:
            id_df = dataframe.loc[dataframe["id"] == str(id_idx)]
            node_set = list(id_df["rank"].unique())
            #output_dict[metric_col_idx][id_idx] = {}
            # means over this stuff (see node_variances.csv for how fucked it currently is)
            node_level_vars = []
            for node in node_set:
                vals = list(map(lambda x: robust_float_cast(x), list(id_df.loc[id_df[node_id_idx] == node][metric_col_idx])))
                node_level_vars.append(score_function(vals))
            node_level_vars = [var for var in node_level_vars if not math.isnan(var)]
            if len(node_level_vars) == 0:
                output_dict[metric_col_idx][id_idx] = float('nan')
            else:
                output_dict[metric_col_idx][id_idx] = sum(node_level_vars) / len(node_level_vars)
            counts_dict[metric_col_idx][id_idx] = len(node_level_vars)
    return output_dict, counts_dict

def variance(inlist):
    if len(inlist) < 2:
        return float('nan')
    else:
        return np.var(np.array(inlist))

def within_delta(inlist):
    if len(inlist) == 1:
        return -1
    return max(inlist) - min(inlist)

def extract_int_string(nodeid: str):
    outstr = ""
    for character in nodeid:
        if character in "0123456789":
            outstr += character
        else:
            break
    return int(outstr)

def repair_missing_rank(id_df):
    no_rank = list(id_df.loc[id_df["rank"] == ''].index)
    has_rank = id_df.loc[id_df["rank"] != '']
    for missing_idx in no_rank:
        prompt_start = missing_idx.split(".")[0]
        neighbors = has_rank.loc[has_rank.index.str.contains(prompt_start)]
        if len(neighbors.index) >= 1:
            replacement = list(neighbors["rank"])[0]
        else:
            print("FAIL")
            replacement = "nan"
        id_df["rank"][missing_idx] = replacement
    return id_df.loc[id_df["rank"] != "nan"]

def robust_float_cast(instr):
    instr = str(instr)
    instr = instr.replace('"','')
    return float(instr)

### Check if we need to reorder using second variable
# get each possible walk and run the correlations (for now might be off)
# return something just indexed by range
def analysis_tree_score(dataframe, metric_col_idces, id_range, score_function, node_id_col="rank", scaled_avg=False, debug = True):
    printif = print if debug else lambda *x: None
    output_dict = {metric_col_idx : {} for metric_col_idx in metric_col_idces}
    val_counts = {metric_col_idx : {} for metric_col_idx in metric_col_idces}

    for id_idx in tqdm(id_range):
        id_df = dataframe.loc[dataframe["id"] == id_idx]
        node_set = list(id_df[node_id_col].unique())
        try:
            node_numbers = node_set
            node_numbers_sorted = list(set(node_numbers))
            node_numbers_sorted.sort()
        except ValueError:
            print(id_df)
            input()
        # probably not ideal, but just build every possible alignment to start
        # aka, dfs on sets of 0, 1, 2, ... to max(node_numbers)
        walks = list(map(lambda x: [x], [i for i, x in enumerate(node_numbers) if x == 0]))
        #for level in range(1,max(node_numbers) + 1):
        for level in node_numbers_sorted:
            new_walks = []
            for parent_walk in walks:
                # affix all the children to each parent (and duplicate)
                for child_node_idx in [i for i, x in enumerate(node_numbers) if x == level]:
                    copied = list(parent_walk)
                    copied.append(child_node_idx)
                    new_walks.append(copied)
            walks = new_walks
        walks_ids = list(map(lambda x: list(map(lambda y: node_set[y], x)), walks))
        walks_x = list(map(lambda x: list(map(lambda y: node_numbers[y], x)), walks))

        for metric_col_idx in metric_col_idces:
            walk_scores = []
            walk_score_counts = []
            for i in range(len(walks_ids)):
                x_vals = []
                walk_ids = walks_ids[i]
                walk_xs = walks_x[i]
                # # # # ITERATE THROUGH AND GENERATE THE X Y, RUN SPEARMAN CORR
                walk_x_array = []
                walk_y_array = []
                for j in range(len(walk_ids)):
                    this_step_ys = list(map(
                        lambda x: robust_float_cast(x),
                        id_df.loc[id_df["rank"] == walk_ids[j]][metric_col_idx]
                    ))
                    # probably need logic for NaNs
                    walk_x_array += [walk_xs[j]] * len(this_step_ys)
                    walk_y_array += this_step_ys
                # Logic for computing the x vs y score would go here
                # important we drop the x for y nans before y for y nans
                walk_x_array = [x for i, x in enumerate(walk_x_array) if not math.isnan(walk_y_array[i])]
                walk_y_array = [y for y in walk_y_array if not math.isnan(y)]
                print(walk_x_array,walk_y_array )
                walk_scores.append(score_function(walk_x_array, walk_y_array))
                printif(walk_scores)
                walk_score_counts.append(len(walk_x_array))
            output_vals = []
            output_counts = []
            for i, score in enumerate(walk_scores):
                if not math.isnan(score):
                    output_vals.append(score)
                    output_counts.append(walk_score_counts[i])
            if len(output_vals) == 0:
                output_dict[metric_col_idx][id_idx] = float('nan')
                val_counts[metric_col_idx][id_idx] = float('nan')
            else:
                if scaled_avg:
                    output_dict_val = sum(map(lambda i: output_vals[i] * output_counts[i], range(len(output_vals)))) / sum(output_counts)
                else:
                    output_dict_val = sum(output_vals) / len(output_vals)
                print(output_vals)
                output_dict[metric_col_idx][id_idx] = output_dict_val
                val_counts[metric_col_idx][id_idx] = len(output_vals)
        
    return output_dict, val_counts



# we will implement this for two sample KS statistic
# https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test
def between_nodepair_ks_score(x_list, y_list):
    # x defines the node for each sample in the y list.
    # separate y into one list for each node, take avg of the test statistic between each pair
    y_lists = [np.array([
        y_list[i] for i in range(len(x_list)) if x_list[i] == node
        ]) for node in list(set(x_list))]
    pair_scores = []
    for pair in itertools.combinations(y_lists, 2):
        pair_scores.append(ks_2samp(pair[0], pair[1]).statistic)
    if len(pair_scores) == 0:
        return 0
    if math.isnan(sum(pair_scores)):
        return 0
    return sum(pair_scores) / len(pair_scores)
# also consider https://en.wikipedia.org/wiki/Cucconi_test, https://en.wikipedia.org/wiki/Lepage_test, examples of https://en.wikipedia.org/wiki/Behrens%E2%80%93Fisher_problem

def spearman_corr(x_list, y_list):
    if len(x_list) <= 1:
        return float('nan')
    out = spearmanr(np.array(x_list), np.array(y_list)).correlation
    # if one of the input sequences is all the same value, spearmanr returns nan.
    # A rank correlation value of 0 is meant to denote "no relationship", so we should return 0 in this case
    if math.isnan(out):
        out = 0
    return out
