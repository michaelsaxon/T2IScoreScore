import math

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

def robust_csv_line_split(line):
    line = line.strip()
    if '"' in line:
        # one or more cells contains commas, you have to split on the quotes first
        line = line.split('",')
        newline = []
        for elem in line:
            # if this is a quote line, we have already split it.
            # remove the other quote and put the whole thing on the list.
            # Else it's a block of comma separated values and should be split
            # this only works because none of my csvs lines end with strings
            if '"' in elem:
                assert elem.count('"') == 1
                if elem[0] == '"':
                    newline.append(elem[1:])
                else:
                    elem = elem.split(',"')
                    newline += elem
            else:
                newline += elem.split(",")
    else:
        newline = line.split(",")
    return newline

def slug_fname(fname):
    return ".".join(fname.split(".")[:-1])

def load_kvp_single_file(fname, convert_float = False, score_idx = 2, do_fname_slug = False):
    output_dict = {}
    with open(fname, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = robust_csv_line_split(line)
        fname_slug = line[0]
        if do_fname_slug:
            fname_slug = slug_fname(fname_slug)
        score = line[score_idx]
        if convert_float:
            score = float(line[score_idx])
        output_dict[fname_slug] = score
    return output_dict

def load_kvp_multiple_files(fname, convert_float = False):
    output_dict = {}
    with open(fname, "r") as f:
        lines = list(map(robust_csv_line_split, f.readlines()))
    keys = lines[0]
    print(keys)
    for metric_name in keys[1:]:
        output_dict[metric_name] = {}
    for line in lines[1:]:
        # remove extension
        fname_slug = slug_fname(line[0])
        for i in range(1,len(line)):
            val = line[i]
            if val != '' and convert_float:
                val = float(val)
            if i > 6:
                print(line)
            output_dict[keys[i]][fname_slug] = val
    return output_dict

# 3 is index for ranks
# 0? is index for ids
def load_col_source_csv(fname, col_idx = 3):
    output_dict = {}
    with open(fname, "r") as f:
        lines = list(map(robust_csv_line_split, f.readlines()))
    for line in lines:
        fname_slug = slug_fname(line[2])
        rank = line[col_idx]
        output_dict[fname_slug] = rank
    return output_dict

def get_row(output_dict, fname_slug, tty = True):
    keys = output_dict.keys()
    out_string = []
    if tty:
        print(fname_slug)
    for key in keys:
        value = output_dict[key].get(fname_slug, '')
        if tty:
            print(f"{key}: {value}")
        out_string.append(value)
    return ",".join(out_string)

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
                #output_dict[metric_col_idx][id_idx][node] = score_function(vals)
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
        print(prompt_start)
        if len(neighbors.index) >= 1:
            replacement = list(neighbors["rank"])[0]
        else:
            print("FAIL")
            print(neighbors)
            replacement = "nan"
        id_df["rank"][missing_idx] = replacement
    return id_df.loc[id_df["rank"] != "nan"]

def robust_float_cast(instr):
    try:
        return float(instr)
    except ValueError:
        if instr != '':
            print(f"RFCError: {instr}")
        return float('nan')

# get each possible walk and run the correlations (for now might be off)
# return something just indexed by range
def tree_correlation_score(dataframe, metric_col_idces, id_range, score_function, node_id_idx = "rank", scaled_avg = False):

    print(dataframe.head())

    print(f'metric_col_idces:{metric_col_idces}')

    output_dict = {metric_col_idx : {} for metric_col_idx in metric_col_idces}

    print(f'output_dict:{output_dict}')
    val_counts = {metric_col_idx : {} for metric_col_idx in metric_col_idces}
    for id_idx in tqdm(id_range):
        id_df = repair_missing_rank(dataframe.loc[dataframe["id"] == str(id_idx)])
        node_set = list(id_df["rank"].unique())
        print(node_set)
        try:
            node_numbers = list(map(extract_int_string, node_set))
            node_numbers_sorted = list(set(node_numbers))
            node_numbers_sorted.sort()
        except ValueError:
            print()
            print(id_df)
            input()
        # probably not ideal, but just build every possible alignment to start
        # aka, dfs on sets of 0, 1, 2, ... to max(node_numbers)
        walks = list(map(lambda x: [x], [i for i, x in enumerate(node_numbers) if x == 0]))
        print(walks)
        #for level in range(1,max(node_numbers) + 1):
        for level in node_numbers_sorted:
            #print(f"{id_idx}: {level}")
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
                    print(f'random-walk:{walk_ids}')
                    print("metric_col_idx:", metric_col_idx)
                    print("Columns in id_df:", id_df.columns)

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
                #print(walk_x_array)
                #print(walk_y_array)
                walk_scores.append(score_function(walk_x_array, walk_y_array))
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
                output_dict[metric_col_idx][id_idx] = output_dict_val
                val_counts[metric_col_idx][id_idx] = len(output_vals)
    return output_dict, val_counts

def spearman_corr(x_list, y_list):
    if len(x_list) <= 1:
        return float('nan')
    out = spearmanr(np.array(x_list), np.array(y_list))
    return out.correlation

def main():
    # Load key-value pairs from multiple files
    output_dict = load_kvp_multiple_files("HalluVision_scores.csv")
    print(output_dict.keys())

    # Load key-value pairs from single files and add them to the output_dict
    output_dict["clipscore"] = load_kvp_single_file("output_csvs/clipscore.csv")
    output_dict["blipscore"] = load_kvp_single_file("output_csvs/blipscore.csv")
    output_dict["alignscore"] = load_kvp_single_file("output_csvs/alignscore.csv")
    output_dict["dsg_mplug1"] = load_kvp_single_file("mplug1_dsg.csv", score_idx=1, do_fname_slug=True)
    output_dict["tifa_mplug1"] = load_kvp_single_file("mplug1_tifa.csv", score_idx=1, do_fname_slug=True)
    output_dict["rank"] = load_col_source_csv("HalluVisionAll.csv", 5)
    output_dict["id"] = load_col_source_csv("HalluVisionAll.csv", 0)
    #print(output_dict.keys)

    # Extract rows from the output_dict
    get_row(output_dict, "235_a tree with yellow leaves on a snow covered mountain, night view_0")
    #get_row(output_dict, "5_A woman on a scooter._0")

    # Create a DataFrame from the output_dict
    dataframe = pd.DataFrame(output_dict)
    print(dataframe.head())

    # Define the ID range
    id_range = list(range(259))
    #print(max(id_range))

    # Define the metrics
    metrics = ['dsg_fuyu', 'dsg_llava', 'dsg_mplug', 'tifa_fuyu', 'tifa_llava', 'tifa_mplug', 'clipscore', 'blipscore', 'alignscore', 'dsg_mplug1', 'tifa_mplug1']

    # Run tree_correlation_score for the "clipscore" metric
    # tree_correlation_score(dataframe, ["clipscore"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], spearman_corr, scaled_avg=True)

    # print("Running tree corr for all samples")

    # # Run tree_correlation_score for all metrics
    tree_spearman_avg, tree_counts = tree_correlation_score(dataframe, metrics, id_range, spearman_corr, scaled_avg=True)

    # # Create DataFrames from the results
    output_dataframe = pd.DataFrame(tree_spearman_avg)
    output_treecounts = pd.DataFrame(tree_counts)

    # # Save results to CSV files
    output_dataframe.to_csv("spearman_corrs_weighted.csv")
    output_treecounts.to_csv("spearman_counts_weighted.csv")

if __name__ == "__main__":
    main()

