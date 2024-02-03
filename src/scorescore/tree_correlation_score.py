import pandas as pd
from tqdm import tqdm
import math
from scoring import extract_int_string, robust_float_cast

def tree_correlation_score(file_path, metric_col_name, id_range, score_function, node_id_col="rank", scaled_avg=False):
    dataframe = pd.read_csv(file_path)

    output_dict = {metric_col_name: {}}
    val_counts = {metric_col_name: {}}

    for id_idx in tqdm(id_range):
        id_df = dataframe[dataframe["id"] == id_idx]
        node_set = list(id_df[node_id_col].unique())

        try:
            node_numbers = list(map(extract_int_string, node_set))
            node_numbers_sorted = sorted(set(node_numbers))
        except ValueError:
            print()
            print(id_df)
            input()

        walks = list(map(lambda x: [x], [i for i, x in enumerate(node_numbers) if x == 0]))

        for level in node_numbers_sorted:
            new_walks = []
            for parent_walk in walks:
                for child_node_idx in [i for i, x in enumerate(node_numbers) if x == level]:
                    copied = list(parent_walk)
                    copied.append(child_node_idx)
                    new_walks.append(copied)
            walks = new_walks

        walks_ids = list(map(lambda x: list(map(lambda y: node_set[y], x)), walks))
        walks_x = list(map(lambda x: list(map(lambda y: node_numbers[y], x)), walks))

        walk_scores = []
        walk_score_counts = []

        for i in range(len(walks_ids)):
            x_vals = []
            walk_ids = walks_ids[i]
            walk_xs = walks_x[i]

            walk_x_array = []
            walk_y_array = []

            for j in range(len(walk_ids)):
                this_step_ys = list(map(
                    lambda x: robust_float_cast(x),
                    id_df[id_df[node_id_col] == walk_ids[j]][metric_col_name]
                ))

                walk_x_array += [walk_xs[j]] * len(this_step_ys)
                walk_y_array += this_step_ys

            walk_x_array = [x for i, x in enumerate(walk_x_array) if not math.isnan(walk_y_array[i])]
            walk_y_array = [y for y in walk_y_array if not math.isnan(y)]

            walk_scores.append(score_function(walk_x_array, walk_y_array))
            walk_score_counts.append(len(walk_x_array))

        output_vals = []
        output_counts = []

        for i, score in enumerate(walk_scores):
            if not math.isnan(score):
                output_vals.append(score)
                output_counts.append(walk_score_counts[i])

        if len(output_vals) == 0:
            print("outputvals len 0")
            output_dict[metric_col_name][id_idx] = float('nan')
            val_counts[metric_col_name][id_idx] = float('nan')
        else:
            if scaled_avg:
                output_dict_val = sum(map(lambda i: output_vals[i] * output_counts[i], range(len(output_vals)))) / sum(output_counts)
            else:
                output_dict_val = sum(output_vals) / len(output_vals)

            output_dict[metric_col_name][id_idx] = output_dict_val
            val_counts[metric_col_name][id_idx] = len(output_vals)

    return output_dict, val_counts
