import pandas as pd
from scorescore.evaluators import SpearmanEvaluator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read the data with specified dtypes
scores_df = pd.read_csv("output/scores_final_all_2 copy.csv")
metadata_df = pd.read_csv("data/metadata.csv", dtype={'rank': str})

# Find partition boundaries using image_source
nat_mask = metadata_df['image_source'] == 'Natural'
nat_ids = metadata_df[nat_mask]['id'].unique()
min_nat_id = nat_ids.min()
max_nat_id = nat_ids.max()

# Create partition labels
def get_partition(id_):
    if id_ < min_nat_id:
        return 'synth'
    elif id_ > max_nat_id:
        return 'real'
    else:
        return 'nat'

# Since files are aligned, we can just add the rank column directly
scores_df['rank'] = metadata_df['rank']

# List of model columns to evaluate (excluding metadata columns)
model_columns = [col for col in scores_df.columns 
                if col not in ['id', 'image_id', 'rank']]

# Initialize evaluator
spearman = SpearmanEvaluator(scaled_avg=True, invert_scores=True)

# Store results for each model
model_results = []

# Process each model
for model in model_columns:
    logger.info(f"Processing {model}")
    results_dict = spearman.process_dataframe(scores_df, score_column=model, node_id_column='rank')
    
    # Convert dictionary to dataframe
    model_df = pd.DataFrame([
        {
            'id': id_,
            f'{model}_spearman': score,
            f'{model}_count': count
        }
        for id_, (score, count) in results_dict.items()
    ])
    model_results.append(model_df)

# Merge all results on id
results_df = model_results[0]
for df in model_results[1:]:
    results_df = results_df.merge(df, on='id', how='outer')

# Add partition labels
results_df['partition'] = results_df['id'].apply(get_partition)

# Print summary statistics by partition
print("\nResults summary by partition:")
spearman_cols = [col for col in results_df.columns if col.endswith('_spearman')]
for partition in ['synth', 'nat', 'real']:
    print(f"\n{partition.upper()} partition:")
    partition_means = results_df[results_df['partition'] == partition][spearman_cols].mean()
    print(partition_means)

# Save results
results_df.to_csv("output/metric_results.csv", index=False) 