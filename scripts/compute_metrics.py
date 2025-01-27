import pandas as pd
from scorescore.evaluators import SpearmanEvaluator, KSTestEvaluator, DeltaEvaluator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read the data with specified dtypes
scores_df = pd.read_csv("output/scores_final_all_2 copy.csv")
metadata_df = pd.read_csv("data/metadata.csv", dtype={'rank': str})

# Since files are aligned, we can just add the rank column directly
scores_df['rank'] = metadata_df['rank']

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

# List of model columns to evaluate (excluding metadata columns)
model_columns = [col for col in scores_df.columns 
                if col not in ['id', 'image_id', 'rank']]

# Initialize evaluators
spearman = SpearmanEvaluator(scaled_avg=True, invert_scores=True)
ks_test = KSTestEvaluator()
delta = DeltaEvaluator()

# Store results
results = []

# Process each model
for model in model_columns:
    logger.info(f"Processing {model}")
    
    # Get scores for each metric
    spearman_results = spearman.process_dataframe(scores_df, score_column=model, node_id_column='rank')
    ks_results = ks_test.process_dataframe(scores_df, score_column=model, node_id_column='rank')
    delta_results = delta.process_dataframe(scores_df, score_column=model, node_id_column='rank')
    
    # Add results for each graph ID
    for graph_id in spearman_results:
        results.append({
            'id': graph_id,
            'model': model,
            'spearman': spearman_results[graph_id][0],
            'spearman_walks': spearman_results[graph_id][1],
            'ks_test': ks_results[graph_id][0],
            'ks_pairs': ks_results[graph_id][1],
            'delta': delta_results[graph_id][0],
            'delta_pairs': delta_results[graph_id][1]
        })

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Add partition column
results_df['partition'] = results_df['id'].apply(get_partition)

# Reshape results_df to have model-specific metric columns
reshaped_results = []
for graph_id in results_df['id'].unique():
    graph_data = results_df[results_df['id'] == graph_id]
    row = {'id': graph_id}
    
    # Add num_walks (same for all models)
    row['num_walks'] = graph_data['spearman_walks'].iloc[0]
    
    # Add partition
    row['partition'] = graph_data['partition'].iloc[0]
    
    # Add model-specific metrics
    for _, result in graph_data.iterrows():
        model = result['model']
        row[f'{model}_spearman'] = result['spearman']
        row[f'{model}_ks_test'] = result['ks_test']
        row[f'{model}_delta'] = result['delta']
    
    reshaped_results.append(row)

# Convert reshaped results to dataframe
reshaped_df = pd.DataFrame(reshaped_results)

# Calculate average scores by model and partition
model_summaries = []
for model in model_columns:
    model_data = results_df[results_df['model'] == model]
    summary = {
        'model': model
    }
    
    # Calculate averages for each partition
    for partition in ['synth', 'nat', 'real']:
        partition_data = model_data[model_data['partition'] == partition]
        for metric in ['spearman', 'ks_test', 'delta']:
            summary[f'{partition}_{metric}'] = partition_data[metric].mean()
    
    # Calculate overall averages
    for metric in ['spearman', 'ks_test', 'delta']:
        summary[f'overall_{metric}'] = model_data[metric].mean()
    
    model_summaries.append(summary)

# Convert summaries to dataframe and display
summary_df = pd.DataFrame(model_summaries)
print("\nResults summary by model and partition:")
print(summary_df)

# Save reshaped results
reshaped_df.to_csv("output/metric_results.csv", index=False)

# Save summary
summary_df.to_csv("output/metric_summary.csv", index=False) 