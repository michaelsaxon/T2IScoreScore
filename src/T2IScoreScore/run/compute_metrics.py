import click
from pathlib import Path
import pandas as pd
import logging
from typing import List, Optional

from T2IScoreScore.evaluators import (
    SpearmanEvaluator, 
    KSTestEvaluator, 
    DeltaEvaluator,
    KendallEvaluator,
    AVAILABLE_EVALUATORS
)

logger = logging.getLogger(__name__)

# HACK we hardcode the partition split points for this version of TS2
def get_partition(id: int) -> str:
    """Get partition name based on rank."""
    if id < 111:
        return 'synth'
    elif id < 136:
        return 'nat'
    else:
        return 'real'

def compute_metrics(
    scores_df: pd.DataFrame,
    evaluators: List[str],
    output_dir: Path,
    metric_name: str
) -> None:
    """Compute metametrics for a given scores dataframe."""
    
    # Add partition information
    scores_df['image_source'] = scores_df['id'].apply(get_partition)
    
    # Create output directory for this metric
    metric_dir = output_dir / 'metametrics'
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluators
    active_evaluators = []
    for eval_name in evaluators:
        if eval_name not in AVAILABLE_EVALUATORS:
            logger.warning(f"Unknown evaluator: {eval_name}, skipping")
            continue
        if eval_name == 'spearman':
            evaluator = AVAILABLE_EVALUATORS[eval_name](scaled_avg=True, invert_scores=True)
        else:
            evaluator = AVAILABLE_EVALUATORS[eval_name]()
        active_evaluators.append((eval_name, evaluator))
    
    # Compute metrics for each ID and save results
    metric_results = {} 

    logger.info(f"Processing {metric_name}")

    for eval_name, evaluator in active_evaluators:
        metric_results[eval_name] = evaluator.process_dataframe(scores_df, score_column="score", node_id_column='rank')
    
    full_results  =  []
    for graph_id in metric_results[list(metric_results.keys())[0]]:
        full_results.append({
            'id': graph_id,
            'image_source': scores_df[scores_df['id'] == graph_id]['image_source'].iloc[0],
            **{eval_name: metric_results[eval_name][graph_id][0] for eval_name in evaluators},
            'count': metric_results['spearman'][graph_id][1] if 'spearman' in evaluators else 1
        })
    
    # Save individual results
    results_df = pd.DataFrame(full_results)
    results_df.to_csv(metric_dir / f"{metric_name}.csv", index=False)
    
    # Compute and save partition averages
    partition_results = []
    
    # Overall average
    overall_avg = {
        'partition': 'overall',
        'count': len(results_df)
    }
    for eval_name in evaluators:
        if eval_name in results_df.columns:
            print(results_df)
            overall_avg[eval_name] = results_df[eval_name].mean()
    partition_results.append(overall_avg)
    
    # Partition averages
    for partition in ['synth', 'nat', 'real']:
        partition_df = results_df[results_df['image_source'] == partition]
        avg = {
            'partition': partition,
            'count': len(partition_df)
        }
        for eval_name in evaluators:
            if eval_name in results_df.columns:
                avg[eval_name] = partition_df[eval_name].mean()
        partition_results.append(avg)
    
    # Save partition averages
    partition_df = pd.DataFrame(partition_results)
    partition_df.to_csv(metric_dir / f"{metric_name}_averages.csv", index=False)
    
    logger.info(f"Results saved to {metric_dir}")

@click.command()
@click.argument('metric_name', type=str)
@click.argument('evaluators', nargs=-1, type=str)
@click.option(
    '--scores-file', '-s',
    type=click.Path(exists=True, path_type=Path),
    help='Optional path to existing scores CSV. If not provided, will load from default location'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    default='output/metametrics',
    help='Output directory for results'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose debug logging'
)
def main(
    metric_name: str,
    evaluators: tuple[str],
    scores_file: Optional[Path],
    output: Path,
    verbose: bool
):
    """Compute metametrics for a T2IMetrics metric."""
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set default evaluators if none provided
    if not evaluators:
        evaluators = ['spearman', 'kstest', 'delta']
    
    # Create output directory
    output.mkdir(parents=True, exist_ok=True)
    
    # Load scores
    if scores_file:
        logger.info(f"Loading scores from {scores_file}")
        scores_df = pd.read_csv(scores_file)
    else:
        # Load scores from default location
        scores_path = Path('output/scores') / f"{metric_name}_scores.csv"
        if not scores_path.exists():
            raise ValueError(f"No scores file found at {scores_path}. Please run evaluation first.")
        logger.info(f"Loading scores from {scores_path}")
        scores_df = pd.read_csv(scores_path)
    
    # Compute metrics
    compute_metrics(scores_df, evaluators, output, metric_name)

if __name__ == '__main__':
    main() 