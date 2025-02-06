import logging
from pathlib import Path
from typing import Optional
import click
from datasets import load_dataset
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict

from T2IMetrics import AVAILABLE_METRICS

logger = logging.getLogger(__name__)

class MetricEvaluator:
    def __init__(self, 
                 metric_name: str,
                 device: str = 'cpu',
                 cache_dir: Optional[Path] = None,
                 **metric_kwargs):
        """Initialize metric evaluator.
        
        Args:
            metric_name: Name of metric from T2IMetrics to evaluate
            device: Device to run on (cuda/cpu)
            cache_dir: Directory to cache results
            **metric_kwargs: Additional arguments passed to metric constructor
        """
        if metric_name not in AVAILABLE_METRICS:
            valid_metrics = "\n".join(f"- {name}" for name in sorted(AVAILABLE_METRICS.keys()))
            raise ValueError(
                f"Unknown metric: {metric_name}\n"
                f"Available metrics:\n{valid_metrics}"
            )
            
        logger.info(f"Initializing {metric_name}")
        self.metric = AVAILABLE_METRICS[metric_name](
            device=device,
            cache_dir=cache_dir,
            **metric_kwargs
        )
        
    def evaluate_dataset(self, dataset) -> pd.DataFrame:
        """Evaluate metric on dataset.
        
        Args:
            dataset: HuggingFace dataset with 'image', 'id', 'target_prompt' fields
            
        Returns:
            DataFrame with columns: id, image_id, rank, score
        """
        results = []
        
        # Group by id to track image_id within each set
        id_counters = defaultdict(int)
        has_errors = False
        
        for item in tqdm(dataset, desc="Evaluating images"):
            # Get image_id (count within this id's set)
            curr_id = item['id']
            image_id = id_counters[curr_id]
            id_counters[curr_id] += 1
            
            # Calculate score
            try:
                score = self.metric.calculate_score(item['image'], item['target_prompt'])
                result = {
                    'id': curr_id,
                    'image_id': image_id,
                    'rank': item.get('rank', None),
                    'score': score
                }
            except Exception as e:
                has_errors = True
                logger.error(f"Error processing id {curr_id}, image_id {image_id}: {str(e)}")
                result = {
                    'id': curr_id,
                    'image_id': image_id,
                    'rank': item.get('rank', None),
                    'score': None,
                    'status': f'error: {str(e)}'
                }
            
            results.append(result)
            
        df = pd.DataFrame(results)
        
        # Only keep status column if there were errors
        if not has_errors and 'status' in df.columns:
            df = df.drop(columns=['status'])
            
        return df

@click.command()
@click.argument('metric_name', type=str)
@click.option(
    '--device', '-d',
    type=str,
    default='cpu',
    help='Device to run on (cuda/cpu)'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    default='output/scores',
    help='Output directory for results'
)
@click.option(
    '--cache/--no-cache',
    default=True,
    help='Enable/disable caching'
)
@click.option(
    '--cache-dir', '-c',
    type=click.Path(path_type=Path),
    default='output/cache',
    help='Cache directory'
)
@click.option(
    '--metric-config',
    type=click.Path(exists=True, path_type=Path),
    help='JSON file with additional metric configuration'
)
@click.option(
    '--kwargs', '-k',
    multiple=True,
    help='Additional keyword arguments for metric (format: key=value)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose debug logging'
)
def evaluate(metric_name: str,
            device: str,
            output: Path,
            cache: bool,
            cache_dir: Path,
            metric_config: Optional[Path],
            kwargs: tuple[str],
            verbose: bool):
    """Evaluate a T2IMetrics metric on the T2IScoreScore dataset."""
    
    # Setup logging level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse kwargs from command line
    metric_kwargs = {}
    for kv in kwargs:
        try:
            key, value = kv.split('=')
            metric_kwargs[key.strip()] = value.strip()
        except ValueError:
            logger.warning(f"Skipping invalid kwarg format: {kv}")
    
    # Load additional config if provided
    if metric_config:
        with open(metric_config) as f:
            metric_kwargs.update(json.load(f))
    
    logger.info(f"Using metric kwargs: {metric_kwargs}")
    
    # Setup cache
    cache_dir_path = cache_dir if cache else None
    if cache_dir_path:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create output directory
    output.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading T2IScoreScore dataset")
    dataset = load_dataset('saxon/T2IScoreScore')['train']
    
    # Initialize evaluator
    evaluator = MetricEvaluator(
        metric_name=metric_name,
        device=device,
        cache_dir=cache_dir_path,
        **metric_kwargs
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(dataset)
    
    # Save results
    output_file = output / f"{metric_name}_scores.csv"
    results.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

if __name__ == '__main__':
    evaluate() 