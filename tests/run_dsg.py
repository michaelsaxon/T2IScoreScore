#!/usr/bin/env python3

import logging
from pathlib import Path
from PIL import Image
import click

from T2IMetrics.qga.dsg import DSGMetric

logger = logging.getLogger(__name__)

@click.command()
@click.argument('prompt', type=str)
@click.option(
    '--image', '-i',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to image file'
)
@click.option(
    '--device', '-d',
    type=str,
    default='cpu',
    help='Device to run on (e.g., "cuda", "cpu")'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable debug logging to see all Q&A pairs'
)
@click.option(
    '--cache/--no-cache',
    default=True,
    help='Enable/disable caching of results'
)
@click.option(
    '--cache-dir', '-c',
    type=click.Path(path_type=Path),
    default='output/cache/run_dsg',
    help='Directory to store cache (only used if caching is enabled)'
)
def main(prompt: str, image: Path, device: str, verbose: bool, cache: bool, cache_dir: Path):
    """
    Calculate DSG score for an image-prompt pair.
    
    PROMPT is the text description to evaluate against the image.
    """
    if verbose:
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set specific loggers to DEBUG
        logging.getLogger('T2IMetrics').setLevel(logging.DEBUG)
        logging.getLogger('T2IMetrics.qga.dsg').setLevel(logging.DEBUG)
        logging.getLogger('T2IMetrics.models.lm').setLevel(logging.DEBUG)
        logging.getLogger('T2IMetrics.models.vlm').setLevel(logging.DEBUG)
    
    # Handle cache settings
    cache_dir_path = None
    if cache:
        cache_dir_path = cache_dir
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Caching enabled. Using directory: {cache_dir_path}")
    else:
        logger.debug("Caching disabled")
    
    # Initialize DSG with MLX models
    metric = DSGMetric(
        #lm_type='llama3-mlx',
        lm_type='openai',
        vlm_type='qwen2-mlx',
        device=device,
        cache_dir=cache_dir_path
    )
    
    # Load image
    img = Image.open(image)
    
    # Calculate score
    score = metric.calculate_score(img, prompt)
    print(f"DSG Score: {score:.4f}")

if __name__ == '__main__':
    main() 