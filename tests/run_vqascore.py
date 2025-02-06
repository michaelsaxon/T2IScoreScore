#!/usr/bin/env python3

import logging
from pathlib import Path
from PIL import Image
import click

from T2IMetrics.likelihood.vqascore import VQAScore

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
    '--vlm-type',
    type=str,
    default='smolvlm',
    help='Type of VLM to use for scoring'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable debug logging'
)
@click.option(
    '--cache/--no-cache',
    default=True,
    help='Enable/disable caching of results'
)
@click.option(
    '--cache-dir', '-c',
    type=click.Path(path_type=Path),
    default='output/cache/run_vqascore',
    help='Directory to store cache (only used if caching is enabled)'
)
def main(prompt: str, image: Path, device: str, vlm_type: str, verbose: bool, cache: bool, cache_dir: Path):
    """
    Calculate VQAScore for an image-prompt pair.
    
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
        logging.getLogger('T2IMetrics.likelihood.vqascore').setLevel(logging.DEBUG)
        logging.getLogger('T2IMetrics.models.vlm').setLevel(logging.DEBUG)
    
    # Handle cache settings
    cache_dir_path = None
    if cache:
        cache_dir_path = cache_dir
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Caching enabled. Using directory: {cache_dir_path}")
    else:
        logger.debug("Caching disabled")
    
    # Initialize VQAScore
    metric = VQAScore(
        vlm_type=vlm_type,
        device=device,
        cache_dir=cache_dir_path
    )
    
    # Load image
    img = Image.open(image)
    
    # Calculate score
    score = metric.calculate_score(img, prompt)
    print(f"VQA Score: {score:.4f}")

if __name__ == '__main__':
    main() 