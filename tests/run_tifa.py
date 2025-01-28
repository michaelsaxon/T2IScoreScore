import click
import logging
import os
from pathlib import Path
from PIL import Image
from T2IMetrics.qa.tifa import TIFAMetric

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'  # More detailed format
)

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
    '--openai-key', '-k',
    type=str,
    envvar='OPENAI_API_KEY',
    help='OpenAI API key. Can also be set via OPENAI_API_KEY environment variable'
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
    default='output/cache/run_tifa',
    help='Directory to store cache (only used if caching is enabled)'
)
def main(prompt: str, image: Path, device: str, openai_key: str, verbose: bool, cache: bool, cache_dir: Path):
    """
    Calculate TIFA score for an image-prompt pair.
    
    PROMPT is the text description to evaluate against the image.
    """
    if verbose:
        # Set root logger to DEBUG
        logging.getLogger().setLevel(logging.DEBUG)
        # Ensure specific loggers are at DEBUG
        logging.getLogger('T2IMetrics').setLevel(logging.DEBUG)
        logging.getLogger('T2IMetrics.qa.tifa').setLevel(logging.DEBUG)
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
    
    # Initialize TIFA with Llama3-MLX
    metric = TIFAMetric(
        lm_type='llama3-mlx',
        vlm_type='qwen2-mlx',
        device=device,
        cache_dir=cache_dir_path
    )
    
    # Load image
    img = Image.open(image)
    
    # Calculate score
    score = metric.calculate_score(img, prompt)
    
    # Final score is already logged by TIFA

if __name__ == '__main__':
    main() 