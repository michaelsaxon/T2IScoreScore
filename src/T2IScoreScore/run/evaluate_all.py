import click
from pathlib import Path
import subprocess
import json
from typing import List

@click.command()
@click.option(
    '--metrics',
    type=click.Path(exists=True, path_type=Path),
    help='JSON file listing metrics to evaluate',
    required=True
)
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
def main(metrics: Path, device: str, output: Path):
    """Run evaluation for multiple metrics."""
    
    # Load metrics configuration
    with open(metrics) as f:
        metric_configs = json.load(f)
    
    # Run each metric
    for metric_name, config in metric_configs.items():
        config_file = output / f"{metric_name}_config.json"
        
        # Save metric config
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        # Run evaluation
        cmd = [
            "python", "-m", "T2IScoreScore.run.evaluate_metric",
            metric_name,
            "--device", device,
            "--output", str(output),
            "--metric-config", str(config_file)
        ]
        
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main() 