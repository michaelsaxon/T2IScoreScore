import click
from .run.evaluate_metric import evaluate
from .run.compute_metrics import main as compute_metrics

@click.group()
def main():
    """T2IScoreScore CLI tool."""
    pass

main.add_command(evaluate)
main.add_command(compute_metrics, name='compute')

if __name__ == '__main__':
    main() 