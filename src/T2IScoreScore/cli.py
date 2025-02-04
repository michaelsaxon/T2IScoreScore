import click
from .run.evaluate_metric import evaluate

@click.group()
def main():
    """T2IScoreScore CLI tool."""
    pass

main.add_command(evaluate)

if __name__ == '__main__':
    main() 