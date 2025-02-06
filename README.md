# T2IScoreScore

T2IScoreScore is a framework for evaluating text-to-image model evaluation metrics. It provides tools to:

1. Run metrics on a standardized dataset of image-text pairs
2. Compute metametrics that assess metric quality
3. Generate visualizations and analysis

## Installation

```bash
git clone https://github.com/saxon-milton/T2IScoreScore
cd T2IScoreScore
pip install -e .
```

## Package Structure

```
src/
├── T2IMetrics/          # Collection of text-to-image evaluation metrics
└── T2IScoreScore/       # Framework for evaluating metrics
    ├── evaluators/      # Metametric implementations (Spearman, KS test, etc.)
    ├── figures/         # Visualization utilities
    └── run/             # CLI entry points
```

## CLI Usage

The package provides two main commands through the `ts2` CLI:

### Evaluate a Metric

Run a metric on the T2IScoreScore dataset:

```bash
# Basic usage
ts2 evaluate CLIPScore

# With options
ts2 evaluate TIFAScore --device cuda --output results/
```

### Compute Metametrics

Analyze metric performance using metametrics:

```bash
# Run all default evaluators (spearman, kstest, delta)
ts2 compute CLIPScore

# Run specific evaluators
ts2 compute TIFAScore spearman kendall
```

## Output Structure

Results are saved in the following structure:

```
output/
├── scores/             # Raw metric scores
│   └── metric_scores.csv
└── metametrics/        # Metametric results
    ├── metric.csv      # Per-example results
    └── metric_averages.csv  # Partition averages
```

## Contributing

To add a new metric:
1. Create a new class in T2IMetrics inheriting from BaseMetric
2. Implement the calculate_score() method
3. Register in T2IMetrics/__init__.py

To add a new metametric:
1. Create a new class in T2IScoreScore/evaluators inheriting from MetricEvaluator
2. Implement required evaluation methods
3. Register in evaluators/__init__.py
