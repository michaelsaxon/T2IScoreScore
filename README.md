# T2IScoreScore

T2IScoreScore is a framework for evaluating text-to-image model evaluation metrics. The framework provides tools to:

1. [Reference implementations](src/T2IMetrics/README.md) of various classes of text-to-image metrics with a consistent API:
   - Correlation-based metrics (CLIPScore)
   - Likelihood-based metrics
   - Visual Question-Answering metrics
   
2. [Run these metrics](src/T2IScoreScore/run/README.md) on the T2IScoreScore dataset of semantic error graphs:
   ```
   ts2 evaluate CLIPScore --device cuda
   ```

3. [Compute metametrics](src/T2IScoreScore/evaluators/README.md) to characterize how well the T2I metric ordered different nodes of images along walks of increasing error count:
   ```
   ts2 compute CLIPScore spearman kstest
   ```

4. [Generate visualizations](src/T2IScoreScore/figures/README.md) and reports to analyze metric performance across different error types and image sources.

## Installation

```bash
git clone https://github.com/michaelsaxon/T2IScoreScore.git
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
