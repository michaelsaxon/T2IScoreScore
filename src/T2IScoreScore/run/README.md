# Run Scripts
This package contains the main entry point scripts for running T2IScoreScore evaluations.

## Available Scripts

### evaluate_metric.py
Evaluates a T2IMetrics metric on the T2IScoreScore dataset.
Usage:
```bash
python evaluate_metric.py --metric_name <metric_name> --output_dir <output_dir>
```

You can directly run this using the `ts2` command:
```bash
ts2 evaluate <metric_name> [options]
```

### compute_metrics.py
Computes all available metrics on the T2IScoreScore dataset.
Usage:
```bash
python compute_metrics.py --output_dir <output_dir>
```

You can directly run this using the `ts2` command:
```bash
ts2 compute [options]
```

## Adding New Scripts

When adding new run scripts:
1. Create new script in this directory
2. Add Click command interface
3. Register in cli.py using main.add_command()
4. Update this README with usage instructions
