import pandas as pd
from T2IMetrics.figures.correlation import CorrelationPlot
from T2IMetrics.figures.bar import BarPlot
from T2IMetrics.figures.scatter import ScatterPlot

def test_correlation_plot():
    # Load SEG-level data
    df = pd.read_csv('output/metric_results.csv')
    
    # Create label map
    label_map = {
        'blip1_dsg': 'BLIP1 (DSG)',
        'llava_dsg': 'LLaVA (DSG)',
        'gpt4v_dsg': 'GPT4V (DSG)',
        'blip1_tifa': 'BLIP1 (TIFA)',
        'llava_tifa': 'LLaVA (TIFA)',
        'gpt4v_tifa': 'GPT4V (TIFA)',
    }
    
    # Test correlation plot
    corr_plot = CorrelationPlot(
        label_map=label_map,
        metric_type='spearman',
        partition='synth',
        figsize=(10, 8)
    )
    corr_plot.generate(df, test=True)
    
    # Test scatter plot
    scatter_plot = ScatterPlot(
        x_metric='llava_dsg',
        y_metrics=['gpt4v_dsg', 'blip1_dsg'],
        label_map=label_map,
        metric_type='spearman',
        partition='synth',
        figsize=(10, 6)
    )
    scatter_plot.generate(df, test=True)
    
    # Load metric-level data
    df_summary = pd.read_csv('output/metric_summary.csv')
    
    # Test bar plot
    bar_plot = BarPlot(
        label_map=label_map,
        metric_type='spearman',
        partition='synth',
        figsize=(10, 6)
    )
    bar_plot.generate(df_summary, test=True)

if __name__ == '__main__':
    test_correlation_plot() 