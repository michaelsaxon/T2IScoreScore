from typing import Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .base import SEGLevelFigure

class ScatterPlot(SEGLevelFigure):
    """Generate scatter plot comparing two metrics using SEG-level data."""
    
    def __init__(self, 
                 x_metric: str,
                 y_metrics: str | list[str],
                 *args, 
                 **kwargs):
        """
        Args:
            x_metric: Name of metric to use for x-axis
            y_metrics: Name(s) of metric(s) to use for y-axis
            *args, **kwargs: Arguments passed to FigureGenerator
        """
        super().__init__(*args, **kwargs)
        self.x_metric = x_metric
        self.y_metrics = [y_metrics] if isinstance(y_metrics, str) else y_metrics
        
    def generate(self, df: pd.DataFrame, output_path: Optional[str] = None, test: bool = False) -> None:
        """Generate scatter plot."""
        # Prepare data
        df = self.prepare_data(df)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get column names with metric type
        x_col = f"{self.x_metric}_{self.metric_type}"
        y_cols = [f"{y}_{self.metric_type}" for y in self.y_metrics]
        
        # Generate scatter plots
        for y_col in y_cols:
            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                label=self.label_map.get(y_col.replace(f"_{self.metric_type}", ""), y_col),
                ax=ax
            )
        
        # Customize appearance
        plt.xlabel(self.label_map.get(self.x_metric, self.x_metric))
        plt.ylabel(f"{self.metric_type.title()} Score")
        
        # Add title
        title = f"{self.metric_type.title()} Score Comparison"
        if self.partition:
            title += f" ({self.partition.title()} Partition)"
            
        self.finalize_plot(title, output_path, test) 