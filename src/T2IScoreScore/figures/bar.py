from typing import Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .base import MetricLevelFigure

class BarPlot(MetricLevelFigure):
    """Generate bar plot comparing metrics using metric-level data."""
    
    def generate(self, df: pd.DataFrame, output_path: Optional[str] = None, test: bool = False) -> None:
        """Generate bar plot."""
        # Prepare data
        df = self.prepare_data(df)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Generate bar plot
        col_name = f"{self.partition}_{self.metric_type}"
        df[col_name].plot(kind='bar', ax=ax)
        
        # Customize appearance
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(f"{self.metric_type.title()} Score")
        
        # Add title
        title = f"{self.metric_type.title()} Scores by Model"
        if self.partition:
            title += f" ({self.partition.title()} Partition)"
            
        self.finalize_plot(title, output_path, test) 