from typing import Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .base import SEGLevelFigure

class CorrelationPlot(SEGLevelFigure):
    """Generate correlation heatmap using SEG-level data."""
    
    def __init__(self, *args, vmin: float = 0, vmax: float = 70, **kwargs):
        super().__init__(*args, **kwargs)
        self.vmin = vmin
        self.vmax = vmax
        
    def generate(self, df: pd.DataFrame, output_path: Optional[str] = None, test: bool = False) -> None:
        """Generate correlation heatmap."""
        print("Input columns:", df.columns.tolist())  # Debug
        
        # Prepare data
        df = self.prepare_data(df)
        print("After prepare_data columns:", df.columns.tolist())  # Debug
        
        # Drop id and partition columns before correlation
        df = df.drop(columns=['id', 'partition'])
        
        # Calculate correlations
        corr = df.corr() * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Generate heatmap
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            vmin=self.vmin,
            vmax=self.vmax,
            fmt=".0f",
            ax=ax
        )
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add title based on metric type and partition
        title = f"Correlation of {self.metric_type.title()} Scores"
        if self.partition:
            title += f" ({self.partition.title()} Partition)"
            
        self.finalize_plot(title, output_path, test)