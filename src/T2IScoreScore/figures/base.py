from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FigureGenerator(ABC):
    """Base class for figure generation."""
    
    def __init__(self, 
                 label_map: Optional[Dict[str, str]] = None,
                 metric_type: str = "spearman",
                 partition: str = "synth",
                 drop_columns: Optional[List[str]] = None,
                 figsize: tuple = (8, 8)):
        """
        Args:
            label_map: Dictionary mapping model names to display labels
            metric_type: Type of metric to plot ('spearman', 'ks_test', 'delta')
            partition: Data partition to use ('synth', 'real', 'nat', 'overall')
            drop_columns: List of model names to exclude
            figsize: Figure size (width, height) in inches
        """
        self.label_map = label_map or {}
        self.metric_type = metric_type
        self.partition = partition
        self.drop_columns = drop_columns or []
        self.figsize = figsize
        
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for plotting."""
        pass
    
    def finalize_plot(self, title: Optional[str] = None, output_path: Optional[str] = None, test: bool = False):
        """Apply final formatting and save plot."""
        if title:
            plt.title(title, fontsize=12, pad=10)
            
        plt.tight_layout()
        
        if test:
            plt.show()
        elif output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            
        plt.close()

class SEGLevelFigure(FigureGenerator):
    """Base class for figures using SEG-level data from metric_results.csv"""
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare SEG-level data."""
        # Filter by partition if specified
        if self.partition:
            df = df[df['partition'] == self.partition]
            
        # Select columns for metric type
        metric_cols = [col for col in df.columns if col.endswith(self.metric_type)]
        df = df[['id', 'partition'] + metric_cols]
        
        # Drop specified columns
        for col in self.drop_columns:
            metric_cols = [c for c in metric_cols if not c.startswith(col)]
            
        # Rename columns using label map
        df = df.rename(columns={
            col: self.label_map.get(col.replace(f"_{self.metric_type}", ""), col)
            for col in metric_cols
        })
        
        return df

class MetricLevelFigure(FigureGenerator):
    """Base class for figures using metric-level data from metric_summary.csv"""
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare metric-level data."""
        # Get column for specific partition and metric
        col_name = f"{self.partition}_{self.metric_type}"
        
        # Select model column and the specific metric column
        df = df[['model', col_name]]
        
        # Drop specified models
        if self.drop_columns:
            df = df[~df['model'].isin(self.drop_columns)]
        
        # Rename models using label map
        df['model'] = df['model'].map(lambda x: self.label_map.get(x, x))
        
        # Set model as index for correlation calculations
        df = df.set_index('model')
        
        return df 