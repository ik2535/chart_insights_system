"""
Scatter chart visualizer.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

class ScatterChartVisualizer(BaseVisualizer):
    """
    Visualizes data as scatter charts.
    """
    
    def visualize(self, 
                data: pd.DataFrame, 
                metadata: Dict[str, Any]) -> np.ndarray:
        """
        Visualize data as scatter chart.
        
        Args:
            data: DataFrame containing chart data
            metadata: Dictionary with chart metadata
            
        Returns:
            RGB image array of visualization
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Extract data
        if 'X' in data.columns and 'Y' in data.columns:
            x_values = data['X']
            y_values = data['Y']
        else:
            # Use first column as X, second as Y
            x_values = data.iloc[:, 0]
            y_values = data.iloc[:, 1] if len(data.columns) > 1 else pd.Series([0] * len(data))
        
        # Check if we have series information
        if 'Series' in data.columns:
            # Get unique series
            series_list = data['Series'].unique()
            
            # Plot each series with different color
            for series in series_list:
                series_data = data[data['Series'] == series]
                ax.scatter(
                    series_data['X'], 
                    series_data['Y'], 
                    label=series, 
                    alpha=0.7,
                    s=50  # marker size
                )
            
            # Add legend
            ax.legend()
        else:
            # Plot single series
            ax.scatter(x_values, y_values, color='#4285F4', alpha=0.7, s=50)
        
        # Add labels and title
        ax.set_xlabel(metadata.get('x_axis_label', ''))
        ax.set_ylabel(metadata.get('y_axis_label', ''))
        ax.set_title(metadata.get('title', ''), fontsize=14, fontweight='bold')
        
        # Apply consistent style
        self.apply_style(fig, ax)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to image array
        img_array = self.figure_to_image(fig)
        
        # Close figure to free memory
        plt.close(fig)
        
        return img_array
