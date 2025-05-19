"""
Line chart visualizer.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

class LineChartVisualizer(BaseVisualizer):
    """
    Visualizes data as line charts.
    """
    
    def visualize(self, 
                data: pd.DataFrame, 
                metadata: Dict[str, Any]) -> np.ndarray:
        """
        Visualize data as line chart.
        
        Args:
            data: DataFrame containing chart data
            metadata: Dictionary with chart metadata
            
        Returns:
            RGB image array of visualization
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Check if we have multiple series
        if len(data.columns) > 2:
            # Multiple series
            x_col = data.columns[0]
            x_values = data[x_col]
            
            # Plot each series
            for i, col in enumerate(data.columns[1:]):
                ax.plot(x_values, data[col], marker='o', markersize=4, 
                       label=col, linewidth=2)
            
            # Add legend
            ax.legend()
        else:
            # Single series
            if 'X' in data.columns and 'Y' in data.columns:
                x_values = data['X']
                y_values = data['Y']
            else:
                # Use first column as X, second as Y
                x_values = data.iloc[:, 0]
                y_values = data.iloc[:, 1] if len(data.columns) > 1 else pd.Series([0] * len(data))
            
            # Plot line
            ax.plot(x_values, y_values, marker='o', markersize=4, 
                   color='#4285F4', linewidth=2)
        
        # Add labels and title
        ax.set_xlabel(metadata.get('x_axis_label', ''))
        ax.set_ylabel(metadata.get('y_axis_label', ''))
        ax.set_title(metadata.get('title', ''), fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels if there are many categories
        if len(x_values) > 5:
            plt.xticks(rotation=45, ha='right')
        
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
