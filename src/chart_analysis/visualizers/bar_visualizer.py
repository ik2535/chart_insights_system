"""
Bar chart visualizer.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

class BarChartVisualizer(BaseVisualizer):
    """
    Visualizes data as bar charts.
    """
    
    def visualize(self, 
                data: pd.DataFrame, 
                metadata: Dict[str, Any]) -> np.ndarray:
        """
        Visualize data as bar chart.
        
        Args:
            data: DataFrame containing chart data
            metadata: Dictionary with chart metadata
            
        Returns:
            RGB image array of visualization
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Extract data
        if 'Category' in data.columns and 'Value' in data.columns:
            categories = data['Category']
            values = data['Value']
        else:
            # Use first column as categories, second as values
            categories = data.iloc[:, 0]
            values = data.iloc[:, 1] if len(data.columns) > 1 else pd.Series([0] * len(data))
        
        # Create bar chart
        bars = ax.bar(categories, values, color='#4285F4')
        
        # Add labels and title
        ax.set_xlabel(metadata.get('x_axis_label', ''))
        ax.set_ylabel(metadata.get('y_axis_label', ''))
        ax.set_title(metadata.get('title', ''), fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels if there are many categories
        if len(categories) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(values),
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Apply consistent style
        self.apply_style(fig, ax)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to image array
        img_array = self.figure_to_image(fig)
        
        # Close figure to free memory
        plt.close(fig)
        
        return img_array
