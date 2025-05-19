"""
Pie chart visualizer.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

from .base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

class PieChartVisualizer(BaseVisualizer):
    """
    Visualizes data as pie charts.
    """
    
    def visualize(self, 
                data: pd.DataFrame, 
                metadata: Dict[str, Any]) -> np.ndarray:
        """
        Visualize data as pie chart.
        
        Args:
            data: DataFrame containing chart data
            metadata: Dictionary with chart metadata
            
        Returns:
            RGB image array of visualization
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Extract data
        if 'Label' in data.columns and 'Value' in data.columns:
            labels = data['Label']
            values = data['Value']
        else:
            # Use first column as labels, second as values
            labels = data.iloc[:, 0]
            values = data.iloc[:, 1] if len(data.columns) > 1 else pd.Series([1] * len(data))
        
        # Generate colors
        colors = plt.cm.tab10(np.arange(len(labels)) % 10)
        
        # Create explode array (slight separation for largest slice)
        explode = np.zeros(len(values))
        if len(values) > 0:
            explode[values.argmax()] = 0.1
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90, 
            explode=explode, 
            colors=colors,
            shadow=True,
            textprops={'fontsize': 9}
        )
        
        # Make percentage text white for better visibility on dark slices
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add title
        ax.set_title(metadata.get('title', ''), fontsize=14, fontweight='bold')
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Add legend if many categories
        if len(labels) > 5:
            ax.legend(
                wedges, 
                labels,
                title='Categories',
                loc='center left',
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
        
        # Apply consistent style (modified for pie charts)
        fig.patch.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to image array
        img_array = self.figure_to_image(fig)
        
        # Close figure to free memory
        plt.close(fig)
        
        return img_array
