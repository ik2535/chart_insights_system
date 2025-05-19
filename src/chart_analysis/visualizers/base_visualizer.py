"""
Base visualizer for chart visualization.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import io

logger = logging.getLogger(__name__)

class BaseVisualizer:
    """
    Base class for chart visualizers.
    
    This class provides common functionality for all chart type visualizers
    and defines the interface that specific visualizers should implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base visualizer.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.default_figsize = (10, 6)
        self.default_dpi = 100
    
    def visualize(self, 
                data: pd.DataFrame, 
                metadata: Dict[str, Any]) -> np.ndarray:
        """
        Visualize data as chart.
        
        Args:
            data: DataFrame containing chart data
            metadata: Dictionary with chart metadata
            
        Returns:
            RGB image array of visualization
        """
        raise NotImplementedError("Subclasses must implement visualize")
    
    def figure_to_image(self, fig) -> np.ndarray:
        """
        Convert matplotlib figure to RGB image array.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            RGB image array
        """
        # Save figure to memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.default_dpi)
        buf.seek(0)
        
        # Convert to PIL Image
        img = Image.open(buf)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Close buffer
        buf.close()
        
        return img_array
    
    def apply_style(self, fig, ax):
        """
        Apply consistent style to figure and axes.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes
        """
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set fonts
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        
        # Apply style to axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
