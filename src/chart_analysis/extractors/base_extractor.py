"""
Base extractor for chart data extraction.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class BaseExtractor:
    """
    Base class for chart data extractors.
    
    This class provides common functionality for all chart type extractors
    and defines the interface that specific extractors should implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base extractor.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
    
    def extract_data(self, 
                   image: Optional[np.ndarray] = None,
                   text_elements: Optional[Dict[str, Any]] = None,
                   data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data from chart image or directly from data.
        
        Args:
            image: RGB image array (optional)
            text_elements: Dictionary with extracted text elements (optional)
            data: DataFrame containing chart data (optional)
            
        Returns:
            Tuple of (DataFrame with extracted data, metadata dictionary)
        """
        if image is not None and text_elements is not None:
            return self.extract_from_image(image, text_elements)
        elif data is not None:
            return self.extract_from_dataframe(data)
        else:
            raise ValueError("Either image and text_elements or data must be provided")
    
    def extract_from_image(self, 
                         image: np.ndarray, 
                         text_elements: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data from chart image.
        
        Args:
            image: RGB image array
            text_elements: Dictionary with extracted text elements
            
        Returns:
            Tuple of (DataFrame with extracted data, metadata dictionary)
        """
        raise NotImplementedError("Subclasses must implement extract_from_image")
    
    def extract_from_dataframe(self, 
                             data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract metadata from chart data.
        
        Args:
            data: DataFrame containing chart data
            
        Returns:
            Tuple of (DataFrame with extracted data, metadata dictionary)
        """
        # Create basic metadata (should be overridden by subclasses)
        metadata = {
            'title': '',
            'x_axis_label': '',
            'y_axis_label': '',
            'legend_items': []
        }
        
        # Try to extract column names for axis labels
        if len(data.columns) >= 2:
            metadata['x_axis_label'] = str(data.columns[0])
            metadata['y_axis_label'] = str(data.columns[1])
        
        return data, metadata
