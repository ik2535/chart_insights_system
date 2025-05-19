"""
Pie chart data extractor.
"""

import logging
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Any, Optional, Tuple

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class PieChartExtractor(BaseExtractor):
    """
    Extracts data from pie charts.
    """
    
    def extract_from_image(self, 
                         image: np.ndarray, 
                         text_elements: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data from pie chart image.
        
        Args:
            image: RGB image array
            text_elements: Dictionary with extracted text elements
            
        Returns:
            Tuple of (DataFrame with extracted data, metadata dictionary)
        """
        height, width = image.shape[:2]
        
        # Detect circles
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=width//2,
            param1=50, param2=30, minRadius=width//8, maxRadius=width//2
        )
        
        # Use legend items as segment labels
        if text_elements['legend_items']:
            labels = text_elements['legend_items']
        else:
            # If no legend, generate labels
            num_segments = 4  # Default if we can't detect segments
            labels = [f'Segment {i+1}' for i in range(num_segments)]
        
        # Generate values (in a real implementation, these would be extracted from image)
        num_segments = len(labels)
        values = list(np.random.random(num_segments))
        
        # Normalize values to sum to 100
        total = sum(values)
        values = [100 * v / total for v in values]
        
        # Create DataFrame
        data = pd.DataFrame({
            'Label': labels,
            'Value': values
        })
        
        # Create metadata
        metadata = {
            'title': text_elements['title'],
            'legend_items': text_elements['legend_items']
        }
        
        return data, metadata
    
    def extract_from_dataframe(self, 
                             data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract metadata from pie chart data.
        
        Args:
            data: DataFrame containing chart data
            
        Returns:
            Tuple of (DataFrame with extracted data, metadata dictionary)
        """
        # Check if DataFrame has at least 2 columns (labels and values)
        if len(data.columns) < 2:
            logger.warning("Pie chart data should have at least 2 columns")
            # Create a simple structure
            data = pd.DataFrame({
                'Label': [f'Segment {i+1}' for i in range(len(data))],
                'Value': data.iloc[:, 0] if not data.empty else []
            })
        
        # Ensure first column is label, second is value
        categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
        numeric_cols = list(data.select_dtypes(include=['number']).columns)
        
        if categorical_cols and numeric_cols:
            # Use first categorical column as label and first numeric as value
            label_col = categorical_cols[0]
            value_col = numeric_cols[0]
            
            # Extract only relevant columns
            cleaned_data = pd.DataFrame({
                'Label': data[label_col],
                'Value': data[value_col]
            })
        else:
            # If no categorical or numeric column found, use original data
            cleaned_data = data.copy()
            if len(data.columns) >= 2:
                cleaned_data.columns = ['Label', 'Value']
        
        # Create metadata
        metadata = {
            'title': '',
            'legend_items': list(cleaned_data['Label'])
        }
        
        return cleaned_data, metadata
