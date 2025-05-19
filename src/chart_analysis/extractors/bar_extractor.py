"""
Bar chart data extractor.
"""

import logging
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Any, Optional, Tuple

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class BarChartExtractor(BaseExtractor):
    """
    Extracts data from bar charts.
    """
    
    def extract_from_image(self, 
                         image: np.ndarray, 
                         text_elements: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data from bar chart image.
        
        Args:
            image: RGB image array
            text_elements: Dictionary with extracted text elements
            
        Returns:
            Tuple of (DataFrame with extracted data, metadata dictionary)
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find bars
        bar_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / max(w, 1)
            
            # Likely a bar if tall and narrow
            if aspect_ratio > 2 and h > height * 0.1:
                bar_contours.append((x, y, w, h))
        
        # Sort bars by x position
        bar_contours.sort(key=lambda b: b[0])
        
        # Create categories from x-tick labels or generate them
        if text_elements['x_tick_labels']:
            categories = text_elements['x_tick_labels']
            
            # Ensure we have the right number of categories
            if len(categories) != len(bar_contours):
                # Adjust by truncating or extending
                if len(categories) > len(bar_contours):
                    categories = categories[:len(bar_contours)]
                else:
                    categories.extend([f'Category {i+len(categories)+1}' 
                                      for i in range(len(bar_contours) - len(categories))])
        else:
            categories = [f'Category {i+1}' for i in range(len(bar_contours))]
        
        # Calculate values based on bar heights
        max_height = max([h for _, y, _, h in bar_contours], default=1)
        ref_height = height * 0.8  # Reference point for 0
        
        values = []
        for _, y, _, h in bar_contours:
            # Normalize height to value
            bar_top = y
            bar_height = h
            
            # Calculate value as proportion of max height
            value = bar_height / max_height
            values.append(value)
        
        # If we have y-tick labels, try to calibrate the values
        if text_elements['y_tick_labels'] and len(text_elements['y_tick_labels']) >= 2:
            try:
                # Extract numeric values from y-tick labels
                y_values = []
                for label in text_elements['y_tick_labels']:
                    # Try to extract a number
                    num_str = ''.join(c for c in label if c.isdigit() or c in '.-')
                    if num_str:
                        y_values.append(float(num_str))
                
                if y_values:
                    y_min = min(y_values)
                    y_max = max(y_values)
                    
                    # Recalibrate values
                    values = [y_min + v * (y_max - y_min) for v in values]
            except Exception as e:
                logger.warning(f"Failed to calibrate values from y-tick labels: {e}")
        
        # Create DataFrame
        data = pd.DataFrame({
            'Category': categories,
            'Value': values
        })
        
        # Create metadata
        metadata = {
            'title': text_elements['title'],
            'x_axis_label': text_elements['x_axis_label'],
            'y_axis_label': text_elements['y_axis_label'],
            'legend_items': text_elements['legend_items']
        }
        
        return data, metadata
    
    def extract_from_dataframe(self, 
                             data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract metadata from bar chart data.
        
        Args:
            data: DataFrame containing chart data
            
        Returns:
            Tuple of (DataFrame with extracted data, metadata dictionary)
        """
        # Check if DataFrame matches bar chart structure
        if len(data.columns) < 2:
            logger.warning("Bar chart data should have at least 2 columns")
            # Create a simple structure
            data = pd.DataFrame({
                'Category': [f'Category {i+1}' for i in range(len(data))],
                'Value': data.iloc[:, 0] if not data.empty else []
            })
        
        # Ensure first column is category, second is value
        categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
        numeric_cols = list(data.select_dtypes(include=['number']).columns)
        
        if categorical_cols and numeric_cols:
            # Use first categorical column as category and first numeric as value
            category_col = categorical_cols[0]
            value_col = numeric_cols[0]
            
            # Extract only relevant columns
            cleaned_data = pd.DataFrame({
                'Category': data[category_col],
                'Value': data[value_col]
            })
        else:
            # If no categorical or numeric column found, use original data
            cleaned_data = data.copy()
            if len(data.columns) >= 2:
                cleaned_data.columns = ['Category', 'Value']
        
        # Create metadata
        metadata = {
            'title': '',
            'x_axis_label': str(data.columns[0]) if len(data.columns) > 0 else 'Category',
            'y_axis_label': str(data.columns[1]) if len(data.columns) > 1 else 'Value',
            'legend_items': []
        }
        
        return cleaned_data, metadata
