"""
Line chart data extractor.
"""

import logging
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Any, Optional, Tuple

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class LineChartExtractor(BaseExtractor):
    """
    Extracts data from line charts.
    """
    
    def extract_from_image(self, 
                         image: np.ndarray, 
                         text_elements: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data from line chart image.
        
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
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50, 
            minLineLength=50, maxLineGap=20
        )
        
        # Extract rough points from lines
        # This is very simplified and would not work well for complex line charts
        line_segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Only consider line segments that aren't too vertical or horizontal
                if 20 < angle < 160:
                    line_segments.append((x1, y1, x2, y2))
        
        # For a real implementation, we would use a line tracing algorithm
        # Here we'll just generate sample data based on x-tick labels
        
        # Create x values from x-tick labels or generate them
        if text_elements['x_tick_labels']:
            x_values = text_elements['x_tick_labels']
        else:
            num_points = max(10, len(line_segments) // 2)
            x_values = [f'Point {i+1}' for i in range(num_points)]
        
        # Generate y values (in a real implementation, these would be extracted from image)
        num_points = len(x_values)
        y_values = list(np.random.random(num_points))
        
        # If we have y-tick labels, try to calibrate the values
        if text_elements['y_tick_labels'] and len(text_elements['y_tick_labels']) >= 2:
            try:
                # Extract numeric values from y-tick labels
                y_ticks = []
                for label in text_elements['y_tick_labels']:
                    # Try to extract a number
                    num_str = ''.join(c for c in label if c.isdigit() or c in '.-')
                    if num_str:
                        y_ticks.append(float(num_str))
                
                if y_ticks:
                    y_min = min(y_ticks)
                    y_max = max(y_ticks)
                    
                    # Recalibrate values
                    y_values = [y_min + v * (y_max - y_min) for v in y_values]
            except Exception as e:
                logger.warning(f"Failed to calibrate values from y-tick labels: {e}")
        
        # Create DataFrame
        data = pd.DataFrame({
            'X': x_values,
            'Y': y_values
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
        Extract metadata from line chart data.
        
        Args:
            data: DataFrame containing chart data
            
        Returns:
            Tuple of (DataFrame with extracted data, metadata dictionary)
        """
        # Check if DataFrame has at least 2 columns (X and Y)
        if len(data.columns) < 2:
            logger.warning("Line chart data should have at least 2 columns")
            # Create a simple structure
            data = pd.DataFrame({
                'X': list(range(len(data))),
                'Y': data.iloc[:, 0] if not data.empty else []
            })
        
        # Determine if we have multiple series
        if len(data.columns) > 2:
            # Multiple series
            x_col = data.columns[0]
            series_cols = data.columns[1:]
            
            # Ensure X column is first
            cleaned_data = data.copy()
        else:
            # Single series
            cleaned_data = data.copy()
            if len(data.columns) >= 2:
                cleaned_data.columns = ['X', 'Y']
        
        # Create metadata
        metadata = {
            'title': '',
            'x_axis_label': str(data.columns[0]) if len(data.columns) > 0 else 'X',
            'y_axis_label': str(data.columns[1]) if len(data.columns) > 1 else 'Y',
            'legend_items': [str(col) for col in data.columns[1:]] if len(data.columns) > 2 else []
        }
        
        return cleaned_data, metadata
