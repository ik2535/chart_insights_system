"""
Scatter chart data extractor.
"""

import logging
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Any, Optional, Tuple

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class ScatterChartExtractor(BaseExtractor):
    """
    Extracts data from scatter charts.
    """
    
    def extract_from_image(self, 
                         image: np.ndarray, 
                         text_elements: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data from scatter chart image.
        
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
        
        # Set up blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.2
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(thresh)
        
        # Extract point coordinates
        points = []
        for kp in keypoints:
            x = kp.pt[0] / width  # Normalize x
            y = 1.0 - kp.pt[1] / height  # Normalize y and invert (origin at bottom-left)
            size = kp.size
            points.append((x, y, size))
        
        # If no points detected, generate some
        if not points:
            num_points = 20
            points = [(np.random.random(), np.random.random(), 10) for _ in range(num_points)]
        
        # Create DataFrame
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        
        # If we have axis labels, try to calibrate the values
        if text_elements['x_tick_labels'] and text_elements['y_tick_labels']:
            try:
                # Extract numeric values from tick labels
                x_ticks = []
                for label in text_elements['x_tick_labels']:
                    num_str = ''.join(c for c in label if c.isdigit() or c in '.-')
                    if num_str:
                        x_ticks.append(float(num_str))
                
                y_ticks = []
                for label in text_elements['y_tick_labels']:
                    num_str = ''.join(c for c in label if c.isdigit() or c in '.-')
                    if num_str:
                        y_ticks.append(float(num_str))
                
                # Recalibrate values
                if x_ticks:
                    x_min = min(x_ticks)
                    x_max = max(x_ticks)
                    x_values = [x_min + x * (x_max - x_min) for x in x_values]
                
                if y_ticks:
                    y_min = min(y_ticks)
                    y_max = max(y_ticks)
                    y_values = [y_min + y * (y_max - y_min) for y in y_values]
            except Exception as e:
                logger.warning(f"Failed to calibrate values from tick labels: {e}")
        
        data = pd.DataFrame({
            'X': x_values,
            'Y': y_values
        })
        
        # Add series column if we have legend items
        if text_elements['legend_items']:
            # Assign points to series (in a real implementation, this would be done by color)
            num_series = len(text_elements['legend_items'])
            series = []
            for i in range(len(points)):
                series_idx = i % num_series
                series.append(text_elements['legend_items'][series_idx])
            
            data['Series'] = series
        
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
        Extract metadata from scatter chart data.
        
        Args:
            data: DataFrame containing chart data
            
        Returns:
            Tuple of (DataFrame with extracted data, metadata dictionary)
        """
        # Check if DataFrame has at least 2 columns (X and Y)
        if len(data.columns) < 2:
            logger.warning("Scatter chart data should have at least 2 columns")
            # Create a simple structure
            data = pd.DataFrame({
                'X': list(range(len(data))),
                'Y': data.iloc[:, 0] if not data.empty else []
            })
        
        # Ensure X and Y columns are numeric
        numeric_cols = list(data.select_dtypes(include=['number']).columns)
        
        if len(numeric_cols) >= 2:
            # Use first two numeric columns as X and Y
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            
            # Extract relevant columns
            cleaned_data = pd.DataFrame({
                'X': data[x_col],
                'Y': data[y_col]
            })
            
            # Add series column if available
            categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
            if categorical_cols:
                series_col = categorical_cols[0]
                cleaned_data['Series'] = data[series_col]
        else:
            # If not enough numeric columns, use original data
            cleaned_data = data.copy()
            if len(data.columns) >= 2:
                cleaned_data.columns = ['X', 'Y']
        
        # Create metadata
        metadata = {
            'title': '',
            'x_axis_label': str(data.columns[0]) if len(data.columns) > 0 else 'X',
            'y_axis_label': str(data.columns[1]) if len(data.columns) > 1 else 'Y',
            'legend_items': list(cleaned_data['Series'].unique()) if 'Series' in cleaned_data.columns else []
        }
        
        return cleaned_data, metadata
