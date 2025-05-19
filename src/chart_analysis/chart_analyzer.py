"""
Chart Analyzer for Chart Insights System.
Coordinates chart recognition, data extraction, and analysis.
"""

import logging
import numpy as np
import pandas as pd
import cv2
import pytesseract
from typing import Dict, List, Any, Optional, Tuple, Union

from .extractors import (
    BaseExtractor,
    BarChartExtractor,
    LineChartExtractor,
    PieChartExtractor,
    ScatterChartExtractor
)

from .visualizers import (
    BaseVisualizer,
    BarChartVisualizer,
    LineChartVisualizer,
    PieChartVisualizer,
    ScatterChartVisualizer
)

logger = logging.getLogger(__name__)

class ChartAnalyzer:
    """
    Analyzes charts to extract data and metadata.
    
    This class handles both direct data sources and chart images,
    extracting structured data and chart properties.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize chart analyzer.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.ocr_config = config.get('chart_analysis', {}).get('ocr_config', '--psm 6')
        self.supported_types = config.get('chart_analysis', {}).get('supported_types', 
                                        ['bar', 'line', 'pie', 'scatter'])
        
        # Initialize extractors
        self.extractors = {
            'bar': BarChartExtractor(config),
            'line': LineChartExtractor(config),
            'pie': PieChartExtractor(config),
            'scatter': ScatterChartExtractor(config)
        }
        
        # Initialize visualizers
        self.visualizers = {
            'bar': BarChartVisualizer(config),
            'line': LineChartVisualizer(config),
            'pie': PieChartVisualizer(config),
            'scatter': ScatterChartVisualizer(config)
        }
    
    def analyze_chart(self, 
                      chart_image: Optional[Union[str, np.ndarray]] = None, 
                      chart_data: Optional[pd.DataFrame] = None,
                      chart_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a chart from image or data.
        
        Args:
            chart_image: Path to chart image or image array
            chart_data: DataFrame containing chart data
            chart_type: Type of chart (bar, line, pie, etc.)
            
        Returns:
            Dictionary with extracted data and metadata
        """
        # Validate inputs
        if chart_image is None and chart_data is None:
            raise ValueError("Either chart_image or chart_data must be provided")
        
        # Process image if provided
        if chart_image is not None:
            return self._analyze_chart_image(chart_image, chart_type)
        
        # Process data if provided
        if chart_data is not None:
            return self._analyze_chart_data(chart_data, chart_type)
    
    def _analyze_chart_image(self, 
                           chart_image: Union[str, np.ndarray],
                           chart_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a chart from image.
        
        Args:
            chart_image: Path to chart image or image array
            chart_type: Type of chart (bar, line, pie, etc.)
            
        Returns:
            Dictionary with extracted data and metadata
        """
        # Load image if path provided
        if isinstance(chart_image, str):
            image = cv2.imread(chart_image)
            if image is None:
                raise ValueError(f"Failed to load image from path: {chart_image}")
        else:
            image = chart_image
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect chart type if not provided
        if chart_type is None:
            chart_type = self._detect_chart_type(image_rgb)
            logger.info(f"Detected chart type: {chart_type}")
        
        # Extract text elements
        text_elements = self._extract_text_elements(image_rgb)
        
        # Get appropriate extractor
        extractor = self.extractors.get(chart_type)
        if extractor is None:
            logger.warning(f"No extractor available for chart type: {chart_type}")
            # Default to bar chart extractor
            extractor = self.extractors['bar']
        
        # Extract data
        chart_data, metadata = extractor.extract_from_image(image_rgb, text_elements)
        
        # Combine results
        result = {
            'chart_type': chart_type,
            'data': chart_data,
            'metadata': metadata,
            'text_elements': text_elements
        }
        
        return result
    
    def _analyze_chart_data(self, 
                           chart_data: pd.DataFrame,
                           chart_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a chart from data.
        
        Args:
            chart_data: DataFrame containing chart data
            chart_type: Type of chart (bar, line, pie, etc.)
            
        Returns:
            Dictionary with extracted data and metadata
        """
        # Infer chart type if not provided
        if chart_type is None:
            chart_type = self._infer_chart_type(chart_data)
            logger.info(f"Inferred chart type: {chart_type}")
        
        # Get appropriate extractor
        extractor = self.extractors.get(chart_type)
        if extractor is None:
            logger.warning(f"No extractor available for chart type: {chart_type}")
            # Default to bar chart extractor
            extractor = self.extractors['bar']
        
        # Extract data and metadata
        chart_data, metadata = extractor.extract_from_dataframe(chart_data)
        
        # Create text elements from metadata
        text_elements = {
            'title': metadata.get('title', ''),
            'x_axis_label': metadata.get('x_axis_label', ''),
            'y_axis_label': metadata.get('y_axis_label', ''),
            'x_tick_labels': [],
            'y_tick_labels': [],
            'legend_items': metadata.get('legend_items', [])
        }
        
        # Return result
        result = {
            'chart_type': chart_type,
            'data': chart_data,
            'metadata': metadata,
            'text_elements': text_elements
        }
        
        return result
    
    def _detect_chart_type(self, image: np.ndarray) -> str:
        """
        Detect chart type from image.
        
        Args:
            image: RGB image array
            
        Returns:
            Chart type string
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=100, 
            minLineLength=100, maxLineGap=10
        )
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=30, maxRadius=300
        )
        
        # Count horizontal and vertical lines
        h_lines = 0
        v_lines = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 20 or angle > 160:
                    h_lines += 1
                elif 70 < angle < 110:
                    v_lines += 1
        
        # Determine chart type based on features
        if circles is not None and len(circles) > 0:
            return 'pie'
        elif h_lines > 10 and v_lines > 10:
            return 'scatter'
        elif h_lines > v_lines:
            return 'bar'
        else:
            return 'line'
    
    def _extract_text_elements(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text elements from chart image using OCR.
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with extracted text elements
        """
        # Extract text using Tesseract
        text_data = pytesseract.image_to_data(
            image, 
            config=self.ocr_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Filter out empty strings
        filtered_indices = [i for i, text in enumerate(text_data['text']) if text.strip()]
        
        # Extract text blocks with positions
        text_blocks = []
        for i in filtered_indices:
            text_blocks.append({
                'text': text_data['text'][i],
                'left': text_data['left'][i],
                'top': text_data['top'][i],
                'width': text_data['width'][i],
                'height': text_data['height'][i],
                'conf': text_data['conf'][i]
            })
        
        # Analyze text positions to identify chart elements
        height, width = image.shape[:2]
        
        # Initialize result
        result = {
            'title': '',
            'x_axis_label': '',
            'y_axis_label': '',
            'x_tick_labels': [],
            'y_tick_labels': [],
            'legend_items': []
        }
        
        # Identify title (typically at top and center)
        top_blocks = sorted([b for b in text_blocks if b['top'] < height * 0.2], 
                           key=lambda b: b['left'])
        
        if top_blocks:
            # Combine adjacent blocks for title
            title_parts = []
            for block in top_blocks:
                if block['conf'] > 60:  # Confidence threshold
                    title_parts.append(block['text'])
            
            result['title'] = ' '.join(title_parts)
        
        # Identify x-axis label (typically at bottom and center)
        bottom_blocks = sorted([b for b in text_blocks if b['top'] > height * 0.8 and b['left'] > width * 0.2],
                              key=lambda b: b['top'])
        
        if bottom_blocks:
            # Find the lowest blocks that aren't tick labels
            x_label_parts = []
            for block in bottom_blocks[:3]:  # Consider up to 3 blocks
                if block['conf'] > 60:
                    x_label_parts.append(block['text'])
            
            result['x_axis_label'] = ' '.join(x_label_parts)
        
        # Identify y-axis label (typically at left and middle)
        left_blocks = sorted([b for b in text_blocks if b['left'] < width * 0.15],
                            key=lambda b: b['top'])
        
        if left_blocks:
            # Find blocks in the middle of the y-axis
            y_middle_blocks = [b for b in left_blocks 
                               if height * 0.3 < b['top'] < height * 0.7]
            
            if y_middle_blocks:
                y_label_parts = []
                for block in y_middle_blocks:
                    if block['conf'] > 60:
                        y_label_parts.append(block['text'])
                
                result['y_axis_label'] = ' '.join(y_label_parts)
        
        # Identify x-tick labels (bottom, aligned horizontally)
        x_tick_candidates = [b for b in text_blocks 
                            if height * 0.75 < b['top'] < height * 0.95
                            and b['conf'] > 60]
        
        if x_tick_candidates:
            result['x_tick_labels'] = [b['text'] for b in sorted(x_tick_candidates, key=lambda b: b['left'])]
        
        # Identify y-tick labels (left, aligned vertically)
        y_tick_candidates = [b for b in text_blocks
                            if b['left'] < width * 0.15
                            and b['conf'] > 60
                            and len(b['text'].strip()) > 0]
        
        if y_tick_candidates:
            result['y_tick_labels'] = [b['text'] for b in sorted(y_tick_candidates, key=lambda b: b['top'], reverse=True)]
        
        # Identify legend items (typically in the corner)
        legend_candidates = [b for b in text_blocks
                            if ((b['left'] > width * 0.7 and b['top'] < height * 0.3) or  # Top right
                                (b['left'] > width * 0.7 and b['top'] > height * 0.7))    # Bottom right
                            and b['conf'] > 60]
        
        if legend_candidates:
            result['legend_items'] = [b['text'] for b in legend_candidates]
        
        return result
    
    def _infer_chart_type(self, data: pd.DataFrame) -> str:
        """
        Infer chart type from DataFrame structure.
        
        Args:
            data: DataFrame containing chart data
            
        Returns:
            Chart type string
        """
        num_cols = len(data.columns)
        num_rows = len(data)
        
        # Check column types
        numeric_cols = len(data.select_dtypes(include=['number']).columns)
        categorical_cols = len(data.select_dtypes(include=['object', 'category']).columns)
        
        # Infer chart type based on data structure
        if num_cols == 2:
            if categorical_cols == 1 and numeric_cols == 1:
                # One category column, one numeric column
                if num_rows <= 10:
                    return 'pie'
                else:
                    return 'bar'
            elif numeric_cols == 2:
                # Two numeric columns
                return 'scatter'
        elif num_cols > 2:
            first_col_type = data.dtypes.iloc[0]
            if pd.api.types.is_numeric_dtype(first_col_type) or pd.api.types.is_datetime64_dtype(first_col_type):
                # First column is numeric or datetime, rest are values
                return 'line'
            else:
                # First column is category, rest are values
                return 'bar'
        
        # Default to bar chart
        return 'bar'
    
    def visualize_chart(self, 
                      chart_data: pd.DataFrame, 
                      chart_type: str, 
                      metadata: Dict[str, Any]) -> np.ndarray:
        """
        Visualize chart from data.
        
        Args:
            chart_data: DataFrame containing chart data
            chart_type: Type of chart (bar, line, pie, etc.)
            metadata: Chart metadata
            
        Returns:
            RGB image array of visualization
        """
        # Get appropriate visualizer
        visualizer = self.visualizers.get(chart_type)
        if visualizer is None:
            logger.warning(f"No visualizer available for chart type: {chart_type}")
            # Default to bar chart visualizer
            visualizer = self.visualizers['bar']
        
        # Generate visualization
        return visualizer.visualize(chart_data, metadata)
