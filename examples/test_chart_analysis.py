#!/usr/bin/env python
"""
Test script for Chart Analysis module.

This script demonstrates the usage of the Chart Analysis module by:
1. Creating sample chart data
2. Analyzing the chart data
3. Visualizing the chart 
4. Extracting insights

Usage:
    python test_chart_analysis.py
"""

import os
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chart_analysis import ChartAnalyzer

def load_config():
    """Load configuration from config file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_sample_data(chart_type):
    """Create sample data for chart."""
    if chart_type == 'bar':
        # Create bar chart data
        categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        values = [23, 45, 56, 78, 42]
        
        data = pd.DataFrame({'Category': categories, 'Value': values})
        metadata = {
            'title': 'Sample Bar Chart',
            'x_axis_label': 'Categories',
            'y_axis_label': 'Values',
            'legend_items': []
        }
        
    elif chart_type == 'line':
        # Create line chart data
        x_values = list(range(10))
        y_values = [i**2 for i in x_values]
        
        data = pd.DataFrame({'X': x_values, 'Y': y_values})
        metadata = {
            'title': 'Sample Line Chart',
            'x_axis_label': 'X Values',
            'y_axis_label': 'Y Values',
            'legend_items': []
        }
        
    elif chart_type == 'pie':
        # Create pie chart data
        labels = ['Segment A', 'Segment B', 'Segment C', 'Segment D']
        values = [25, 30, 15, 30]
        
        data = pd.DataFrame({'Label': labels, 'Value': values})
        metadata = {
            'title': 'Sample Pie Chart',
            'legend_items': labels
        }
        
    elif chart_type == 'scatter':
        # Create scatter chart data
        num_points = 30
        x_values = np.random.uniform(0, 10, num_points)
        y_values = 2 * x_values + np.random.normal(0, 2, num_points)
        
        data = pd.DataFrame({'X': x_values, 'Y': y_values})
        metadata = {
            'title': 'Sample Scatter Chart',
            'x_axis_label': 'X Values',
            'y_axis_label': 'Y Values',
            'legend_items': []
        }
        
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    return data, metadata

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test Chart Analysis module')
    parser.add_argument('--chart-type', type=str, default='bar',
                       choices=['bar', 'line', 'pie', 'scatter'],
                       help='Type of chart to test')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Create chart analyzer
    analyzer = ChartAnalyzer(config)
    
    # Create sample data
    data, metadata = create_sample_data(args.chart_type)
    
    print(f"Created sample {args.chart_type} chart data:")
    print(data.head())
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Analyze chart data
    result = analyzer.analyze_chart(chart_data=data, chart_type=args.chart_type)
    
    print(f"\nAnalyzed chart data:")
    print(f"  Chart type: {result['chart_type']}")
    print(f"  Data shape: {result['data'].shape}")
    
    # Visualize chart
    print("\nVisualizing chart...")
    chart_image = analyzer.visualize_chart(
        chart_data=result['data'],
        chart_type=result['chart_type'],
        metadata=result['metadata']
    )
    
    # Convert to PIL Image and display
    from PIL import Image
    img = Image.fromarray(chart_image)
    
    # Save image
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"sample_{args.chart_type}_chart.png")
    img.save(output_path)
    
    print(f"Chart image saved to: {output_path}")
    
    # Display image
    plt.figure(figsize=(10, 6))
    plt.imshow(chart_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
