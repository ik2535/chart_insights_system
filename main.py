#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for Chart Insights System.

This script provides a command-line interface to the Chart Insights System,
allowing users to analyze charts from data or images and generate insights.

Usage:
    python main.py --data path/to/data.csv --chart-type bar
    python main.py --image path/to/chart.png
    python main.py --ui  # Launch Streamlit UI
"""

import os
import sys
import argparse
import yaml
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chart_insights.log')
    ]
)

logger = logging.getLogger(__name__)

# Import chart insights modules
from src.chart_analysis import ChartAnalyzer
from src.knowledge_graph.builder import ChartKnowledgeGraphBuilder
from src.graph_rag import ChartGraphRAG
from src.insights_generation import InsightGenerator

def load_config():
    """Load configuration from config file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Return default configuration
        return {
            'chart_analysis': {
                'ocr_config': '--psm 6',
                'supported_types': ['bar', 'line', 'pie', 'scatter']
            }
        }

def load_data(data_path):
    """
    Load data from file.
    
    Args:
        data_path: Path to data file
        
    Returns:
        Pandas DataFrame with data
    """
    file_ext = os.path.splitext(data_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            data = pd.read_csv(data_path)
        elif file_ext in ['.xls', '.xlsx']:
            data = pd.read_excel(data_path)
        elif file_ext == '.json':
            data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def analyze_chart(args, config):
    """
    Analyze chart and generate insights.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    # Initialize chart analyzer
    analyzer = ChartAnalyzer(config)
    
    # Process based on input type
    if args.data:
        # Load data
        data = load_data(args.data)
        
        # Analyze chart
        result = analyzer.analyze_chart(chart_data=data, chart_type=args.chart_type)
    elif args.image:
        # Analyze chart image
        result = analyzer.analyze_chart(chart_image=args.image)
    else:
        logger.error("No input data or image provided")
        return
    
    # Print analysis results
    logger.info(f"Chart type: {result['chart_type']}")
    logger.info(f"Data shape: {result['data'].shape}")
    
    # Visualize chart
    chart_image = analyzer.visualize_chart(
        chart_data=result['data'],
        chart_type=result['chart_type'],
        metadata=result['metadata']
    )
    
    # Save visualization
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'chart_analysis_output.png')
    Image.fromarray(chart_image).save(output_path)
    logger.info(f"Chart image saved to: {output_path}")
    
    # Display visualization if requested
    if args.display:
        plt.figure(figsize=(10, 6))
        plt.imshow(chart_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Generate insights using GraphRAG
    try:
        # Initialize insight generator
        insight_generator = InsightGenerator(config)
        
        # Generate insights
        insights = insight_generator.generate_insights(
            chart_data=result['data'],
            chart_type=result['chart_type'],
            chart_metadata=result['metadata']
        )
        
        # Print insights
        logger.info(f"Generated {len(insights)} insights")
        
        if insights:  # Check if insights list is not empty
            for i, insight in enumerate(insights, 1):
                confidence = insight.get('confidence', 0)
                logger.info(f"Insight {i}: {insight.get('text', '')} (confidence: {confidence:.2f})")
                
                if 'explanation' in insight:
                    logger.info(f"   Explanation: {insight['explanation']}")
        else:
            logger.info("No insights generated")
        
        # Add insights to result
        result['insights'] = insights
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        # Continue without insights
        result['insights'] = []
    
    return result

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Chart Insights System')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--data', type=str, help='Path to data file')
    input_group.add_argument('--image', type=str, help='Path to chart image')
    input_group.add_argument('--ui', action='store_true', help='Launch Streamlit UI')
    
    # Chart options
    parser.add_argument('--chart-type', type=str, choices=['bar', 'line', 'pie', 'scatter'],
                       help='Type of chart (required when using --data)')
    
    # Insight options
    parser.add_argument('--insight-types', type=str, nargs='+', 
                      choices=['trend', 'comparison', 'anomaly', 'correlation'],
                      default=['trend', 'comparison'],
                      help='Types of insights to generate')
    
    # Output options
    parser.add_argument('--display', action='store_true', help='Display visualization')
    parser.add_argument('--output', type=str, help='Path to save output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Launch UI if requested
    if args.ui:
        try:
            import streamlit
            # Get the path to the Streamlit app
            ui_path = os.path.join(os.path.dirname(__file__), 'src', 'ui', 'streamlit_app.py')
            logger.info(f"Launching Streamlit UI from: {ui_path}")
            
            # Use the streamlit CLI to run the app
            os.system(f"streamlit run {ui_path}")
            return
        except ImportError:
            logger.error("Streamlit not installed. Run 'pip install streamlit'.")
            sys.exit(1)
    
    # Validate arguments
    if args.data and not args.chart_type:
        parser.error("--chart-type is required when using --data")
    
    # Load configuration
    config = load_config()
    
    # Update config with command-line arguments
    config['insights']['types'] = args.insight_types
    
    try:
        # Analyze chart and generate insights
        result = analyze_chart(args, config)
        
        # Save results if output path provided
        if args.output and result:
            try:
                output_dir = os.path.dirname(args.output)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Determine file format
                file_ext = os.path.splitext(args.output)[1].lower()
                
                if file_ext == '.json':
                    # Save as JSON
                    with open(args.output, 'w') as f:
                        # Convert insights to JSON-compatible format
                        json_result = {
                            'chart_type': result['chart_type'],
                            'insights': result['insights'],
                            'metadata': {
                                key: value for key, value in result['metadata'].items()
                                if isinstance(value, (str, int, float, bool, list, dict))
                            }
                        }
                        json.dump(json_result, f, indent=2)
                else:
                    # Default to pickle for other formats
                    import pickle
                    with open(args.output, 'wb') as f:
                        pickle.dump(result, f)
                
                logger.info(f"Results saved to: {args.output}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
        
    except Exception as e:
        logger.error(f"Error processing chart: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
