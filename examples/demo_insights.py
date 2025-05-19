#!/usr/bin/env python
"""
Demo script for Chart Insights System.
Demonstrates the end-to-end functionality with sample data.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chart_analysis import ChartAnalyzer
from src.knowledge_graph.builder import ChartKnowledgeGraphBuilder
from src.graph_rag import ChartGraphRAG
from src.insights_generation import InsightGenerator
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def demo_line_chart():
    """Demo with line chart data."""
    print("\n=== Demo: Line Chart Analysis ===\n")
    
    # Load sample data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples', 'sales_data.csv')
    data = pd.read_csv(data_path)
    
    # Create chart metadata
    metadata = {
        'title': 'Monthly Sales Data',
        'x_axis_label': 'Month',
        'y_axis_label': 'Sales ($1000s)'
    }
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data['Month'], data['Sales'], marker='o')
    plt.title(metadata['title'])
    plt.xlabel(metadata['x_axis_label'])
    plt.ylabel(metadata['y_axis_label'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Run analysis and generate insights
    config = load_config()
    insight_generator = InsightGenerator(config)
    
    insights = insight_generator.generate_insights(
        chart_data=data,
        chart_type='line',
        chart_metadata=metadata
    )
    
    # Display insights
    print(f"\nGenerated {len(insights)} insights:\n")
    for i, insight in enumerate(insights, 1):
        confidence = insight.get('confidence', 0)
        print(f"Insight {i}: {insight.get('text', '')}")
        print(f"Confidence: {confidence:.2f}")
        
        if 'explanation' in insight:
            print(f"Explanation: {insight['explanation']}")
        
        print()

def demo_bar_chart():
    """Demo with bar chart data."""
    print("\n=== Demo: Bar Chart Analysis ===\n")
    
    # Load sample data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples', 'category_data.csv')
    data = pd.read_csv(data_path)
    
    # Create chart metadata
    metadata = {
        'title': 'Sales by Category',
        'x_axis_label': 'Category',
        'y_axis_label': 'Sales ($1000s)'
    }
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.bar(data['Category'], data['Value'], color='skyblue')
    plt.title(metadata['title'])
    plt.xlabel(metadata['x_axis_label'])
    plt.ylabel(metadata['y_axis_label'])
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Run analysis and generate insights
    config = load_config()
    insight_generator = InsightGenerator(config)
    
    insights = insight_generator.generate_insights(
        chart_data=data,
        chart_type='bar',
        chart_metadata=metadata
    )
    
    # Display insights
    print(f"\nGenerated {len(insights)} insights:\n")
    for i, insight in enumerate(insights, 1):
        confidence = insight.get('confidence', 0)
        print(f"Insight {i}: {insight.get('text', '')}")
        print(f"Confidence: {confidence:.2f}")
        
        if 'explanation' in insight:
            print(f"Explanation: {insight['explanation']}")
        
        print()

def demo_correlation():
    """Demo with correlation data."""
    print("\n=== Demo: Correlation Analysis ===\n")
    
    # Load sample data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples', 'correlation_data.csv')
    data = pd.read_csv(data_path)
    
    # Create chart metadata
    metadata = {
        'title': 'Ice Cream Sales vs Temperature',
        'x_axis_label': 'Temperature (°C)',
        'y_axis_label': 'Ice Cream Sales ($)'
    }
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Temperature'], data['IceCreamSales'], alpha=0.7)
    plt.title(metadata['title'])
    plt.xlabel(metadata['x_axis_label'])
    plt.ylabel(metadata['y_axis_label'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Run analysis and generate insights
    config = load_config()
    insight_generator = InsightGenerator(config)
    
    insights = insight_generator.generate_insights(
        chart_data=data,
        chart_type='scatter',
        chart_metadata=metadata
    )
    
    # Display insights
    print(f"\nGenerated {len(insights)} insights:\n")
    for i, insight in enumerate(insights, 1):
        confidence = insight.get('confidence', 0)
        print(f"Insight {i}: {insight.get('text', '')}")
        print(f"Confidence: {confidence:.2f}")
        
        if 'explanation' in insight:
            print(f"Explanation: {insight['explanation']}")
        
        print()

def visualize_knowledge_graph(chart_type):
    """Visualize knowledge graph for a chart type."""
    print(f"\n=== Knowledge Graph Visualization: {chart_type.capitalize()} Chart ===\n")
    
    # Load sample data
    if chart_type == 'line':
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples', 'sales_data.csv')
        title = 'Monthly Sales Data'
    elif chart_type == 'bar':
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples', 'category_data.csv')
        title = 'Sales by Category'
    elif chart_type == 'scatter':
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples', 'correlation_data.csv')
        title = 'Ice Cream Sales vs Temperature'
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    data = pd.read_csv(data_path)
    
    # Create chart metadata
    metadata = {'title': title}
    
    # Generate knowledge graph
    config = load_config()
    kg_builder = ChartKnowledgeGraphBuilder(config)
    
    graph = kg_builder.build_graph(
        chart_data=data,
        chart_type=chart_type,
        chart_metadata=metadata
    )
    
    # Print graph statistics
    print(f"Knowledge graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Count node types
    node_types = {}
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode types:")
    for node_type, count in node_types.items():
        print(f"  - {node_type}: {count}")
    
    # Count edge types
    edge_types = {}
    for _, _, attrs in graph.edges(data=True):
        edge_type = attrs.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("\nEdge types:")
    for edge_type, count in edge_types.items():
        print(f"  - {edge_type}: {count}")
    
    # Visualize graph
    import networkx as nx
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Create layout
    pos = nx.spring_layout(graph, k=0.3, iterations=50)
    
    # Node colors based on type
    node_color_map = {
        'chart': 'royalblue',
        'category': 'green',
        'segment': 'green',
        'data_point': 'orange',
        'series': 'red',
        'statistics': 'purple',
        'statistic': 'mediumpurple',
        'value': 'cyan'
    }
    
    node_colors = [node_color_map.get(graph.nodes[node].get('type', 'unknown'), 'gray') for node in graph.nodes()]
    
    # Draw graph
    nx.draw(graph, pos, with_labels=False, node_color=node_colors, 
            node_size=100, alpha=0.8, linewidths=0.5, width=0.5)
    
    # Add labels to important nodes
    labels = {}
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', '')
        if node_type in ['chart', 'statistics']:
            labels[node] = node_type
        elif node_type in ['category', 'segment']:
            labels[node] = graph.nodes[node].get('name', node)[:10]
    
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    
    plt.title(f"Knowledge Graph for {chart_type.capitalize()} Chart")
    plt.tight_layout()
    plt.show()

def main():
    """Run all demos."""
    print("\nChart Insights Generation System - Demo\n")
    print("This demo will show the system's capabilities with different chart types.\n")
    
    # Ask the user which demo to run
    print("Available demos:")
    print("1. Line Chart Analysis")
    print("2. Bar Chart Analysis")
    print("3. Correlation Analysis")
    print("4. Knowledge Graph Visualization")
    print("5. Run All Demos")
    
    choice = input("\nSelect a demo (1-5): ")
    
    if choice == '1':
        demo_line_chart()
    elif choice == '2':
        demo_bar_chart()
    elif choice == '3':
        demo_correlation()
    elif choice == '4':
        chart_type = input("Select chart type (line, bar, scatter): ")
        if chart_type in ['line', 'bar', 'scatter']:
            visualize_knowledge_graph(chart_type)
        else:
            print("Invalid chart type!")
    elif choice == '5':
        demo_line_chart()
        demo_bar_chart()
        demo_correlation()
        visualize_knowledge_graph('line')
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
