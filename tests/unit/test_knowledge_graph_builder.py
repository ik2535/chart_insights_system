"""
Unit tests for ChartKnowledgeGraphBuilder
"""

import pytest
import networkx as nx
from src.knowledge_graph.builder import ChartKnowledgeGraphBuilder

def test_build_graph_bar_chart(test_config, sample_bar_data, sample_chart_metadata):
    """Test building a graph from bar chart data."""
    # Initialize the builder
    builder = ChartKnowledgeGraphBuilder(test_config)
    
    # Build the graph
    graph = builder.build_graph(sample_bar_data, 'bar', sample_chart_metadata)
    
    # Verify the graph structure
    assert isinstance(graph, nx.DiGraph)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    
    # Check for chart node
    chart_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chart']
    assert len(chart_nodes) == 1
    
    chart_node = chart_nodes[0]
    assert graph.nodes[chart_node]['chart_type'] == 'bar'
    assert graph.nodes[chart_node]['title'] == sample_chart_metadata['title']
    
    # Check for category nodes
    category_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'category']
    assert len(category_nodes) == len(sample_bar_data)
    
    # Check for value nodes
    value_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'value']
    assert len(value_nodes) == len(sample_bar_data)
    
    # Check for statistics nodes
    statistic_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'statistic']
    assert len(statistic_nodes) > 0

def test_build_graph_line_chart(test_config, sample_line_data, sample_chart_metadata):
    """Test building a graph from line chart data."""
    # Initialize the builder
    builder = ChartKnowledgeGraphBuilder(test_config)
    
    # Build the graph
    graph = builder.build_graph(sample_line_data, 'line', sample_chart_metadata)
    
    # Verify the graph structure
    assert isinstance(graph, nx.DiGraph)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    
    # Check for chart node
    chart_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chart']
    assert len(chart_nodes) == 1
    
    # Check for data point nodes
    point_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'data_point']
    assert len(point_nodes) == len(sample_line_data)
    
    # Verify the sequence of points
    for i in range(len(point_nodes) - 1):
        # There should be NEXT relationships between consecutive points
        next_edges = [e for e in graph.edges(data=True) 
                      if e[0] == point_nodes[i] and e[1] == point_nodes[i+1] and e[2].get('type') == 'NEXT']
        assert len(next_edges) > 0

def test_build_graph_pie_chart(test_config, sample_pie_data, sample_chart_metadata):
    """Test building a graph from pie chart data."""
    # Initialize the builder
    builder = ChartKnowledgeGraphBuilder(test_config)
    
    # Build the graph
    graph = builder.build_graph(sample_pie_data, 'pie', sample_chart_metadata)
    
    # Verify the graph structure
    assert isinstance(graph, nx.DiGraph)
    
    # Check for chart node
    chart_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chart']
    assert len(chart_nodes) == 1
    
    # Check for segment nodes
    segment_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'segment']
    assert len(segment_nodes) == len(sample_pie_data)
    
    # Check total percentage equals 100%
    total_percentage = sum(graph.nodes[n].get('percentage', 0) for n in segment_nodes)
    assert abs(total_percentage - 100.0) < 0.001  # Allow for floating point errors

def test_build_graph_scatter_chart(test_config, sample_scatter_data, sample_chart_metadata):
    """Test building a graph from scatter chart data."""
    # Initialize the builder
    builder = ChartKnowledgeGraphBuilder(test_config)
    
    # Build the graph
    graph = builder.build_graph(sample_scatter_data, 'scatter', sample_chart_metadata)
    
    # Verify the graph structure
    assert isinstance(graph, nx.DiGraph)
    
    # Check for chart node
    chart_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chart']
    assert len(chart_nodes) == 1
    
    # Check for data point nodes
    point_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'data_point']
    assert len(point_nodes) == len(sample_scatter_data)
    
    # Check that each point has x and y values
    for node in point_nodes:
        assert 'x' in graph.nodes[node]
        assert 'y' in graph.nodes[node]
