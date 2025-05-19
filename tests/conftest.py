"""
PyTest configuration file for Chart Insights System tests.
"""

import os
import sys
import pytest
import yaml
import networkx as nx
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    config = {
        'graph_db': {
            'provider': 'memory',  # Use in-memory database for tests
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'password',
            'database': 'chart_insights_test',
            'require_connection': False
        },
        'llm': {
            'provider': 'mock',  # Use mock LLM for tests
            'model': 'test-model',
            'api_key': 'test-key'
        },
        'chart_analysis': {
            'ocr_config': '--psm 6',
            'supported_types': ['bar', 'line', 'pie', 'scatter']
        },
        'insights': {
            'types': ['trend', 'comparison', 'anomaly', 'correlation'],
            'confidence_threshold': 0.5,
            'max_insights_per_chart': 5
        },
        'graph_rag': {
            'max_hops': 3,
            'context_window': 5,
            'query_strategies': ['direct_match', 'similarity_match', 'multi_hop']
        },
        'knowledge_graph': {
            'relationship_threshold': 0.5,
            'schema_validation': True
        }
    }
    return config

@pytest.fixture
def sample_bar_data():
    """Fixture providing sample bar chart data."""
    return pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D', 'E'],
        'Value': [10, 25, 15, 30, 20]
    })

@pytest.fixture
def sample_line_data():
    """Fixture providing sample line chart data."""
    return pd.DataFrame({
        'Month': list(range(1, 13)),
        'Sales': [10, 15, 13, 17, 20, 25, 30, 35, 30, 25, 20, 15]
    })

@pytest.fixture
def sample_pie_data():
    """Fixture providing sample pie chart data."""
    return pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Value': [30, 20, 25, 25]
    })

@pytest.fixture
def sample_scatter_data():
    """Fixture providing sample scatter chart data."""
    np.random.seed(42)
    x = np.random.rand(50) * 10
    y = 2 * x + np.random.randn(50) * 2
    return pd.DataFrame({
        'X': x,
        'Y': y
    })

@pytest.fixture
def sample_chart_metadata():
    """Fixture providing sample chart metadata."""
    return {
        'title': 'Test Chart',
        'x_axis_label': 'X Axis',
        'y_axis_label': 'Y Axis'
    }

@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM."""
    mock = MagicMock()
    mock.generate.return_value = """
    INSIGHT: This is a test insight
    CONFIDENCE: 0.85
    EXPLANATION: This is a test explanation for the insight.
    
    INSIGHT: This is another test insight
    CONFIDENCE: 0.75
    EXPLANATION: This is another explanation.
    """
    return mock

@pytest.fixture
def sample_graph():
    """Fixture providing a sample graph for testing."""
    graph = nx.DiGraph()
    
    # Add chart node
    graph.add_node('chart_1', type='chart', chart_type='bar', title='Test Chart')
    
    # Add category nodes
    for i, category in enumerate(['A', 'B', 'C']):
        node_id = f'category_{i}'
        graph.add_node(node_id, type='category', name=category, value=10 + i*5, index=i)
        graph.add_edge('chart_1', node_id, type='HAS_CATEGORY')
        
        # Add value node for each category
        val_id = f'value_{i}'
        graph.add_node(val_id, type='value', value=10 + i*5, name=str(10 + i*5))
        graph.add_edge(node_id, val_id, type='HAS_VALUE')
    
    # Add statistics node
    stats_id = 'stats_1'
    graph.add_node(stats_id, type='statistics')
    graph.add_edge('chart_1', stats_id, type='HAS_STATISTICS')
    
    # Add statistic nodes
    stats = {
        'mean': 15,
        'median': 15,
        'min': 10,
        'max': 20,
        'std': 5
    }
    
    for stat_name, stat_value in stats.items():
        stat_id = f'{stat_name}_1'
        graph.add_node(stat_id, type='statistic', name=stat_name, value=stat_value, column='Value')
        graph.add_edge(stats_id, stat_id, type='HAS_STATISTIC')
    
    return graph
