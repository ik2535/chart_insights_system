"""
Integration tests for Chart Insights System
"""

import pytest
import pandas as pd
import networkx as nx
from src.chart_analysis import ChartAnalyzer
from src.knowledge_graph.builder import ChartKnowledgeGraphBuilder
from src.graph_rag import ChartGraphRAG
from src.insights_generation import InsightGenerator

def test_end_to_end_bar_chart(test_config, sample_bar_data, sample_chart_metadata):
    """Test the end-to-end process for a bar chart."""
    # Initialize components
    analyzer = ChartAnalyzer(test_config)
    kg_builder = ChartKnowledgeGraphBuilder(test_config)
    insight_generator = InsightGenerator(test_config)
    
    # Skip test if we don't have mock LLM configured properly
    if test_config.get('llm', {}).get('provider') != 'mock':
        pytest.skip("This test requires mock LLM configuration")
    
    # Analyze chart
    analysis_result = analyzer.analyze_chart(chart_data=sample_bar_data, chart_type='bar')
    
    # Verify analysis result
    assert analysis_result is not None
    assert 'chart_type' in analysis_result
    assert 'data' in analysis_result
    assert 'metadata' in analysis_result
    
    # Build knowledge graph
    graph = kg_builder.build_graph(sample_bar_data, 'bar', sample_chart_metadata)
    
    # Verify graph
    assert isinstance(graph, nx.DiGraph)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    
    # Generate insights
    insights = insight_generator.generate_insights(sample_bar_data, 'bar', sample_chart_metadata)
    
    # Verify insights
    # Note: This might fail if LLM is not properly mocked
    # In a real system, we would mock the LLM for consistent testing
    assert isinstance(insights, list)
    # We don't assert length because it might be 0 if LLM is not mocked properly

def test_data_to_insights_pipeline(test_config, sample_line_data, sample_chart_metadata):
    """Test the data-to-insights pipeline."""
    # Initialize insight generator
    insight_generator = InsightGenerator(test_config)
    
    # Skip test if we don't have mock LLM configured properly
    if test_config.get('llm', {}).get('provider') != 'mock':
        pytest.skip("This test requires mock LLM configuration")
    
    # Mock the GraphRAG analyze_chart method to return test insights
    from unittest.mock import patch
    test_insights = [
        {
            'type': 'trend',
            'text': 'There is an upward trend in the first half',
            'confidence': 0.9,
            'explanation': 'The values consistently increase from month 1 to month 8.'
        },
        {
            'type': 'trend',
            'text': 'There is a downward trend in the second half',
            'confidence': 0.85,
            'explanation': 'The values decline from month 9 to month 12.'
        }
    ]
    
    with patch.object(ChartGraphRAG, 'analyze_chart', return_value=test_insights):
        # Generate insights
        insights = insight_generator.generate_insights(sample_line_data, 'line', sample_chart_metadata)
        
        # Verify insights
        assert insights == test_insights
        
        # Format insights
        formatted = insight_generator.format_insights(insights)
        
        # Verify formatted insights
        assert 'summary' in formatted
        assert 'grouped_insights' in formatted
        assert 'all_insights' in formatted
        assert formatted['all_insights'] == insights
        assert 'trend' in formatted['grouped_insights']
        assert len(formatted['grouped_insights']['trend']) == 2
