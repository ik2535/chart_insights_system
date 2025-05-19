"""
Unit tests for ChartGraphRAG
"""

import pytest
import networkx as nx
from unittest.mock import patch, MagicMock
from src.graph_rag.graph_rag_engine import ChartGraphRAG

def test_init_graphrag(test_config):
    """Test initialization of ChartGraphRAG."""
    # Initialize GraphRAG
    graph_rag = ChartGraphRAG(test_config)
    
    # Verify initialization
    assert graph_rag.config == test_config
    assert graph_rag.max_hops == test_config['graph_rag']['max_hops']
    assert graph_rag.insight_types == test_config['insights']['types']
    assert graph_rag.confidence_threshold == test_config['insights']['confidence_threshold']

@patch('src.graph_rag.graph_rag_engine.Neo4jConnector')
def test_generate_insights(mock_connector, test_config, sample_graph, mock_llm):
    """Test generating insights from a graph."""
    # Setup mock connector
    mock_instance = mock_connector.return_value
    mock_instance.load_graph.return_value = sample_graph
    
    # Create a mock for the LLM
    with patch.object(ChartGraphRAG, '_init_llm'):
        graph_rag = ChartGraphRAG(test_config)
        graph_rag.llm = mock_llm
        
        # Generate insights
        chart_id = 'test_chart_id'
        insights = graph_rag.generate_insights(chart_id)
        
        # Verify insights were generated
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Check insight structure
        for insight in insights:
            assert 'type' in insight
            assert 'text' in insight
            assert 'confidence' in insight
            assert 'explanation' in insight

@patch('src.graph_rag.graph_rag_engine.Neo4jConnector')
@patch('src.knowledge_graph.builder.ChartKnowledgeGraphBuilder')
def test_analyze_chart(mock_builder, mock_connector, test_config, sample_bar_data, sample_chart_metadata, mock_llm, sample_graph):
    """Test analyzing a chart directly."""
    # Setup mocks
    mock_builder_instance = mock_builder.return_value
    mock_builder_instance.build_graph.return_value = sample_graph
    
    mock_connector_instance = mock_connector.return_value
    mock_connector_instance.store_graph.return_value = 'test_graph_id'
    
    # Create a mock for the LLM
    with patch.object(ChartGraphRAG, '_init_llm'):
        graph_rag = ChartGraphRAG(test_config)
        graph_rag.llm = mock_llm
        
        # Mock generate_insights to return a list of insights
        graph_rag.generate_insights = MagicMock(return_value=[
            {
                'type': 'trend',
                'text': 'Test insight',
                'confidence': 0.85,
                'explanation': 'Test explanation'
            }
        ])
        
        # Analyze chart
        insights = graph_rag.analyze_chart(sample_bar_data, 'bar', sample_chart_metadata)
        
        # Verify insights were generated
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Verify correct calls were made
        mock_builder_instance.build_graph.assert_called_once()
        mock_connector_instance.store_graph.assert_called_once()
        graph_rag.generate_insights.assert_called_once()

def test_parse_insights(test_config):
    """Test parsing insights from LLM response."""
    # Initialize GraphRAG
    with patch.object(ChartGraphRAG, '_init_llm'):
        graph_rag = ChartGraphRAG(test_config)
    
    # Test parsing insights from a well-formatted response
    llm_response = """
    INSIGHT: The sales show a clear upward trend from January to August
    CONFIDENCE: 0.92
    EXPLANATION: There is a consistent increase in sales values from month 1 to month 8, with values rising from 10 to 35.
    
    INSIGHT: There is a seasonal pattern with a decline in the final quarter
    CONFIDENCE: 0.85
    EXPLANATION: After reaching a peak in month 8, values decline from month 9 to month 12, suggesting a possible seasonal effect.
    """
    
    insights = graph_rag._parse_insights(llm_response, 'trend')
    
    # Verify parsed insights
    assert len(insights) == 2
    assert insights[0]['type'] == 'trend'
    assert 'upward trend' in insights[0]['text']
    assert insights[0]['confidence'] == 0.92
    assert 'consistent increase' in insights[0]['explanation']
    
    # Test parsing with percentage confidence
    llm_response = """
    INSIGHT: Category B has the highest value
    CONFIDENCE: 85%
    EXPLANATION: Category B has a value of 25, which is higher than all other categories.
    """
    
    insights = graph_rag._parse_insights(llm_response, 'comparison')
    
    # Verify parsed insights
    assert len(insights) == 1
    assert insights[0]['type'] == 'comparison'
    assert insights[0]['confidence'] == 0.85
    
    # Test filtering by confidence threshold
    graph_rag.confidence_threshold = 0.9
    insights = graph_rag._parse_insights(llm_response, 'comparison')
    
    # Verify filtering
    assert len(insights) == 0
