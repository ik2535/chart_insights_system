"""
Insight Generator for Chart Insights System.
Coordinates insight generation from chart data.
"""

import logging
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Union

from ..knowledge_graph.builder import ChartKnowledgeGraphBuilder
from ..graph_rag.graph_rag_engine import ChartGraphRAG

logger = logging.getLogger(__name__)

class InsightGenerator:
    """
    Generates insights from chart data.
    
    This class coordinates the end-to-end process of analyzing chart data,
    building knowledge graphs, and generating insights using GraphRAG.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the insight generator.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.kg_builder = ChartKnowledgeGraphBuilder(config)
        self.graph_rag = ChartGraphRAG(config)
        
        # Get insight settings
        self.insight_types = config.get('insights', {}).get('types', 
                                       ['trend', 'comparison', 'anomaly', 'correlation'])
        self.max_insights = config.get('insights', {}).get('max_insights_per_chart', 5)
    
    def generate_insights(self, 
                         chart_data: pd.DataFrame, 
                         chart_type: str, 
                         chart_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate insights from chart data.
        
        Args:
            chart_data: DataFrame containing chart data
            chart_type: Type of chart (bar, line, pie, etc.)
            chart_metadata: Metadata about the chart
            
        Returns:
            List of insights
        """
        logger.info(f"Generating insights for {chart_type} chart: {chart_metadata.get('title', 'Untitled')}")
        
        # Validate inputs
        if chart_data is None or chart_data.empty:
            logger.error("No chart data provided")
            return []
        
        if chart_type not in ['bar', 'line', 'pie', 'scatter']:
            logger.warning(f"Unsupported chart type: {chart_type}. Using 'bar' as fallback.")
            chart_type = 'bar'
        
        # Build knowledge graph
        try:
            graph = self.kg_builder.build_graph(chart_data, chart_type, chart_metadata)
            logger.info(f"Knowledge graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return []
        
        # Generate insights using GraphRAG
        try:
            insights = self.graph_rag.analyze_chart(chart_data, chart_type, chart_metadata)
            logger.info(f"Generated {len(insights)} insights")
            return insights
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []
    
    def format_insights(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format insights for display.
        
        Args:
            insights: List of raw insights
            
        Returns:
            Formatted insights dictionary
        """
        # Group insights by type
        grouped_insights = {}
        
        for insight in insights:
            insight_type = insight.get('type', 'general')
            
            if insight_type not in grouped_insights:
                grouped_insights[insight_type] = []
            
            grouped_insights[insight_type].append(insight)
        
        # Create formatted result
        result = {
            'summary': self._create_summary(insights),
            'grouped_insights': grouped_insights,
            'all_insights': insights
        }
        
        return result
    
    def _create_summary(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of insights.
        
        Args:
            insights: List of insights
            
        Returns:
            Summary dictionary
        """
        # Count insights by type
        type_counts = {}
        for insight in insights:
            insight_type = insight.get('type', 'general')
            type_counts[insight_type] = type_counts.get(insight_type, 0) + 1
        
        # Get highest confidence insight
        highest_confidence = None
        if insights:
            highest_confidence = max(insights, key=lambda x: x.get('confidence', 0))
        
        # Create summary
        summary = {
            'total_insights': len(insights),
            'type_counts': type_counts,
            'highest_confidence': highest_confidence
        }
        
        return summary
    
    def get_recommended_insight_types(self, chart_type: str) -> List[str]:
        """
        Get recommended insight types for a chart type.
        
        Args:
            chart_type: Type of chart
            
        Returns:
            List of recommended insight types
        """
        # Different chart types are better suited for different insights
        if chart_type == 'line':
            return ['trend', 'anomaly', 'correlation']
        elif chart_type == 'bar':
            return ['comparison', 'trend', 'anomaly']
        elif chart_type == 'pie':
            return ['comparison']
        elif chart_type == 'scatter':
            return ['correlation', 'anomaly']
        else:
            return ['trend', 'comparison', 'anomaly', 'correlation']
