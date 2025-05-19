"""
Base builder for knowledge graphs.
"""

import logging
import networkx as nx
import pandas as pd
import uuid
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class BaseGraphBuilder:
    """
    Base class for knowledge graph builders.
    
    This class provides common functionality for all chart type graph builders
    and defines the interface that specific builders should implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base graph builder.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.relationship_threshold = config.get('knowledge_graph', {}).get('relationship_threshold', 0.5)
    
    def build_graph(self, 
                   chart_data: pd.DataFrame, 
                   chart_type: str,
                   chart_metadata: Dict[str, Any]) -> nx.DiGraph:
        """
        Build knowledge graph from chart data.
        
        Args:
            chart_data: DataFrame containing chart data
            chart_type: Type of chart (bar, line, pie, etc.)
            chart_metadata: Metadata about the chart
            
        Returns:
            NetworkX DiGraph representing the knowledge graph
        """
        # Initialize graph
        graph = nx.DiGraph()
        
        # Add chart node as root
        chart_id = self._generate_id('chart')
        graph.add_node(chart_id, 
                      type='chart', 
                      chart_type=chart_type, 
                      title=chart_metadata.get('title', 'Untitled'),
                      x_label=chart_metadata.get('x_axis_label', ''),
                      y_label=chart_metadata.get('y_axis_label', ''))
        
        # Add chart data based on chart type
        graph = self._add_chart_data(graph, chart_id, chart_data, chart_metadata)
        
        # Add metadata nodes
        graph = self._add_metadata_nodes(graph, chart_id, chart_metadata)
        
        # Add statistical insights
        graph = self._add_statistical_insights(graph, chart_id, chart_data, chart_metadata)
        
        # Add relationships between nodes
        graph = self._add_relationships(graph, chart_id, chart_data, chart_metadata)
        
        return graph
    
    def _generate_id(self, prefix: str) -> str:
        """
        Generate unique ID for graph node.
        
        Args:
            prefix: Prefix for the ID
            
        Returns:
            Unique ID string
        """
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _add_chart_data(self, 
                       graph: nx.DiGraph,
                       chart_id: str,
                       chart_data: pd.DataFrame,
                       chart_metadata: Dict[str, Any]) -> nx.DiGraph:
        """
        Add chart data to graph.
        
        Args:
            graph: Existing graph
            chart_id: ID of chart node
            chart_data: DataFrame containing chart data
            chart_metadata: Metadata about the chart
            
        Returns:
            Updated graph
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _add_chart_data")
    
    def _add_metadata_nodes(self, 
                          graph: nx.DiGraph,
                          chart_id: str,
                          chart_metadata: Dict[str, Any]) -> nx.DiGraph:
        """
        Add metadata nodes to graph.
        
        Args:
            graph: Existing graph
            chart_id: ID of chart node
            chart_metadata: Metadata about the chart
            
        Returns:
            Updated graph
        """
        # Add metadata container node
        metadata_id = self._generate_id('metadata')
        graph.add_node(metadata_id, type='metadata')
        graph.add_edge(chart_id, metadata_id, type='HAS_METADATA')
        
        # Add title node if present
        if chart_metadata.get('title'):
            title_id = self._generate_id('title')
            graph.add_node(title_id, 
                          type='title', 
                          value=chart_metadata['title'])
            graph.add_edge(metadata_id, title_id, type='HAS_TITLE')
        
        # Add axis label nodes if present
        if chart_metadata.get('x_axis_label'):
            x_label_id = self._generate_id('x_label')
            graph.add_node(x_label_id, 
                          type='axis_label', 
                          axis='x',
                          value=chart_metadata['x_axis_label'])
            graph.add_edge(metadata_id, x_label_id, type='HAS_X_LABEL')
        
        if chart_metadata.get('y_axis_label'):
            y_label_id = self._generate_id('y_label')
            graph.add_node(y_label_id, 
                          type='axis_label', 
                          axis='y',
                          value=chart_metadata['y_axis_label'])
            graph.add_edge(metadata_id, y_label_id, type='HAS_Y_LABEL')
        
        # Add legend items if present
        if chart_metadata.get('legend_items'):
            for i, item in enumerate(chart_metadata['legend_items']):
                legend_id = self._generate_id('legend')
                graph.add_node(legend_id, 
                              type='legend_item',
                              index=i,
                              value=item)
                graph.add_edge(metadata_id, legend_id, type='HAS_LEGEND_ITEM')
        
        return graph
    
    def _add_statistical_insights(self, 
                                graph: nx.DiGraph,
                                chart_id: str,
                                chart_data: pd.DataFrame,
                                chart_metadata: Dict[str, Any]) -> nx.DiGraph:
        """
        Add statistical insights to graph.
        
        Args:
            graph: Existing graph
            chart_id: ID of chart node
            chart_data: DataFrame containing chart data
            chart_metadata: Metadata about the chart
            
        Returns:
            Updated graph
        """
        # Add insights container node
        insights_id = self._generate_id('insights')
        graph.add_node(insights_id, type='insights')
        graph.add_edge(chart_id, insights_id, type='HAS_INSIGHTS')
        
        # Extract numeric columns for analysis
        numeric_cols = chart_data.select_dtypes(include=['number']).columns
        
        # Skip if no numeric columns
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for statistical analysis")
            return graph
        
        # Add basic statistical insights for each numeric column
        for col in numeric_cols:
            # Create column node
            col_id = self._generate_id('column')
            graph.add_node(col_id, 
                          type='column',
                          name=col)
            graph.add_edge(insights_id, col_id, type='HAS_COLUMN')
            
            # Add basic statistics
            try:
                # Mean
                mean_val = chart_data[col].mean()
                mean_id = self._generate_id('stat')
                graph.add_node(mean_id, 
                              type='statistic',
                              name='mean',
                              value=mean_val)
                graph.add_edge(col_id, mean_id, type='HAS_STATISTIC')
                
                # Median
                median_val = chart_data[col].median()
                median_id = self._generate_id('stat')
                graph.add_node(median_id, 
                              type='statistic',
                              name='median',
                              value=median_val)
                graph.add_edge(col_id, median_id, type='HAS_STATISTIC')
                
                # Min
                min_val = chart_data[col].min()
                min_id = self._generate_id('stat')
                graph.add_node(min_id, 
                              type='statistic',
                              name='minimum',
                              value=min_val)
                graph.add_edge(col_id, min_id, type='HAS_STATISTIC')
                
                # Max
                max_val = chart_data[col].max()
                max_id = self._generate_id('stat')
                graph.add_node(max_id, 
                              type='statistic',
                              name='maximum',
                              value=max_val)
                graph.add_edge(col_id, max_id, type='HAS_STATISTIC')
                
                # Standard deviation
                std_val = chart_data[col].std()
                std_id = self._generate_id('stat')
                graph.add_node(std_id, 
                              type='statistic',
                              name='standard_deviation',
                              value=std_val)
                graph.add_edge(col_id, std_id, type='HAS_STATISTIC')
                
                # Range
                range_val = max_val - min_val
                range_id = self._generate_id('stat')
                graph.add_node(range_id, 
                              type='statistic',
                              name='range',
                              value=range_val)
                graph.add_edge(col_id, range_id, type='HAS_STATISTIC')
                
                # Quartiles
                q1_val = chart_data[col].quantile(0.25)
                q1_id = self._generate_id('stat')
                graph.add_node(q1_id, 
                              type='statistic',
                              name='first_quartile',
                              value=q1_val)
                graph.add_edge(col_id, q1_id, type='HAS_STATISTIC')
                
                q3_val = chart_data[col].quantile(0.75)
                q3_id = self._generate_id('stat')
                graph.add_node(q3_id, 
                              type='statistic',
                              name='third_quartile',
                              value=q3_val)
                graph.add_edge(col_id, q3_id, type='HAS_STATISTIC')
                
                # Interquartile range
                iqr_val = q3_val - q1_val
                iqr_id = self._generate_id('stat')
                graph.add_node(iqr_id, 
                              type='statistic',
                              name='interquartile_range',
                              value=iqr_val)
                graph.add_edge(col_id, iqr_id, type='HAS_STATISTIC')
                
            except Exception as e:
                logger.warning(f"Error calculating statistics for column {col}: {e}")
        
        return graph
    
    def _add_relationships(self, 
                          graph: nx.DiGraph,
                          chart_id: str,
                          chart_data: pd.DataFrame,
                          chart_metadata: Dict[str, Any]) -> nx.DiGraph:
        """
        Add relationships between nodes.
        
        Args:
            graph: Existing graph
            chart_id: ID of chart node
            chart_data: DataFrame containing chart data
            chart_metadata: Metadata about the chart
            
        Returns:
            Updated graph
        """
        # This method can be implemented by subclasses for chart-specific relationships
        return graph
