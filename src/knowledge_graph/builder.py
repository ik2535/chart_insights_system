"""
Knowledge Graph Builder for Chart Insights System.
Creates knowledge graphs from chart data.
"""

import networkx as nx
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ChartKnowledgeGraphBuilder:
    """
    Builds knowledge graphs from chart data.
    
    This class takes chart data and metadata to create a knowledge graph
    representing entities and relationships within the chart.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the knowledge graph builder.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.relationship_threshold = config.get('knowledge_graph', {}).get('relationship_threshold', 0.5)
        self.schema_validation = config.get('knowledge_graph', {}).get('schema_validation', True)
    
    def build_graph(self, 
                    chart_data: pd.DataFrame, 
                    chart_type: str, 
                    chart_metadata: Dict[str, Any]) -> nx.DiGraph:
        """
        Build a knowledge graph from chart data.
        
        Args:
            chart_data: DataFrame containing chart data
            chart_type: Type of chart (bar, line, pie, etc.)
            chart_metadata: Metadata about the chart (title, labels, etc.)
            
        Returns:
            NetworkX DiGraph representing the knowledge graph
        """
        logger.info(f"Building knowledge graph for {chart_type} chart: {chart_metadata.get('title', 'Untitled')}")
        
        # Initialize graph
        graph = nx.DiGraph()
        
        # Add chart node
        chart_id = self._generate_id('chart')
        graph.add_node(chart_id, 
                      type='chart', 
                      chart_type=chart_type, 
                      title=chart_metadata.get('title', 'Untitled'),
                      x_label=chart_metadata.get('x_axis_label', ''),
                      y_label=chart_metadata.get('y_axis_label', ''))
        
        # Process based on chart type
        if chart_type == 'bar':
            self._process_bar_chart(graph, chart_id, chart_data, chart_metadata)
        elif chart_type == 'line':
            self._process_line_chart(graph, chart_id, chart_data, chart_metadata)
        elif chart_type == 'pie':
            self._process_pie_chart(graph, chart_id, chart_data, chart_metadata)
        elif chart_type == 'scatter':
            self._process_scatter_chart(graph, chart_id, chart_data, chart_metadata)
        else:
            logger.warning(f"Unsupported chart type: {chart_type}. Building generic graph.")
            self._process_generic_chart(graph, chart_id, chart_data, chart_metadata)
        
        # Add statistical nodes and relationships
        self._add_statistics(graph, chart_id, chart_data, chart_metadata)
        
        # Validate graph if enabled
        if self.schema_validation:
            self._validate_graph(graph)
        
        logger.info(f"Knowledge graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with prefix"""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _process_bar_chart(self, 
                           graph: nx.DiGraph, 
                           chart_id: str, 
                           data: pd.DataFrame, 
                           metadata: Dict[str, Any]) -> None:
        """Process bar chart data and add to graph"""
        # Identify category and value columns
        category_col = data.columns[0]
        value_col = data.columns[1] if len(data.columns) > 1 else None
        
        if value_col is None:
            logger.warning("Bar chart data missing value column")
            return
        
        # Add category and value nodes
        for idx, row in data.iterrows():
            category = str(row[category_col])
            value = float(row[value_col])
            
            # Add category node
            category_id = self._generate_id('category')
            graph.add_node(category_id, 
                          type='category', 
                          name=category,
                          value=value,
                          index=idx)
            
            # Link to chart
            graph.add_edge(chart_id, category_id, type='HAS_CATEGORY')
            
            # Add value node
            value_id = self._generate_id('value')
            graph.add_node(value_id,
                          type='value',
                          value=value,
                          name=f"{value}")
            
            # Link category to value
            graph.add_edge(category_id, value_id, type='HAS_VALUE')
    
    def _process_line_chart(self, 
                            graph: nx.DiGraph, 
                            chart_id: str, 
                            data: pd.DataFrame, 
                            metadata: Dict[str, Any]) -> None:
        """Process line chart data and add to graph"""
        # Identify X and Y columns
        x_col = data.columns[0]
        
        # Check if we have multiple series
        if len(data.columns) == 2:
            # Single series
            y_col = data.columns[1]
            self._add_time_series(graph, chart_id, data, x_col, y_col)
        else:
            # Multiple series
            for i in range(1, len(data.columns)):
                y_col = data.columns[i]
                series_id = self._generate_id('series')
                series_name = y_col
                
                # Add series node
                graph.add_node(series_id,
                              type='series',
                              name=series_name)
                
                # Link series to chart
                graph.add_edge(chart_id, series_id, type='HAS_SERIES')
                
                # Add time series for this series
                self._add_time_series(graph, series_id, data, x_col, y_col)
    
    def _add_time_series(self, 
                         graph: nx.DiGraph, 
                         parent_id: str, 
                         data: pd.DataFrame, 
                         x_col: str, 
                         y_col: str) -> None:
        """Add time series data to graph"""
        # Sort data by x column if it's time-based
        try:
            data = data.sort_values(by=x_col)
        except:
            # If sorting fails, use the data as is
            pass
        
        # Add data points
        prev_point_id = None
        for idx, row in data.iterrows():
            x_value = row[x_col]
            y_value = row[y_col]
            
            # Add data point node
            point_id = self._generate_id('point')
            graph.add_node(point_id,
                          type='data_point',
                          x=x_value,
                          y=y_value,
                          index=idx)
            
            # Link to parent
            graph.add_edge(parent_id, point_id, type='HAS_POINT')
            
            # Link to previous point (for sequence)
            if prev_point_id is not None:
                # Calculate change
                prev_y = graph.nodes[prev_point_id].get('y', 0)
                change = y_value - prev_y
                change_pct = (change / prev_y) * 100 if prev_y != 0 else 0
                
                graph.add_edge(prev_point_id, point_id, 
                              type='NEXT',
                              change=change,
                              change_pct=change_pct)
            
            prev_point_id = point_id
    
    def _process_pie_chart(self, 
                           graph: nx.DiGraph, 
                           chart_id: str, 
                           data: pd.DataFrame, 
                           metadata: Dict[str, Any]) -> None:
        """Process pie chart data and add to graph"""
        # Identify label and value columns
        label_col = data.columns[0]
        value_col = data.columns[1] if len(data.columns) > 1 else None
        
        if value_col is None:
            logger.warning("Pie chart data missing value column")
            return
        
        # Calculate total for percentages
        total = data[value_col].sum()
        
        # Track segment nodes for comparison relationships
        segment_nodes = []
        
        # Add segment nodes
        for idx, row in data.iterrows():
            label = str(row[label_col])
            value = float(row[value_col])
            percentage = (value / total) * 100 if total != 0 else 0
            
            # Add segment node
            segment_id = self._generate_id('segment')
            graph.add_node(segment_id,
                          type='segment',
                          name=label,
                          value=value,
                          percentage=percentage,
                          index=idx)
            
            # Link to chart
            graph.add_edge(chart_id, segment_id, type='HAS_SEGMENT')
            
            # Add to segment nodes list
            segment_nodes.append(segment_id)
            
            # Add relationships between segments (for comparison)
            # Use the segment_nodes list instead of predecessors
            for prev_idx in range(len(segment_nodes) - 1):
                prev_node_id = segment_nodes[prev_idx]
                prev_value = graph.nodes[prev_node_id].get('value', 0)
                
                if value > prev_value:
                    graph.add_edge(segment_id, prev_node_id, type='GREATER_THAN')
                else:
                    graph.add_edge(prev_node_id, segment_id, type='GREATER_THAN')
    
    def _process_scatter_chart(self, 
                              graph: nx.DiGraph, 
                              chart_id: str, 
                              data: pd.DataFrame, 
                              metadata: Dict[str, Any]) -> None:
        """Process scatter chart data and add to graph"""
        # Identify X and Y columns
        x_col = data.columns[0]
        y_col = data.columns[1] if len(data.columns) > 1 else None
        
        if y_col is None:
            logger.warning("Scatter chart data missing Y column")
            return
        
        # Optional color/size columns
        color_col = data.columns[2] if len(data.columns) > 2 else None
        size_col = data.columns[3] if len(data.columns) > 3 else None
        
        # Add data points
        for idx, row in data.iterrows():
            x_value = float(row[x_col])
            y_value = float(row[y_col])
            
            # Additional properties
            properties = {
                'type': 'data_point',
                'x': x_value,
                'y': y_value,
                'index': idx
            }
            
            if color_col:
                properties['color'] = row[color_col]
            
            if size_col:
                properties['size'] = row[size_col]
            
            # Add point node
            point_id = self._generate_id('point')
            graph.add_node(point_id, **properties)
            
            # Link to chart
            graph.add_edge(chart_id, point_id, type='HAS_POINT')
    
    def _process_generic_chart(self, 
                              graph: nx.DiGraph, 
                              chart_id: str, 
                              data: pd.DataFrame, 
                              metadata: Dict[str, Any]) -> None:
        """Process generic chart data and add to graph"""
        # Add columns as attributes
        for col in data.columns:
            attr_id = self._generate_id('attribute')
            graph.add_node(attr_id,
                          type='attribute',
                          name=col)
            
            # Link to chart
            graph.add_edge(chart_id, attr_id, type='HAS_ATTRIBUTE')
        
        # Add each row as a data point
        for idx, row in data.iterrows():
            point_id = self._generate_id('data_point')
            point_props = {'type': 'data_point', 'index': idx}
            
            # Add all values as properties
            for col in data.columns:
                point_props[col] = row[col]
            
            # Add data point node
            graph.add_node(point_id, **point_props)
            
            # Link to chart
            graph.add_edge(chart_id, point_id, type='HAS_DATA_POINT')
    
    def _add_statistics(self, 
                       graph: nx.DiGraph, 
                       chart_id: str, 
                       data: pd.DataFrame, 
                       metadata: Dict[str, Any]) -> None:
        """Add statistical nodes and relationships to the graph"""
        # Add a stats container node
        stats_id = self._generate_id('stats')
        graph.add_node(stats_id, type='statistics')
        graph.add_edge(chart_id, stats_id, type='HAS_STATISTICS')
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            # Skip index column if exists
            if col == 'index':
                continue
                
            # Basic statistics
            try:
                mean_val = data[col].mean()
                median_val = data[col].median()
                min_val = data[col].min()
                max_val = data[col].max()
                std_val = data[col].std()
                
                # Add statistic nodes
                mean_id = self._generate_id('mean')
                graph.add_node(mean_id, 
                              type='statistic', 
                              name='mean',
                              column=col,
                              value=mean_val)
                graph.add_edge(stats_id, mean_id, type='HAS_STATISTIC')
                
                median_id = self._generate_id('median')
                graph.add_node(median_id,
                              type='statistic',
                              name='median',
                              column=col,
                              value=median_val)
                graph.add_edge(stats_id, median_id, type='HAS_STATISTIC')
                
                min_id = self._generate_id('min')
                graph.add_node(min_id,
                              type='statistic',
                              name='minimum',
                              column=col,
                              value=min_val)
                graph.add_edge(stats_id, min_id, type='HAS_STATISTIC')
                
                max_id = self._generate_id('max')
                graph.add_node(max_id,
                              type='statistic',
                              name='maximum',
                              column=col,
                              value=max_val)
                graph.add_edge(stats_id, max_id, type='HAS_STATISTIC')
                
                std_id = self._generate_id('std')
                graph.add_node(std_id,
                              type='statistic',
                              name='standard_deviation',
                              column=col,
                              value=std_val)
                graph.add_edge(stats_id, std_id, type='HAS_STATISTIC')
                
                # Add relationships to data points
                for point_id in graph.predecessors(chart_id):
                    if graph.nodes[point_id].get('type') in ['category', 'data_point', 'segment']:
                        point_value = graph.nodes[point_id].get('value', None)
                        
                        if point_value is not None:
                            # Relationship to mean
                            if point_value > mean_val:
                                graph.add_edge(point_id, mean_id, type='ABOVE_MEAN')
                            else:
                                graph.add_edge(point_id, mean_id, type='BELOW_MEAN')
                            
                            # Relationship to max/min
                            if point_value == max_val:
                                graph.add_edge(point_id, max_id, type='IS_MAX')
                            
                            if point_value == min_val:
                                graph.add_edge(point_id, min_id, type='IS_MIN')
                
            except Exception as e:
                logger.warning(f"Error calculating statistics for column {col}: {e}")
    
    def _validate_graph(self, graph: nx.DiGraph) -> None:
        """Validate the graph structure against schema"""
        # Check that we have at least one chart node
        chart_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chart']
        if not chart_nodes:
            logger.warning("Validation warning: No chart nodes found in graph")
        
        # Check that chart node has required attributes
        for node in chart_nodes:
            attrs = graph.nodes[node]
            if 'title' not in attrs:
                logger.warning(f"Validation warning: Chart node {node} missing 'title' attribute")
            if 'chart_type' not in attrs:
                logger.warning(f"Validation warning: Chart node {node} missing 'chart_type' attribute")
        
        # Check for orphaned nodes
        orphaned_nodes = [n for n in graph.nodes() if graph.degree(n) == 0]
        for node in orphaned_nodes:
            logger.warning(f"Validation warning: Orphaned node found: {node}")
        
        # Check for data nodes without type attribute
        untyped_nodes = [n for n, d in graph.nodes(data=True) if 'type' not in d]
        for node in untyped_nodes:
            logger.warning(f"Validation warning: Node missing 'type' attribute: {node}")
        
        # Check for edges without type attribute
        untyped_edges = [(u, v) for u, v, d in graph.edges(data=True) if 'type' not in d]
        for edge in untyped_edges:
            logger.warning(f"Validation warning: Edge missing 'type' attribute: {edge}")
    
    def save_graph(self, graph: nx.DiGraph, path: str) -> None:
        """Save the graph to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(graph, f)
        logger.info(f"Graph saved to {path}")
    
    def load_graph(self, path: str) -> nx.DiGraph:
        """Load a graph from disk"""
        import pickle
        with open(path, 'rb') as f:
            graph = pickle.load(f)
        logger.info(f"Graph loaded from {path}")
        return graph
