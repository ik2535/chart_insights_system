"""
Bar chart knowledge graph builder.
"""

import logging
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from .base_builder import BaseGraphBuilder

logger = logging.getLogger(__name__)

class BarChartGraphBuilder(BaseGraphBuilder):
    """
    Builds knowledge graphs from bar chart data.
    """
    
    def _add_chart_data(self, 
                       graph: nx.DiGraph,
                       chart_id: str,
                       chart_data: pd.DataFrame,
                       chart_metadata: Dict[str, Any]) -> nx.DiGraph:
        """
        Add bar chart data to graph.
        
        Args:
            graph: Existing graph
            chart_id: ID of chart node
            chart_data: DataFrame containing chart data
            chart_metadata: Metadata about the chart
            
        Returns:
            Updated graph
        """
        # Identify category and value columns
        category_col = chart_data.columns[0]
        value_col = chart_data.columns[1] if len(chart_data.columns) > 1 else None
        
        if value_col is None:
            logger.warning("Bar chart data missing value column")
            return graph
        
        # Add data container node
        data_id = self._generate_id('data')
        graph.add_node(data_id, type='data')
        graph.add_edge(chart_id, data_id, type='HAS_DATA')
        
        # Add category nodes and values
        categories = {}
        for idx, row in chart_data.iterrows():
            category = str(row[category_col])
            value = float(row[value_col])
            
            # Create category node
            category_id = self._generate_id('category')
            graph.add_node(category_id, 
                          type='category', 
                          name=category,
                          value=value,
                          index=idx)
            
            # Link to data node
            graph.add_edge(data_id, category_id, type='HAS_CATEGORY')
            
            # Store category ID for relationship building
            categories[category] = category_id
        
        # Calculate total and percentages
        total = chart_data[value_col].sum()
        
        # Add percentage information
        for idx, row in chart_data.iterrows():
            category = str(row[category_col])
            value = float(row[value_col])
            category_id = categories[category]
            
            # Calculate percentage
            percentage = (value / total) * 100 if total > 0 else 0
            
            # Add percentage property to category node
            graph.nodes[category_id]['percentage'] = percentage
            
            # Create percentage node
            pct_id = self._generate_id('percentage')
            graph.add_node(pct_id, 
                          type='percentage',
                          value=percentage)
            
            # Link percentage to category
            graph.add_edge(category_id, pct_id, type='HAS_PERCENTAGE')
        
        # Find max and min categories
        max_value = chart_data[value_col].max()
        min_value = chart_data[value_col].min()
        
        max_categories = chart_data[chart_data[value_col] == max_value][category_col].tolist()
        min_categories = chart_data[chart_data[value_col] == min_value][category_col].tolist()
        
        # Add superlative nodes
        if max_categories:
            max_id = self._generate_id('max')
            graph.add_node(max_id, 
                          type='superlative',
                          name='maximum',
                          value=max_value)
            graph.add_edge(data_id, max_id, type='HAS_MAXIMUM')
            
            # Link max categories to max node
            for category in max_categories:
                if category in categories:
                    graph.add_edge(max_id, categories[category], type='REFERS_TO')
        
        if min_categories:
            min_id = self._generate_id('min')
            graph.add_node(min_id, 
                          type='superlative',
                          name='minimum',
                          value=min_value)
            graph.add_edge(data_id, min_id, type='HAS_MINIMUM')
            
            # Link min categories to min node
            for category in min_categories:
                if category in categories:
                    graph.add_edge(min_id, categories[category], type='REFERS_TO')
        
        return graph
    
    def _add_relationships(self, 
                          graph: nx.DiGraph,
                          chart_id: str,
                          chart_data: pd.DataFrame,
                          chart_metadata: Dict[str, Any]) -> nx.DiGraph:
        """
        Add relationships between nodes for bar charts.
        
        Args:
            graph: Existing graph
            chart_id: ID of chart node
            chart_data: DataFrame containing chart data
            chart_metadata: Metadata about the chart
            
        Returns:
            Updated graph
        """
        # Find data node
        data_nodes = [n for n, d in graph.nodes(data=True) 
                      if d.get('type') == 'data' and (chart_id, n) in graph.edges()]
        
        if not data_nodes:
            return graph
        
        data_id = data_nodes[0]
        
        # Find category nodes
        category_nodes = [n for n, d in graph.nodes(data=True) 
                         if d.get('type') == 'category' and (data_id, n) in graph.edges()]
        
        # Add comparison relationships between categories
        for i, cat1_id in enumerate(category_nodes):
            cat1_value = graph.nodes[cat1_id].get('value', 0)
            
            for j in range(i+1, len(category_nodes)):
                cat2_id = category_nodes[j]
                cat2_value = graph.nodes[cat2_id].get('value', 0)
                
                # Add comparison relationship
                if cat1_value > cat2_value:
                    graph.add_edge(cat1_id, cat2_id, 
                                  type='GREATER_THAN', 
                                  difference=cat1_value - cat2_value)
                else:
                    graph.add_edge(cat2_id, cat1_id, 
                                  type='GREATER_THAN', 
                                  difference=cat2_value - cat1_value)
        
        # Add average comparison relationships
        if len(category_nodes) > 0:
            # Calculate average
            avg_value = sum([graph.nodes[n].get('value', 0) for n in category_nodes]) / len(category_nodes)
            
            # Create average node
            avg_id = self._generate_id('average')
            graph.add_node(avg_id, 
                          type='statistic',
                          name='average',
                          value=avg_value)
            graph.add_edge(data_id, avg_id, type='HAS_AVERAGE')
            
            # Add relationship to categories
            for cat_id in category_nodes:
                cat_value = graph.nodes[cat_id].get('value', 0)
                
                if cat_value > avg_value:
                    graph.add_edge(cat_id, avg_id, 
                                  type='ABOVE_AVERAGE', 
                                  difference=cat_value - avg_value)
                else:
                    graph.add_edge(cat_id, avg_id, 
                                  type='BELOW_AVERAGE', 
                                  difference=avg_value - cat_value)
        
        return graph
