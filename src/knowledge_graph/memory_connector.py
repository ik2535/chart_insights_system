"""
In-Memory Graph Connector for Chart Insights System.
Provides a fallback when Neo4j is not available.
"""

import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# Global in-memory storage for graphs across instances
GLOBAL_GRAPH_STORAGE = {}

class MemoryGraphConnector:
    """
    In-memory graph storage using NetworkX.
    Implements the same interface as Neo4jConnector for compatibility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize in-memory graph connector.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        # Use global storage to share graphs between instances
        self.graphs = GLOBAL_GRAPH_STORAGE
        logger.info("Using in-memory graph storage (Neo4j alternative)")
    
    def _connect(self) -> None:
        """No connection needed for in-memory storage"""
        pass
    
    def close(self) -> None:
        """No connection to close for in-memory storage"""
        pass
    
    def store_graph(self, graph: nx.DiGraph, chart_id: str) -> str:
        """
        Store a NetworkX graph in memory.
        
        Args:
            graph: NetworkX DiGraph to store
            chart_id: Unique identifier for the chart
            
        Returns:
            Identifier for the stored graph
        """
        # Create a database identifier for the graph
        db_id = f"graph_{chart_id}"
        
        # Store a copy of the graph in the global storage
        self.graphs[db_id] = graph.copy()
        
        logger.info(f"Graph stored in memory with ID: {db_id} (Storage size: {len(self.graphs)})")
        return db_id
    
    def load_graph(self, graph_id: str) -> nx.DiGraph:
        """
        Load a graph from memory.
        
        Args:
            graph_id: Identifier for the graph
            
        Returns:
            NetworkX DiGraph loaded from memory
        """
        # Check if graph_id is in storage
        if graph_id not in self.graphs:
            # Log available graph IDs for debugging
            available_ids = list(self.graphs.keys())
            logger.error(f"Graph ID not found: {graph_id}. Available IDs: {available_ids}")
            raise ValueError(f"Graph ID not found: {graph_id}")
        
        # Return a copy of the stored graph
        graph = self.graphs[graph_id].copy()
        
        logger.info(f"Graph loaded from memory (ID: {graph_id}) with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    def get_chart_ids(self) -> List[str]:
        """
        Get list of all chart IDs in memory.
        
        Returns:
            List of chart IDs
        """
        return list(self.graphs.keys())
    
    def execute_query(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a simplified query against the in-memory graph.
        This is a very limited implementation that only supports basic operations.
        
        Args:
            cypher_query: Cypher query string (ignored in this implementation)
            params: Query parameters (graph_id is required)
            
        Returns:
            List of results as dictionaries
        """
        if params is None or 'graph_id' not in params:
            logger.warning("No graph_id provided in params")
            return []
        
        graph_id = params['graph_id']
        if graph_id not in self.graphs:
            logger.warning(f"Graph ID not found in execute_query: {graph_id}, available IDs: {list(self.graphs.keys())}")
            return []
        
        graph = self.graphs[graph_id]
        
        # Very simplified query processing
        # This implementation doesn't actually parse the Cypher query
        # It just returns some standard information based on the graph structure
        
        results = []
        
        # Find chart node
        chart_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chart']
        if chart_nodes:
            chart_node = chart_nodes[0]
            results.append({
                'chart_type': graph.nodes[chart_node].get('chart_type', 'unknown'),
                'title': graph.nodes[chart_node].get('title', 'Untitled')
            })
            
            # If query contains 'trend', return trend-related data
            if 'trend' in cypher_query.lower():
                # Find data points for trend analysis
                data_points = []
                for node in graph.nodes():
                    if graph.nodes[node].get('type') == 'data_point':
                        point_data = graph.nodes[node]
                        data_points.append({
                            'x': point_data.get('x', 0),
                            'y': point_data.get('y', 0),
                            'index': point_data.get('index', 0)
                        })
                
                # Sort data points by index or x
                data_points.sort(key=lambda p: p.get('index', p.get('x', 0)))
                
                # Create results for trend analysis
                for i in range(len(data_points) - 1):
                    results.append({
                        'x1': data_points[i].get('x', i),
                        'y1': data_points[i].get('y', 0),
                        'x2': data_points[i+1].get('x', i+1),
                        'y2': data_points[i+1].get('y', 0),
                        'change': data_points[i+1].get('y', 0) - data_points[i].get('y', 0)
                    })
            
            # If query contains 'comparison', return comparison data
            elif 'comparison' in cypher_query.lower():
                categories = []
                for node in graph.nodes():
                    node_type = graph.nodes[node].get('type', '')
                    if node_type in ['category', 'segment']:
                        categories.append({
                            'name': graph.nodes[node].get('name', 'Unknown'),
                            'value': graph.nodes[node].get('value', 0)
                        })
                
                # Sort categories by value
                categories.sort(key=lambda c: c['value'], reverse=True)
                
                # Add to results
                for category in categories:
                    results.append(category)
            
            # If query contains 'statistics', return statistics
            elif 'statistics' in cypher_query.lower():
                # Find statistics nodes
                for node in graph.nodes():
                    if graph.nodes[node].get('type') == 'statistic':
                        results.append({
                            'statistic': graph.nodes[node].get('name', 'unknown'),
                            'column': graph.nodes[node].get('column', 'default'),
                            'value': graph.nodes[node].get('value', 0)
                        })
        
        logger.info(f"Executed simplified query on in-memory graph (ID: {graph_id}), returned {len(results)} results")
        return results
    
    def delete_graph(self, graph_id: str) -> None:
        """
        Delete a graph from memory.
        
        Args:
            graph_id: Identifier for the graph
        """
        if graph_id in self.graphs:
            del self.graphs[graph_id]
            logger.info(f"Graph deleted from memory (ID: {graph_id})")
