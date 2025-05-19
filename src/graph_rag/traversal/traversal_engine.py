"""
Graph Traversal Engine for GraphRAG.
Handles traversal strategies for exploring the knowledge graph.
"""

import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class GraphTraversalEngine:
    """
    Engine for traversing knowledge graphs to find relevant context nodes.
    
    This class implements various traversal strategies for exploring
    knowledge graphs to find nodes relevant to specific insight types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the traversal engine.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.max_hops = config.get('graph_rag', {}).get('max_hops', 3)
        self.context_window = config.get('graph_rag', {}).get('context_window', 5)
    
    def get_context_nodes(self, 
                         graph: nx.DiGraph, 
                         start_node: str, 
                         insight_type: str, 
                         max_hops: Optional[int] = None) -> Dict[str, Any]:
        """
        Get context nodes relevant to a specific insight type.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            start_node: ID of the starting node (usually chart node)
            insight_type: Type of insight being generated
            max_hops: Maximum number of hops from start node
            
        Returns:
            Dictionary with context nodes and their attributes
        """
        if max_hops is None:
            max_hops = self.max_hops
        
        # Determine traversal strategy based on insight type
        if insight_type == 'trend':
            context = self._traverse_for_trends(graph, start_node, max_hops)
        elif insight_type == 'comparison':
            context = self._traverse_for_comparisons(graph, start_node, max_hops)
        elif insight_type == 'anomaly':
            context = self._traverse_for_anomalies(graph, start_node, max_hops)
        elif insight_type == 'correlation':
            context = self._traverse_for_correlations(graph, start_node, max_hops)
        else:
            # Default to community-based traversal
            context = self._traverse_community(graph, start_node, max_hops)
        
        return context
    
    def _traverse_for_trends(self, 
                           graph: nx.DiGraph, 
                           start_node: str, 
                           max_hops: int) -> Dict[str, Any]:
        """
        Traverse graph to find nodes relevant to trend insights.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            start_node: ID of the starting node
            max_hops: Maximum number of hops from start node
            
        Returns:
            Dictionary with context nodes and structured information
        """
        # Get chart type
        chart_type = graph.nodes[start_node].get('chart_type', 'unknown')
        
        # Different traversal strategies based on chart type
        if chart_type in ['line', 'bar']:
            # For time series charts, focus on sequences
            return self._traverse_time_sequence(graph, start_node, max_hops)
        else:
            # For other charts, use value-based traversal
            return self._traverse_value_based(graph, start_node, max_hops)
    
    def _traverse_time_sequence(self, 
                              graph: nx.DiGraph, 
                              start_node: str, 
                              max_hops: int) -> Dict[str, Any]:
        """
        Traverse graph focusing on time sequences.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            start_node: ID of the starting node
            max_hops: Maximum number of hops from start node
            
        Returns:
            Dictionary with context nodes and structured information
        """
        context = {
            'chart': dict(graph.nodes[start_node]),
            'series': [],
            'statistics': {}
        }
        
        # Find series nodes
        series_nodes = []
        for node in graph.successors(start_node):
            if graph.nodes[node].get('type') == 'series':
                series_nodes.append(node)
        
        # If no explicit series nodes, look for data points
        if not series_nodes:
            # Create a default series
            series = {
                'name': 'main',
                'points': []
            }
            
            # Find data points
            for node in graph.successors(start_node):
                if graph.nodes[node].get('type') == 'data_point':
                    series['points'].append(dict(graph.nodes[node]))
            
            # Sort points by index or x value
            series['points'].sort(key=lambda p: p.get('index', p.get('x', 0)))
            
            context['series'].append(series)
        else:
            # Process each series
            for series_node in series_nodes:
                series = {
                    'name': graph.nodes[series_node].get('name', 'unnamed'),
                    'points': []
                }
                
                # Get data points for this series
                for node in graph.successors(series_node):
                    if graph.nodes[node].get('type') == 'data_point':
                        series['points'].append(dict(graph.nodes[node]))
                
                # Sort points
                series['points'].sort(key=lambda p: p.get('index', p.get('x', 0)))
                
                context['series'].append(series)
        
        # Find statistics node
        for node in graph.successors(start_node):
            if graph.nodes[node].get('type') == 'statistics':
                # Get all statistics
                for stat_node in graph.successors(node):
                    stat_type = graph.nodes[stat_node].get('name', 'unknown')
                    stat_value = graph.nodes[stat_node].get('value', 0)
                    
                    if stat_type not in context['statistics']:
                        context['statistics'][stat_type] = {}
                    
                    # Group by column if available
                    column = graph.nodes[stat_node].get('column', 'default')
                    context['statistics'][stat_type][column] = stat_value
        
        return context
    
    def _traverse_value_based(self, 
                            graph: nx.DiGraph, 
                            start_node: str, 
                            max_hops: int) -> Dict[str, Any]:
        """
        Traverse graph focusing on value-based relationships.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            start_node: ID of the starting node
            max_hops: Maximum number of hops from start node
            
        Returns:
            Dictionary with context nodes and structured information
        """
        context = {
            'chart': dict(graph.nodes[start_node]),
            'categories': [],
            'statistics': {}
        }
        
        # Find category/segment nodes
        for node in graph.successors(start_node):
            node_type = graph.nodes[node].get('type', '')
            if node_type in ['category', 'segment']:
                category = dict(graph.nodes[node])
                
                # Get associated value if exists
                for value_node in graph.successors(node):
                    if graph.nodes[value_node].get('type') == 'value':
                        category['value_node'] = dict(graph.nodes[value_node])
                
                context['categories'].append(category)
        
        # Find statistics node
        for node in graph.successors(start_node):
            if graph.nodes[node].get('type') == 'statistics':
                # Get all statistics
                for stat_node in graph.successors(node):
                    stat_type = graph.nodes[stat_node].get('name', 'unknown')
                    stat_value = graph.nodes[stat_node].get('value', 0)
                    
                    if stat_type not in context['statistics']:
                        context['statistics'][stat_type] = {}
                    
                    # Group by column if available
                    column = graph.nodes[stat_node].get('column', 'default')
                    context['statistics'][stat_type][column] = stat_value
        
        # Sort categories by value if available
        if context['categories']:
            context['categories'].sort(
                key=lambda c: c.get('value', c.get('value_node', {}).get('value', 0) if 'value_node' in c else 0),
                reverse=True
            )
        
        return context
    
    def _traverse_for_comparisons(self, 
                                graph: nx.DiGraph, 
                                start_node: str, 
                                max_hops: int) -> Dict[str, Any]:
        """
        Traverse graph to find nodes relevant to comparison insights.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            start_node: ID of the starting node
            max_hops: Maximum number of hops from start node
            
        Returns:
            Dictionary with context nodes and structured information
        """
        context = {
            'chart': dict(graph.nodes[start_node]),
            'comparisons': [],
            'statistics': {}
        }
        
        # Get chart type
        chart_type = graph.nodes[start_node].get('chart_type', 'unknown')
        
        # Find data categories or segments
        data_nodes = []
        for node in graph.successors(start_node):
            node_type = graph.nodes[node].get('type', '')
            if node_type in ['category', 'segment', 'data_point']:
                data_nodes.append(node)
        
        # Find comparison relationships
        for node in data_nodes:
            comparisons = []
            
            # Find direct comparison edges
            for target in graph.successors(node):
                edge_data = graph.get_edge_data(node, target)
                edge_type = edge_data.get('type', '')
                
                if edge_type in ['GREATER_THAN', 'LESS_THAN', 'EQUAL_TO']:
                    comparisons.append({
                        'source': dict(graph.nodes[node]),
                        'target': dict(graph.nodes[target]),
                        'relation': edge_type,
                        'data': edge_data
                    })
            
            if comparisons:
                # Add node with its comparisons
                context['comparisons'].append({
                    'node': dict(graph.nodes[node]),
                    'relations': comparisons
                })
        
        # Find statistical comparisons
        stats_node = None
        for node in graph.successors(start_node):
            if graph.nodes[node].get('type') == 'statistics':
                stats_node = node
                break
        
        if stats_node:
            # Find statistical nodes
            mean_nodes = []
            for stat_node in graph.successors(stats_node):
                stat_type = graph.nodes[stat_node].get('name', '')
                if stat_type in ['mean', 'median', 'minimum', 'maximum']:
                    # Find nodes that are above/below this stat
                    stat_comparisons = []
                    
                    for source, target, edge_data in graph.in_edges(stat_node, data=True):
                        edge_type = edge_data.get('type', '')
                        if edge_type in ['ABOVE_MEAN', 'BELOW_MEAN', 'IS_MAX', 'IS_MIN']:
                            stat_comparisons.append({
                                'node': dict(graph.nodes[source]),
                                'relation': edge_type,
                                'data': edge_data
                            })
                    
                    context['statistics'][stat_type] = {
                        'value': graph.nodes[stat_node].get('value', 0),
                        'column': graph.nodes[stat_node].get('column', 'default'),
                        'comparisons': stat_comparisons
                    }
        
        return context
    
    def _traverse_for_anomalies(self, 
                              graph: nx.DiGraph, 
                              start_node: str, 
                              max_hops: int) -> Dict[str, Any]:
        """
        Traverse graph to find nodes relevant to anomaly insights.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            start_node: ID of the starting node
            max_hops: Maximum number of hops from start node
            
        Returns:
            Dictionary with context nodes and structured information
        """
        context = {
            'chart': dict(graph.nodes[start_node]),
            'extreme_values': [],
            'statistics': {},
            'data_points': []
        }
        
        # Find statistics node
        stats_node = None
        for node in graph.successors(start_node):
            if graph.nodes[node].get('type') == 'statistics':
                stats_node = node
                break
        
        if stats_node:
            # Gather key statistics
            for stat_node in graph.successors(stats_node):
                stat_type = graph.nodes[stat_node].get('name', '')
                
                # Only include relevant statistics
                if stat_type in ['mean', 'median', 'minimum', 'maximum', 'standard_deviation']:
                    if stat_type not in context['statistics']:
                        context['statistics'][stat_type] = {}
                    
                    column = graph.nodes[stat_node].get('column', 'default')
                    value = graph.nodes[stat_node].get('value', 0)
                    context['statistics'][stat_type][column] = value
                    
                    # For min/max, also record the actual data point
                    if stat_type in ['minimum', 'maximum']:
                        for source, target, edge_data in graph.in_edges(stat_node, data=True):
                            edge_type = edge_data.get('type', '')
                            if edge_type in ['IS_MAX', 'IS_MIN']:
                                context['extreme_values'].append({
                                    'type': stat_type,
                                    'node': dict(graph.nodes[source]),
                                    'value': value
                                })
        
        # Collect all data points or categories
        for node in graph.successors(start_node):
            node_type = graph.nodes[node].get('type', '')
            
            if node_type in ['data_point', 'category', 'segment']:
                node_data = dict(graph.nodes[node])
                
                # Calculate Z-score if mean and std dev are available
                if 'standard_deviation' in context['statistics'] and 'mean' in context['statistics']:
                    value = node_data.get('value', 0)
                    mean = context['statistics']['mean'].get('default', 0)
                    std_dev = context['statistics']['standard_deviation'].get('default', 1)
                    
                    if std_dev > 0:
                        z_score = (value - mean) / std_dev
                        node_data['z_score'] = z_score
                        
                        # Flag potential anomalies
                        if abs(z_score) > 2:
                            node_data['potential_anomaly'] = True
                            node_data['anomaly_severity'] = abs(z_score)
                
                context['data_points'].append(node_data)
        
        # Sort data points by anomaly severity if available
        context['data_points'].sort(
            key=lambda p: p.get('anomaly_severity', 0),
            reverse=True
        )
        
        return context
    
    def _traverse_for_correlations(self, 
                                 graph: nx.DiGraph, 
                                 start_node: str, 
                                 max_hops: int) -> Dict[str, Any]:
        """
        Traverse graph to find nodes relevant to correlation insights.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            start_node: ID of the starting node
            max_hops: Maximum number of hops from start node
            
        Returns:
            Dictionary with context nodes and structured information
        """
        context = {
            'chart': dict(graph.nodes[start_node]),
            'series_pairs': [],
            'statistics': {}
        }
        
        # For correlations, we need to identify series pairs or related categories
        chart_type = graph.nodes[start_node].get('chart_type', 'unknown')
        
        if chart_type in ['scatter', 'line']:
            # For scatter plots, x and y values directly show correlation
            if chart_type == 'scatter':
                points = []
                
                # Collect all data points
                for node in graph.successors(start_node):
                    if graph.nodes[node].get('type') == 'data_point':
                        point_data = dict(graph.nodes[node])
                        points.append(point_data)
                
                # Add as a series pair
                if points:
                    context['series_pairs'].append({
                        'x_series': graph.nodes[start_node].get('x_label', 'X'),
                        'y_series': graph.nodes[start_node].get('y_label', 'Y'),
                        'points': points
                    })
            
            # For line charts, we need to compare multiple series
            elif chart_type == 'line':
                series_nodes = []
                
                # Find series nodes
                for node in graph.successors(start_node):
                    if graph.nodes[node].get('type') == 'series':
                        series_nodes.append(node)
                
                # Compare pairs of series
                for i in range(len(series_nodes)):
                    for j in range(i + 1, len(series_nodes)):
                        series1 = series_nodes[i]
                        series2 = series_nodes[j]
                        
                        # Collect points for each series
                        points1 = []
                        points2 = []
                        
                        for node in graph.successors(series1):
                            if graph.nodes[node].get('type') == 'data_point':
                                points1.append(dict(graph.nodes[node]))
                        
                        for node in graph.successors(series2):
                            if graph.nodes[node].get('type') == 'data_point':
                                points2.append(dict(graph.nodes[node]))
                        
                        # Sort points
                        points1.sort(key=lambda p: p.get('index', p.get('x', 0)))
                        points2.sort(key=lambda p: p.get('index', p.get('x', 0)))
                        
                        # Create paired points if possible
                        paired_points = []
                        min_len = min(len(points1), len(points2))
                        
                        for k in range(min_len):
                            paired_points.append({
                                'x': points1[k].get('x', k),
                                'series1_value': points1[k].get('y', 0),
                                'series2_value': points2[k].get('y', 0)
                            })
                        
                        # Add as a series pair
                        context['series_pairs'].append({
                            'series1': graph.nodes[series1].get('name', f'Series {i+1}'),
                            'series2': graph.nodes[series2].get('name', f'Series {j+1}'),
                            'paired_points': paired_points
                        })
        
        else:
            # For other chart types, look for category relationships
            categories = []
            
            # Collect categories or segments
            for node in graph.successors(start_node):
                node_type = graph.nodes[node].get('type', '')
                if node_type in ['category', 'segment']:
                    categories.append({
                        'id': node,
                        'data': dict(graph.nodes[node])
                    })
            
            # Look for relationships between categories
            for i in range(len(categories)):
                for j in range(i + 1, len(categories)):
                    cat1 = categories[i]
                    cat2 = categories[j]
                    
                    # Check for direct relationship
                    if graph.has_edge(cat1['id'], cat2['id']) or graph.has_edge(cat2['id'], cat1['id']):
                        edge = graph.get_edge_data(cat1['id'], cat2['id']) or graph.get_edge_data(cat2['id'], cat1['id'])
                        
                        context['category_relations'] = context.get('category_relations', [])
                        context['category_relations'].append({
                            'category1': cat1['data'],
                            'category2': cat2['data'],
                            'relation': edge.get('type', 'RELATED'),
                            'data': edge
                        })
        
        return context
    
    def _traverse_community(self, 
                          graph: nx.DiGraph, 
                          start_node: str, 
                          max_hops: int) -> Dict[str, Any]:
        """
        Traverse graph using community detection.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            start_node: ID of the starting node
            max_hops: Maximum number of hops from start node
            
        Returns:
            Dictionary with context nodes organized by community
        """
        # Convert to undirected for community detection
        undirected_graph = graph.to_undirected()
        
        try:
            # Use Louvain community detection
            from community import best_partition
            partition = best_partition(undirected_graph)
        except ImportError:
            logger.warning("Community detection package not available. Using generic traversal.")
            return self._traverse_generic(graph, start_node, max_hops)
        
        # Get the community of the start node
        start_community = partition.get(start_node, 0)
        
        # Get nodes in the same community
        community_nodes = [n for n, c in partition.items() if c == start_community]
        
        # Limit to nodes within max_hops
        reachable_nodes = set()
        current_nodes = {start_node}
        
        for hop in range(max_hops):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update(graph.successors(node))
                next_nodes.update(graph.predecessors(node))
            
            reachable_nodes.update(current_nodes)
            current_nodes = next_nodes - reachable_nodes
            
            if not current_nodes:
                break
        
        # Intersection of community and reachable nodes
        context_node_ids = set(community_nodes).intersection(reachable_nodes)
        
        # Create context structure
        context = {
            'chart': dict(graph.nodes[start_node]),
            'community_nodes': [],
            'community_edges': []
        }
        
        # Add nodes and edges
        for node_id in context_node_ids:
            context['community_nodes'].append({
                'id': node_id,
                'data': dict(graph.nodes[node_id])
            })
        
        # Add edges between community nodes
        for source, target, data in graph.edges(data=True):
            if source in context_node_ids and target in context_node_ids:
                context['community_edges'].append({
                    'source': source,
                    'target': target,
                    'data': data
                })
        
        return context
    
    def _traverse_generic(self, 
                        graph: nx.DiGraph, 
                        start_node: str, 
                        max_hops: int) -> Dict[str, Any]:
        """
        Generic graph traversal strategy.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            start_node: ID of the starting node
            max_hops: Maximum number of hops from start node
            
        Returns:
            Dictionary with context nodes and structured information
        """
        context = {
            'chart': dict(graph.nodes[start_node]),
            'nodes': [],
            'edges': []
        }
        
        # Simple BFS traversal
        visited = set()
        queue = [(start_node, 0)]  # (node, hop_count)
        
        while queue:
            node, hops = queue.pop(0)
            
            if node in visited:
                continue
            
            visited.add(node)
            
            # Add node to context
            context['nodes'].append({
                'id': node,
                'data': dict(graph.nodes[node]),
                'hops': hops
            })
            
            # Stop at max hops
            if hops >= max_hops:
                continue
            
            # Visit neighbors
            for neighbor in list(graph.successors(node)) + list(graph.predecessors(node)):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))
                    
                    # Add edge to context
                    if graph.has_edge(node, neighbor):
                        edge_data = graph.get_edge_data(node, neighbor)
                        context['edges'].append({
                            'source': node,
                            'target': neighbor,
                            'data': edge_data
                        })
                    elif graph.has_edge(neighbor, node):
                        edge_data = graph.get_edge_data(neighbor, node)
                        context['edges'].append({
                            'source': neighbor,
                            'target': node,
                            'data': edge_data
                        })
        
        return context
