"""
Neo4j Connector for Chart Insights System.
Handles storage and retrieval of knowledge graphs in Neo4j database.
"""

import logging
import networkx as nx
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

class Neo4jConnector:
    """
    Connects to Neo4j database and manages knowledge graph persistence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Neo4j connector.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.uri = config.get('graph_db', {}).get('uri', 'bolt://localhost:7687')
        self.username = config.get('graph_db', {}).get('username', 'neo4j')
        self.password = config.get('graph_db', {}).get('password', '')
        self.database = config.get('graph_db', {}).get('database', 'chart_insights')
        
        self.driver = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            logger.info(f"Connected to Neo4j database at {self.uri}")
            
            # Verify connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value != 1:
                    raise Exception("Connection test failed")
                
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
            
            # Don't raise exception, we'll use fallback in-memory storage
            if self.config.get('graph_db', {}).get('require_connection', False):
                raise
            else:
                logger.warning("Will use in-memory storage as fallback")
    
    def close(self) -> None:
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def store_graph(self, graph: nx.DiGraph, chart_id: str) -> str:
        """
        Store a NetworkX graph in Neo4j.
        
        Args:
            graph: NetworkX DiGraph to store
            chart_id: Unique identifier for the chart
            
        Returns:
            Neo4j identifier for the stored graph
        """
        if not self.driver:
            logger.error("No active Neo4j connection, using fallback in-memory storage")
            
            # Use fallback in-memory storage
            from .memory_connector import MemoryGraphConnector
            memory_connector = MemoryGraphConnector(self.config)
            return memory_connector.store_graph(graph, chart_id)
        
        # Create a database identifier for the graph
        db_id = f"graph_{chart_id}"
        
        with self.driver.session(database=self.database) as session:
            # Clear any existing graph with same ID
            session.run(
                """
                MATCH (n {graph_id: $graph_id})
                DETACH DELETE n
                """,
                graph_id=db_id
            )
            
            # Create nodes
            for node_id, attrs in graph.nodes(data=True):
                # Create a copy of attributes with proper types for Neo4j
                neo4j_attrs = self._prepare_properties(attrs)
                neo4j_attrs['graph_id'] = db_id
                neo4j_attrs['node_id'] = str(node_id)
                
                # Get node type for label
                node_type = attrs.get('type', 'Unknown')
                node_label = node_type.capitalize()
                
                # Create node with appropriate label
                cypher = f"""
                CREATE (n:{node_label} $props)
                RETURN n
                """
                session.run(cypher, props=neo4j_attrs)
            
            # Create relationships
            for source, target, attrs in graph.edges(data=True):
                # Get relationship type
                rel_type = attrs.get('type', 'RELATED_TO')
                
                # Create a copy of attributes with proper types for Neo4j
                neo4j_attrs = self._prepare_properties(attrs)
                neo4j_attrs['graph_id'] = db_id
                
                # Create relationship
                cypher = f"""
                MATCH (s {{graph_id: $graph_id, node_id: $source}}),
                      (t {{graph_id: $graph_id, node_id: $target}})
                CREATE (s)-[r:{rel_type} $props]->(t)
                RETURN r
                """
                session.run(
                    cypher, 
                    graph_id=db_id, 
                    source=str(source), 
                    target=str(target), 
                    props=neo4j_attrs
                )
            
            logger.info(f"Graph stored in Neo4j with ID: {db_id}")
            return db_id
    
    def load_graph(self, graph_id: str) -> nx.DiGraph:
        """
        Load a graph from Neo4j.
        
        Args:
            graph_id: Neo4j identifier for the graph
            
        Returns:
            NetworkX DiGraph loaded from Neo4j
        """
        if not self.driver:
            logger.error("No active Neo4j connection, using fallback in-memory storage")
            
            # Use fallback in-memory storage
            from .memory_connector import MemoryGraphConnector
            memory_connector = MemoryGraphConnector(self.config)
            return memory_connector.load_graph(graph_id)
        
        graph = nx.DiGraph()
        
        with self.driver.session(database=self.database) as session:
            # Load nodes
            result = session.run(
                """
                MATCH (n {graph_id: $graph_id})
                RETURN n
                """,
                graph_id=graph_id
            )
            
            for record in result:
                node = record["n"]
                node_id = node["node_id"]
                
                # Add node with all properties except neo4j internal ones
                props = dict(node.items())
                exclude_keys = ['graph_id', 'node_id']
                node_props = {k: v for k, v in props.items() if k not in exclude_keys}
                
                graph.add_node(node_id, **node_props)
            
            # Load relationships
            result = session.run(
                """
                MATCH (s {graph_id: $graph_id})-[r]->(t {graph_id: $graph_id})
                RETURN s.node_id as source, t.node_id as target, type(r) as type, properties(r) as props
                """,
                graph_id=graph_id
            )
            
            for record in result:
                source = record["source"]
                target = record["target"]
                rel_type = record["type"]
                props = record["props"]
                
                # Add edge with all properties
                edge_props = {k: v for k, v in props.items() if k != 'graph_id'}
                edge_props['type'] = rel_type
                
                graph.add_edge(source, target, **edge_props)
        
        logger.info(f"Graph loaded from Neo4j (ID: {graph_id}) with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    def get_chart_ids(self) -> List[str]:
        """
        Get list of all chart IDs in the database.
        
        Returns:
            List of chart IDs
        """
        if not self.driver:
            logger.error("No active Neo4j connection, using fallback in-memory storage")
            
            # Use fallback in-memory storage
            from .memory_connector import MemoryGraphConnector
            memory_connector = MemoryGraphConnector(self.config)
            return memory_connector.get_chart_ids()
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (n:Chart) 
                RETURN DISTINCT n.graph_id as chart_id
                """
            )
            
            chart_ids = [record["chart_id"] for record in result]
            return chart_ids
    
    def execute_query(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the database.
        
        Args:
            cypher_query: Cypher query string
            params: Query parameters
            
        Returns:
            List of results as dictionaries
        """
        if not self.driver:
            logger.error("No active Neo4j connection, using fallback in-memory storage")
            
            # Use fallback in-memory storage
            from .memory_connector import MemoryGraphConnector
            memory_connector = MemoryGraphConnector(self.config)
            return memory_connector.execute_query(cypher_query, params or {})
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query, params or {})
            
            # Convert result to list of dictionaries
            return [dict(record) for record in result]
    
    def delete_graph(self, graph_id: str) -> None:
        """
        Delete a graph from Neo4j.
        
        Args:
            graph_id: Neo4j identifier for the graph
        """
        if not self.driver:
            logger.error("No active Neo4j connection, using fallback in-memory storage")
            
            # Use fallback in-memory storage
            from .memory_connector import MemoryGraphConnector
            memory_connector = MemoryGraphConnector(self.config)
            return memory_connector.delete_graph(graph_id)
        
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (n {graph_id: $graph_id})
                DETACH DELETE n
                """,
                graph_id=graph_id
            )
            
            logger.info(f"Graph deleted from Neo4j (ID: {graph_id})")
    
    def _prepare_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare properties for Neo4j by ensuring all values are of supported types.
        
        Args:
            props: Dictionary of properties
            
        Returns:
            Dictionary with Neo4j-compatible property values
        """
        result = {}
        
        for key, value in props.items():
            # Skip None values
            if value is None:
                continue
                
            # Convert to Neo4j-compatible types
            if isinstance(value, (str, int, float, bool)):
                # These types are directly supported
                result[key] = value
            elif isinstance(value, list):
                # Convert lists to strings
                result[key] = str(value)
            elif isinstance(value, dict):
                # Convert dictionaries to strings
                result[key] = str(value)
            else:
                # Convert other types to strings
                result[key] = str(value)
        
        return result
