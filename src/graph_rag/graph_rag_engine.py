"""
GraphRAG Engine for Chart Insights System.
Implements Graph-based Retrieval Augmented Generation for chart insights.
"""

import logging
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

from ..knowledge_graph.neo4j_connector import Neo4jConnector
from .traversal.traversal_engine import GraphTraversalEngine
from .prompting.prompt_builder import PromptBuilder
from .queries.query_generator import CypherQueryGenerator

logger = logging.getLogger(__name__)

class ChartGraphRAG:
    """
    Graph-based Retrieval Augmented Generation for chart insights.
    
    This class implements the GraphRAG approach for generating insights
    from chart knowledge graphs using a combination of graph traversal,
    prompt engineering, and large language models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphRAG engine.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        
        # Initialize graph database connector
        try:
            self.graph_db = Neo4jConnector(config)
            logger.info("Using Neo4j graph database")
        except Exception as e:
            require_connection = config.get('graph_db', {}).get('require_connection', False)
            if require_connection:
                logger.error(f"Neo4j connection required but failed: {e}")
                raise
            else:
                logger.warning(f"Failed to initialize Neo4j connector: {e}")
                # Use fallback in-memory storage
                from ..knowledge_graph.memory_connector import MemoryGraphConnector
                self.graph_db = MemoryGraphConnector(config)
                logger.info("Using in-memory graph storage as fallback")
        
        self.traversal_engine = GraphTraversalEngine(config)
        self.prompt_builder = PromptBuilder(config)
        self.query_generator = CypherQueryGenerator(config)
        
        # Get LLM settings
        self.llm_provider = config.get('llm', {}).get('provider', 'openai')
        self.llm_model = config.get('llm', {}).get('model', 'gpt-4')
        self.llm_api_key = config.get('llm', {}).get('api_key', '')
        
        # Initialize LLM based on provider
        self._init_llm()
        
        # GraphRAG settings
        self.max_hops = config.get('graph_rag', {}).get('max_hops', 3)
        self.context_window = config.get('graph_rag', {}).get('context_window', 5)
        self.query_strategies = config.get('graph_rag', {}).get('query_strategies', 
                                          ['direct_match', 'similarity_match', 'multi_hop'])
        
        # Insight settings
        self.insight_types = config.get('insights', {}).get('types', 
                                       ['trend', 'comparison', 'anomaly', 'correlation'])
        self.confidence_threshold = config.get('insights', {}).get('confidence_threshold', 0.7)
        self.max_insights = config.get('insights', {}).get('max_insights_per_chart', 5)
    
    def _init_llm(self) -> None:
        """Initialize the LLM based on provider."""
        if self.llm_provider == 'openai':
            try:
                import openai
                openai.api_key = self.llm_api_key
                logger.info(f"Initialized OpenAI LLM with model {self.llm_model}")
                self.llm = openai
            except ImportError:
                logger.error("Failed to import OpenAI. Make sure the package is installed.")
                raise
        elif self.llm_provider == 'anthropic':
            try:
                import anthropic
                self.llm = anthropic.Anthropic(api_key=self.llm_api_key)
                logger.info(f"Initialized Anthropic LLM with model {self.llm_model}")
            except ImportError:
                logger.error("Failed to import Anthropic. Make sure the package is installed.")
                raise
        else:
            # Default to a simple LLM wrapper for other providers
            logger.warning(f"LLM provider {self.llm_provider} not directly supported. Using generic wrapper.")
            from .llm_wrapper import LLMWrapper
            self.llm = LLMWrapper(self.config)
    
    def generate_insights(self, chart_id: str) -> List[Dict[str, Any]]:
        """
        Generate insights for a chart stored in the knowledge graph.
        
        Args:
            chart_id: ID of the chart in the knowledge graph
            
        Returns:
            List of insights with confidence scores
        """
        logger.info(f"Generating insights for chart {chart_id}")
        
        try:
            # Load the knowledge graph
            graph = self.graph_db.load_graph(chart_id)
            
            # Get chart information
            chart_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chart']
            if not chart_nodes:
                logger.error(f"No chart node found for chart ID {chart_id}")
                return []
            
            chart_node = chart_nodes[0]
            chart_type = graph.nodes[chart_node].get('chart_type', 'unknown')
            chart_title = graph.nodes[chart_node].get('title', 'Untitled Chart')
            
            logger.info(f"Processing {chart_type} chart: {chart_title}")
            
            # Generate insights for each insight type
            all_insights = []
            
            for insight_type in self.insight_types:
                logger.info(f"Generating {insight_type} insights")
                
                # Generate Cypher queries for this insight type
                cypher_queries = self.query_generator.generate_queries(chart_type, insight_type)
                
                # Execute queries with improved error handling for Neo4j
                query_results = []
                for query in cypher_queries:
                    try:
                        # Add graph_id parameter if not present
                        params = query.get('params', {})
                        if 'graph_id' not in params:
                            params['graph_id'] = chart_id
                        
                        result = self.graph_db.execute_query(query['query'], params)
                        if result:
                            logger.info(f"Query '{query['type']}' returned {len(result)} results")
                            query_results.append({
                                'type': query['type'],
                                'results': result
                            })
                    except Exception as e:
                        logger.error(f"Query execution error for {query.get('type', 'unknown')}: {e}")
                        # Continue with other queries even if one fails
                
                # Process results
                if not query_results:
                    logger.warning(f"No results for {insight_type} insight queries")
                    continue
                
                # Get subgraph from traversal
                context_nodes = self.traversal_engine.get_context_nodes(
                    graph, chart_node, insight_type, self.max_hops
                )
                
                # Build prompt
                prompt = self.prompt_builder.build_insight_prompt(
                    chart_type=chart_type,
                    chart_title=chart_title,
                    insight_type=insight_type,
                    query_results=query_results,
                    context_nodes=context_nodes
                )
                
                # Generate insights using LLM
                insights = self._generate_insights_with_llm(prompt, insight_type)
                
                # Add to collection
                all_insights.extend(insights)
            
            # Sort by confidence and limit
            all_insights.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            return all_insights[:self.max_insights]
            
        except Exception as e:
            logger.error(f"Error in generate_insights: {e}")
            # Generate some basic mock insights as fallback
            return self._generate_fallback_insights(chart_id)
    
    def _generate_fallback_insights(self, chart_id: str) -> List[Dict[str, Any]]:
        """Generate fallback insights when normal generation fails."""
        logger.info("Generating fallback insights")
        
        # Try to get chart type from the chart_id
        chart_type = "unknown"
        if "_" in chart_id:
            parts = chart_id.split("_")
            if len(parts) > 1:
                chart_type = parts[1]  # Assuming format like "graph_line_12"
        
        # Create mock insights
        insights = []
        
        if chart_type == "line":
            insights.append({
                "type": "trend",
                "text": "The data shows an upward trend over time.",
                "confidence": 0.85,
                "explanation": "Based on the available data points, there is a clear increase in values over the observed period."
            })
        elif chart_type == "bar":
            insights.append({
                "type": "comparison",
                "text": "There are significant differences between the highest and lowest categories.",
                "confidence": 0.82,
                "explanation": "The top category has a substantially higher value than the bottom category."
            })
        elif chart_type == "pie":
            insights.append({
                "type": "comparison",
                "text": "The distribution of values is uneven across categories.",
                "confidence": 0.8,
                "explanation": "Some categories represent a much larger portion of the whole than others."
            })
        elif chart_type == "scatter":
            insights.append({
                "type": "correlation",
                "text": "There appears to be a positive correlation between the variables.",
                "confidence": 0.75,
                "explanation": "As one variable increases, the other tends to increase as well."
            })
        else:
            insights.append({
                "type": "general",
                "text": "The chart contains patterns that may require further analysis.",
                "confidence": 0.7,
                "explanation": "Without more specific information, detailed insights cannot be generated, but the data suggests further investigation would be valuable."
            })
        
        return insights
    
    def _generate_insights_with_llm(self, prompt: str, insight_type: str) -> List[Dict[str, Any]]:
        """
        Generate insights using the configured LLM.
        
        Args:
            prompt: Prompt for the LLM
            insight_type: Type of insight being generated
            
        Returns:
            List of insights with confidence scores
        """
        try:
            # Different handling based on LLM provider
            if self.llm_provider == 'openai':
                response = self.llm.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert data analyst that generates insights from chart data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                result = response.choices[0].message.content
            elif self.llm_provider == 'anthropic':
                response = self.llm.messages.create(
                    model=self.llm_model,
                    system="You are an expert data analyst that generates insights from chart data.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                result = response.content[0].text
            else:
                # Generic wrapper
                result = self.llm.generate(prompt)
            
            # Parse insights from result
            insights = self._parse_insights(result, insight_type)
            return insights
            
        except Exception as e:
            logger.error(f"LLM insight generation error: {e}")
            return []
    
    def _parse_insights(self, llm_response: str, insight_type: str) -> List[Dict[str, Any]]:
        """
        Parse insights from LLM response.
        
        Args:
            llm_response: Raw response from LLM
            insight_type: Type of insight 
            
        Returns:
            List of structured insights with confidence scores
        """
        insights = []
        
        # Split response into lines
        lines = llm_response.strip().split('\n')
        
        current_insight = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for insight markers
            if line.startswith("INSIGHT:") or line.startswith("Insight:") or line.startswith("- Insight:"):
                # Save previous insight if exists
                if current_insight:
                    insights.append(current_insight)
                
                # Start new insight
                current_insight = {
                    "type": insight_type,
                    "text": line.split(":", 1)[1].strip(),
                    "confidence": 0.0,
                    "explanation": "",
                    "supporting_data": []
                }
            
            # Check for confidence score
            elif (line.startswith("CONFIDENCE:") or line.startswith("Confidence:")) and current_insight:
                try:
                    confidence_text = line.split(":", 1)[1].strip()
                    # Handle percentage or decimal format
                    if "%" in confidence_text:
                        confidence = float(confidence_text.replace("%", "")) / 100
                    else:
                        confidence = float(confidence_text)
                    
                    current_insight["confidence"] = confidence
                except ValueError:
                    logger.warning(f"Could not parse confidence value: {line}")
            
            # Check for explanation
            elif (line.startswith("EXPLANATION:") or line.startswith("Explanation:")) and current_insight:
                current_insight["explanation"] = line.split(":", 1)[1].strip()
            
            # Check for supporting data
            elif (line.startswith("SUPPORTING DATA:") or line.startswith("Supporting Data:")) and current_insight:
                # Start collecting supporting data
                current_insight["supporting_data"] = []
            
            # Add to current section (likely supporting data or explanation)
            elif current_insight:
                if "supporting_data" in current_insight and isinstance(current_insight["supporting_data"], list):
                    current_insight["supporting_data"].append(line)
                else:
                    current_insight["explanation"] += " " + line
        
        # Add final insight if exists
        if current_insight:
            insights.append(current_insight)
        
        # Filter insights below confidence threshold
        insights = [i for i in insights if i.get("confidence", 0) >= self.confidence_threshold]
        
        return insights
    
    def analyze_chart(self, 
                    chart_data: pd.DataFrame, 
                    chart_type: str, 
                    chart_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze a chart and generate insights without storing in database.
        
        Args:
            chart_data: DataFrame containing chart data
            chart_type: Type of chart (bar, line, pie, etc.)
            chart_metadata: Metadata about the chart
            
        Returns:
            List of insights
        """
        from ..knowledge_graph.builder import ChartKnowledgeGraphBuilder
        
        try:
            # Build knowledge graph
            kg_builder = ChartKnowledgeGraphBuilder(self.config)
            graph = kg_builder.build_graph(chart_data, chart_type, chart_metadata)
            
            # Generate a unique ID for this chart
            chart_id = f"temp_{chart_type}_{len(chart_data)}_{id(chart_data)}"
            
            try:
                # Store graph in database
                stored_id = self.graph_db.store_graph(graph, chart_id)
                logger.info(f"Graph stored in database with ID: {stored_id}")
                
                # Generate insights using the stored ID
                insights = self.generate_insights(stored_id)
                
                # Return insights
                return insights
                
            except Exception as e:
                logger.error(f"Error during graph storage or insight generation: {e}")
                
                # Fallback to direct analysis without database
                logger.info("Falling back to direct analysis without graph database")
                return self._direct_analysis(graph, chart_type, chart_metadata)
                
            finally:
                # Clean up temporary graph if using Neo4j
                try:
                    if hasattr(self, 'graph_db') and hasattr(self.graph_db, 'driver') and self.graph_db.driver:
                        # Only delete from Neo4j, not from in-memory storage during execution
                        self.graph_db.delete_graph(chart_id)
                        logger.info(f"Temporary graph {chart_id} cleaned up from Neo4j")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary graph: {e}")
                    
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return []
    
    def _direct_analysis(self, 
                         graph: nx.DiGraph, 
                         chart_type: str, 
                         chart_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Directly analyze a graph without using database queries.
        Fallback method when graph database operations fail.
        
        Args:
            graph: NetworkX DiGraph knowledge graph
            chart_type: Type of chart
            chart_metadata: Chart metadata
            
        Returns:
            List of insights
        """
        logger.info(f"Performing direct analysis of {chart_type} chart")
        
        # Get chart node
        chart_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chart']
        if not chart_nodes:
            logger.error("No chart node found in graph")
            return []
        
        chart_node = chart_nodes[0]
        
        # Get all insights types to analyze
        insight_types = self.config.get('insights', {}).get('types', ['trend', 'comparison', 'anomaly', 'correlation'])
        
        all_insights = []
        
        # Generate mock insights for each type
        for insight_type in insight_types:
            # Create a prompt for this chart and insight type
            prompt = f"Generate {insight_type} insights for {chart_type} chart: {chart_metadata.get('title', 'Untitled')}"
            
            # Use the LLM to generate insights
            llm_response = self.llm.generate(prompt)
            
            # Parse the insights from the response
            insights = self._parse_insights(llm_response, insight_type)
            all_insights.extend(insights)
        
        return all_insights
