"""
Cypher Query Generator for Chart Insights System.
Generates Neo4j Cypher queries based on chart type and insight type.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class CypherQueryGenerator:
    """
    Generates Neo4j Cypher queries for different chart and insight types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the query generator.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
    
    def generate_queries(self, chart_type: str, insight_type: str) -> List[Dict[str, Any]]:
        """
        Generate Cypher queries for a specific chart and insight type.
        
        Args:
            chart_type: Type of chart (bar, line, pie, etc.)
            insight_type: Type of insight (trend, comparison, anomaly, correlation)
            
        Returns:
            List of dictionaries with query and params
        """
        # Call specific query generator based on insight type
        if insight_type == 'trend':
            return self._generate_trend_queries(chart_type)
        elif insight_type == 'comparison':
            return self._generate_comparison_queries(chart_type)
        elif insight_type == 'anomaly':
            return self._generate_anomaly_queries(chart_type)
        elif insight_type == 'correlation':
            return self._generate_correlation_queries(chart_type)
        else:
            logger.warning(f"Unknown insight type: {insight_type}. Using generic queries.")
            return self._generate_generic_queries(chart_type)
    
    def _generate_trend_queries(self, chart_type: str) -> List[Dict[str, Any]]:
        """
        Generate queries for trend insights.
        
        Args:
            chart_type: Type of chart
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # Different queries based on chart type
        if chart_type in ['line', 'bar']:
            # Query for points in sequence (for trends)
            queries.append({
                'type': 'sequence_points',
                'query': """
                MATCH (c:Chart)-[:HAS_SERIES]->(s:Series)-[:HAS_POINT]->(p1:DataPoint)
                MATCH (p1)-[r:NEXT]->(p2:DataPoint)
                WHERE c.graph_id = $graph_id
                RETURN s.name as series, 
                       p1.x as x1, p1.y as y1, 
                       p2.x as x2, p2.y as y2,
                       r.change as change, r.change_pct as change_pct
                ORDER BY s.name, p1.x
                """
            })
            
            # Query for trend statistics
            queries.append({
                'type': 'trend_stats',
                'query': """
                MATCH (c:Chart)-[:HAS_SERIES]->(s:Series)-[:HAS_POINT]->(p1:DataPoint)
                MATCH (p1)-[r:NEXT]->(p2:DataPoint)
                WHERE c.graph_id = $graph_id
                WITH s.name as series, 
                     count(r) as steps,
                     sum(CASE WHEN r.change > 0 THEN 1 ELSE 0 END) as increases,
                     sum(CASE WHEN r.change < 0 THEN 1 ELSE 0 END) as decreases,
                     sum(CASE WHEN r.change = 0 THEN 1 ELSE 0 END) as unchanged,
                     avg(r.change) as avg_change,
                     avg(r.change_pct) as avg_change_pct
                RETURN series, steps, increases, decreases, unchanged, 
                       avg_change, avg_change_pct,
                       CASE 
                         WHEN increases > decreases THEN 'UPWARD' 
                         WHEN decreases > increases THEN 'DOWNWARD'
                         ELSE 'STABLE'
                       END as trend_direction
                """
            })
            
            # If no series are present, use direct chart-to-point relationship
            queries.append({
                'type': 'direct_points_sequence',
                'query': """
                MATCH (c:Chart)-[:HAS_POINT]->(p1:DataPoint)
                MATCH (p1)-[r:NEXT]->(p2:DataPoint)
                WHERE c.graph_id = $graph_id
                RETURN p1.x as x1, p1.y as y1, 
                       p2.x as x2, p2.y as y2,
                       r.change as change, r.change_pct as change_pct
                ORDER BY p1.x
                """
            })
        
        elif chart_type in ['pie', 'scatter']:
            # For non-time series charts, look at relative values
            queries.append({
                'type': 'value_distribution',
                'query': """
                MATCH (c:Chart)-[:HAS_SEGMENT]->(s:Segment)
                WHERE c.graph_id = $graph_id
                RETURN s.name as category, s.value as value, s.percentage as percentage
                ORDER BY s.percentage DESC
                """
            })
            
            # Also try with categories if segments not found
            queries.append({
                'type': 'category_distribution',
                'query': """
                MATCH (c:Chart)-[:HAS_CATEGORY]->(cat:Category)
                WHERE c.graph_id = $graph_id
                RETURN cat.name as category, cat.value as value
                ORDER BY cat.value DESC
                """
            })
        
        # Add query for statistics regardless of chart type
        queries.append({
            'type': 'statistics',
            'query': """
            MATCH (c:Chart)-[:HAS_STATISTICS]->(stats:Statistics)-[:HAS_STATISTIC]->(stat:Statistic)
            WHERE c.graph_id = $graph_id
            RETURN stat.name as statistic, stat.column as column, stat.value as value
            """
        })
        
        return queries
    
    def _generate_comparison_queries(self, chart_type: str) -> List[Dict[str, Any]]:
        """
        Generate queries for comparison insights.
        
        Args:
            chart_type: Type of chart
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # Find explicit comparison relationships
        queries.append({
            'type': 'explicit_comparisons',
            'query': """
            MATCH (c:Chart)-[:HAS_CATEGORY|HAS_SEGMENT]->(n1)
            MATCH (n1)-[r:GREATER_THAN|LESS_THAN|EQUAL_TO]->(n2)
            WHERE c.graph_id = $graph_id
            RETURN n1.name as source, n2.name as target, type(r) as relation,
                   n1.value as source_value, n2.value as target_value
            """
        })
        
        # Find items above and below mean
        queries.append({
            'type': 'mean_comparisons',
            'query': """
            MATCH (c:Chart)-[:HAS_STATISTICS]->(stats:Statistics)-[:HAS_STATISTIC]->(mean:Statistic)
            WHERE c.graph_id = $graph_id AND mean.name = 'mean'
            MATCH (n)-[r:ABOVE_MEAN|BELOW_MEAN]->(mean)
            RETURN n.name as item, n.value as value, mean.value as mean_value,
                   type(r) as relation
            """
        })
        
        # Minimum and maximum values
        queries.append({
            'type': 'extremes',
            'query': """
            MATCH (c:Chart)-[:HAS_STATISTICS]->(stats:Statistics)-[:HAS_STATISTIC]->(stat:Statistic)
            WHERE c.graph_id = $graph_id AND stat.name IN ['minimum', 'maximum']
            MATCH (n)-[r:IS_MIN|IS_MAX]->(stat)
            RETURN n.name as item, n.value as value, stat.name as extreme_type
            """
        })
        
        # Value comparison (for bar/pie charts)
        if chart_type in ['bar', 'pie']:
            queries.append({
                'type': 'value_comparison',
                'query': """
                MATCH (c:Chart)-[:HAS_CATEGORY|HAS_SEGMENT]->(item)
                WHERE c.graph_id = $graph_id
                RETURN item.name as item, item.value as value
                ORDER BY item.value DESC
                """
            })
            
            # Percentage of total (for pie charts)
            if chart_type == 'pie':
                queries.append({
                    'type': 'percentage_comparison',
                    'query': """
                    MATCH (c:Chart)-[:HAS_SEGMENT]->(segment)
                    WHERE c.graph_id = $graph_id
                    RETURN segment.name as item, segment.percentage as percentage
                    ORDER BY segment.percentage DESC
                    """
                })
        
        # Compare series end points (for line charts)
        if chart_type == 'line':
            queries.append({
                'type': 'series_endpoints',
                'query': """
                MATCH (c:Chart)-[:HAS_SERIES]->(s:Series)-[:HAS_POINT]->(start:DataPoint)
                WHERE c.graph_id = $graph_id
                AND NOT (:DataPoint)-[:NEXT]->(start)
                MATCH (s)-[:HAS_POINT]->(end:DataPoint)
                WHERE NOT (end)-[:NEXT]->(:DataPoint)
                RETURN s.name as series, start.x as start_x, start.y as start_y,
                       end.x as end_x, end.y as end_y,
                       end.y - start.y as total_change,
                       CASE WHEN start.y <> 0 
                            THEN (end.y - start.y) / start.y * 100 
                            ELSE NULL 
                       END as total_change_pct
                """
            })
        
        return queries
    
    def _generate_anomaly_queries(self, chart_type: str) -> List[Dict[str, Any]]:
        """
        Generate queries for anomaly insights.
        
        Args:
            chart_type: Type of chart
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # Get statistics for Z-score calculation
        queries.append({
            'type': 'stats_for_zscore',
            'query': """
            MATCH (c:Chart)-[:HAS_STATISTICS]->(stats:Statistics)-[:HAS_STATISTIC]->(stat:Statistic)
            WHERE c.graph_id = $graph_id AND stat.name IN ['mean', 'standard_deviation']
            RETURN stat.name as stat_name, stat.column as column, stat.value as value
            """
        })
        
        # For line/time series charts, look for change anomalies
        if chart_type in ['line', 'bar']:
            queries.append({
                'type': 'change_anomalies',
                'query': """
                MATCH (c:Chart)-[:HAS_SERIES|HAS_POINT]->(s)-[:HAS_POINT|NEXT*1..2]->(p1:DataPoint)-[r:NEXT]->(p2:DataPoint)
                WHERE c.graph_id = $graph_id
                WITH c, s, p1, p2, r,
                     CASE WHEN s:Series THEN s.name ELSE 'main' END as series_name
                MATCH (c)-[:HAS_STATISTICS]->(stats:Statistics)-[:HAS_STATISTIC]->(mean:Statistic)
                WHERE mean.name = 'mean' 
                MATCH (c)-[:HAS_STATISTICS]->(stats)-[:HAS_STATISTIC]->(std:Statistic)
                WHERE std.name = 'standard_deviation'
                WITH series_name, p1, p2, r, mean.value as mean_change, std.value as std_dev_change
                WHERE std_dev_change > 0 AND ABS((r.change - mean_change) / std_dev_change) > 2
                RETURN series_name, p1.x as x1, p1.y as y1, p2.x as x2, p2.y as y2,
                       r.change as change, r.change_pct as change_pct,
                       mean_change, std_dev_change,
                       ABS((r.change - mean_change) / std_dev_change) as z_score
                ORDER BY z_score DESC
                LIMIT 5
                """
            })
        
        # For all chart types, look for value anomalies
        queries.append({
            'type': 'value_anomalies',
            'query': """
            MATCH (c:Chart)-[:HAS_STATISTICS]->(stats:Statistics)
            MATCH (stats)-[:HAS_STATISTIC]->(mean:Statistic {name: 'mean'})
            MATCH (stats)-[:HAS_STATISTIC]->(std:Statistic {name: 'standard_deviation'})
            WHERE c.graph_id = $graph_id AND mean.column = std.column
            WITH c, mean, std
            
            MATCH (c)-[:HAS_CATEGORY|HAS_SEGMENT|HAS_POINT]->(item)
            WHERE item.value IS NOT NULL
            WITH item, mean, std,
                 CASE WHEN std.value > 0 
                      THEN ABS((item.value - mean.value) / std.value)
                      ELSE 0
                 END as z_score
            WHERE z_score > 2
            RETURN item.name as item, item.value as value,
                   mean.value as mean, std.value as std_dev,
                   z_score
            ORDER BY z_score DESC
            LIMIT 5
            """
        })
        
        # For line charts, look for pattern breaks
        if chart_type == 'line':
            queries.append({
                'type': 'pattern_breaks',
                'query': """
                MATCH (c:Chart)-[:HAS_SERIES|HAS_POINT]->(s)-[:HAS_POINT|NEXT*1..2]->(p1:DataPoint)-[r1:NEXT]->(p2:DataPoint)-[r2:NEXT]->(p3:DataPoint)
                WHERE c.graph_id = $graph_id
                WITH c, s, p1, p2, p3, r1, r2,
                     CASE WHEN s:Series THEN s.name ELSE 'main' END as series_name,
                     SIGN(r1.change) as sign1, SIGN(r2.change) as sign2
                WHERE sign1 <> sign2 AND sign1 <> 0 AND sign2 <> 0
                RETURN series_name, 
                       p1.x as x1, p1.y as y1, 
                       p2.x as x2, p2.y as y2, 
                       p3.x as x3, p3.y as y3,
                       r1.change as change1, r2.change as change2,
                       CASE 
                         WHEN sign1 > 0 AND sign2 < 0 THEN 'PEAK' 
                         WHEN sign1 < 0 AND sign2 > 0 THEN 'VALLEY'
                         ELSE 'UNKNOWN'
                       END as pattern_type
                """
            })
        
        return queries
    
    def _generate_correlation_queries(self, chart_type: str) -> List[Dict[str, Any]]:
        """
        Generate queries for correlation insights.
        
        Args:
            chart_type: Type of chart
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # For scatter plots, get all points for correlation
        if chart_type == 'scatter':
            queries.append({
                'type': 'scatter_correlation',
                'query': """
                MATCH (c:Chart)-[:HAS_POINT]->(p:DataPoint)
                WHERE c.graph_id = $graph_id
                RETURN p.x as x, p.y as y
                ORDER BY p.x
                """
            })
        
        # For line charts with multiple series, compare series
        if chart_type == 'line':
            queries.append({
                'type': 'series_correlation',
                'query': """
                MATCH (c:Chart)-[:HAS_SERIES]->(s1:Series)
                MATCH (c)-[:HAS_SERIES]->(s2:Series)
                WHERE c.graph_id = $graph_id AND id(s1) < id(s2)
                MATCH (s1)-[:HAS_POINT]->(p1:DataPoint)
                MATCH (s2)-[:HAS_POINT]->(p2:DataPoint)
                WHERE p1.x = p2.x
                RETURN s1.name as series1, s2.name as series2,
                       p1.x as x, p1.y as y1, p2.y as y2
                ORDER BY p1.x
                """
            })
            
            # Also calculate correlation coefficients
            queries.append({
                'type': 'series_correlation_stats',
                'query': """
                MATCH (c:Chart)-[:HAS_SERIES]->(s1:Series)
                MATCH (c)-[:HAS_SERIES]->(s2:Series)
                WHERE c.graph_id = $graph_id AND id(s1) < id(s2)
                MATCH (s1)-[:HAS_POINT]->(p1:DataPoint)
                MATCH (s2)-[:HAS_POINT]->(p2:DataPoint)
                WHERE p1.x = p2.x
                WITH s1.name as series1, s2.name as series2,
                     collect(p1.y) as y1_values,
                     collect(p2.y) as y2_values,
                     count(*) as point_count
                WHERE point_count > 2
                
                // This is a simplified correlation calculation in Cypher
                // It doesn't handle all edge cases but works for basic correlation
                WITH series1, series2, y1_values, y2_values, point_count,
                     reduce(sum=0, i IN range(0, size(y1_values)-1) | sum + y1_values[i]) as sum_y1,
                     reduce(sum=0, i IN range(0, size(y2_values)-1) | sum + y2_values[i]) as sum_y2
                
                WITH series1, series2, y1_values, y2_values, point_count, sum_y1, sum_y2,
                     sum_y1 / point_count as avg_y1,
                     sum_y2 / point_count as avg_y2
                
                WITH series1, series2, y1_values, y2_values, point_count, avg_y1, avg_y2,
                     reduce(sum=0, i IN range(0, size(y1_values)-1) | 
                          sum + (y1_values[i] - avg_y1) * (y2_values[i] - avg_y2)) as cov,
                     reduce(sum=0, i IN range(0, size(y1_values)-1) | 
                          sum + (y1_values[i] - avg_y1) * (y1_values[i] - avg_y1)) as var_y1,
                     reduce(sum=0, i IN range(0, size(y2_values)-1) | 
                          sum + (y2_values[i] - avg_y2) * (y2_values[i] - avg_y2)) as var_y2
                
                WITH series1, series2, point_count,
                     CASE WHEN var_y1 * var_y2 > 0
                          THEN cov / sqrt(var_y1 * var_y2)
                          ELSE 0
                     END as correlation
                
                RETURN series1, series2, point_count, correlation,
                       CASE
                         WHEN correlation > 0.7 THEN 'STRONG_POSITIVE'
                         WHEN correlation > 0.3 THEN 'MODERATE_POSITIVE'
                         WHEN correlation > -0.3 THEN 'WEAK_OR_NONE'
                         WHEN correlation > -0.7 THEN 'MODERATE_NEGATIVE'
                         ELSE 'STRONG_NEGATIVE'
                       END as correlation_type
                ORDER BY ABS(correlation) DESC
                """
            })
        
        # For bar/pie charts, look for category correlations
        if chart_type in ['bar', 'pie']:
            queries.append({
                'type': 'category_correlation',
                'query': """
                MATCH (c:Chart)-[:HAS_CATEGORY|HAS_SEGMENT]->(cat1)
                MATCH (c)-[:HAS_CATEGORY|HAS_SEGMENT]->(cat2)
                WHERE c.graph_id = $graph_id AND id(cat1) < id(cat2)
                RETURN cat1.name as category1, cat1.value as value1,
                       cat2.name as category2, cat2.value as value2,
                       CASE 
                         WHEN cat1.value > 0 AND cat2.value > 0
                         THEN cat2.value / cat1.value
                         ELSE 0
                       END as ratio
                ORDER BY ABS(1 - ratio)
                """
            })
        
        return queries
    
    def _generate_generic_queries(self, chart_type: str) -> List[Dict[str, Any]]:
        """
        Generate generic queries that work for any chart type.
        
        Args:
            chart_type: Type of chart
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # Query for chart information
        queries.append({
            'type': 'chart_info',
            'query': """
            MATCH (c:Chart)
            WHERE c.graph_id = $graph_id
            RETURN c.chart_type as chart_type, c.title as title,
                   c.x_label as x_label, c.y_label as y_label
            """
        })
        
        # Query for all data elements
        queries.append({
            'type': 'all_data',
            'query': """
            MATCH (c:Chart)-[r]->(n)
            WHERE c.graph_id = $graph_id
            RETURN type(r) as relation, labels(n) as node_labels,
                   count(n) as count
            """
        })
        
        # Query for statistics
        queries.append({
            'type': 'statistics',
            'query': """
            MATCH (c:Chart)-[:HAS_STATISTICS]->(stats:Statistics)-[:HAS_STATISTIC]->(stat:Statistic)
            WHERE c.graph_id = $graph_id
            RETURN stat.name as statistic, stat.column as column, stat.value as value
            """
        })
        
        return queries
