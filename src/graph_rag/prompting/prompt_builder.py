"""
Prompt Builder for GraphRAG.
Builds prompts for large language models to generate insights.
"""

import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Builds prompts for the LLM to generate insights from graph data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the prompt builder.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
    
    def build_insight_prompt(self,
                            chart_type: str,
                            chart_title: str,
                            insight_type: str,
                            query_results: List[Dict[str, Any]],
                            context_nodes: Dict[str, Any]) -> str:
        """
        Build a prompt for generating insights.
        
        Args:
            chart_type: Type of chart (bar, line, pie, etc.)
            chart_title: Title of the chart
            insight_type: Type of insight to generate
            query_results: Results from Cypher queries
            context_nodes: Context nodes from traversal
            
        Returns:
            Prompt string for the LLM
        """
        # Basic prompt structure
        prompt = f"""You are an expert data analyst generating insights from chart data. 
Provide meaningful and accurate insights about the following {chart_type.upper()} chart: "{chart_title}".

Focus on generating {insight_type.upper()} insights.

I will provide you with:
1. Chart information
2. Key data points 
3. Statistical information
4. Graph analysis results

For each insight:
- Start with "INSIGHT:" followed by a clear, concise statement
- Add "CONFIDENCE:" with a value between 0 and 1 (or 0-100%) indicating how confident you are
- Add "EXPLANATION:" with a brief explanation of the insight
- Reference specific data provided below

Provide 2-4 high-quality insights. Only include insights that are directly supported by the data.

"""
        
        # Add chart information
        prompt += f"""
## CHART INFORMATION
Chart Type: {chart_type}
Chart Title: {chart_title}
"""
        
        # Add context from traversal engine based on chart type
        prompt += self._format_context_by_chart_type(chart_type, context_nodes)
        
        # Add query results
        prompt += "\n## QUERY RESULTS\n"
        for query_result in query_results:
            query_type = query_result.get('type', 'unknown')
            results = query_result.get('results', [])
            
            if not results:
                continue
            
            prompt += f"\n### {query_type.upper()}\n"
            
            # Format results based on query type
            if query_type in ['sequence_points', 'direct_points_sequence']:
                # Format sequence points (for trend analysis)
                prompt += "Points in sequence:\n"
                for idx, result in enumerate(results[:10]):  # Limit to 10 rows
                    prompt += f"- Point {idx+1}: x1={result.get('x1')}, y1={result.get('y1')} → x2={result.get('x2')}, y2={result.get('y2')}, change={result.get('change')}, change_pct={result.get('change_pct')}%\n"
                
                if len(results) > 10:
                    prompt += f"... and {len(results) - 10} more points\n"
            
            elif query_type in ['trend_stats']:
                # Format trend statistics
                prompt += "Trend statistics:\n"
                for result in results:
                    series = result.get('series', 'main')
                    steps = result.get('steps', 0)
                    increases = result.get('increases', 0)
                    decreases = result.get('decreases', 0)
                    unchanged = result.get('unchanged', 0)
                    avg_change = result.get('avg_change', 0)
                    avg_change_pct = result.get('avg_change_pct', 0)
                    direction = result.get('trend_direction', 'UNKNOWN')
                    
                    prompt += f"- Series: {series}, Direction: {direction}\n"
                    prompt += f"  Steps: {steps}, Increases: {increases}, Decreases: {decreases}, Unchanged: {unchanged}\n"
                    prompt += f"  Avg Change: {avg_change}, Avg Change %: {avg_change_pct}%\n"
            
            elif query_type in ['value_distribution', 'category_distribution', 'value_comparison']:
                # Format value distributions
                prompt += "Value distribution:\n"
                for idx, result in enumerate(results[:10]):  # Limit to 10 rows
                    category = result.get('category', result.get('item', f'Item {idx+1}'))
                    value = result.get('value', 0)
                    percentage = result.get('percentage', None)
                    
                    if percentage is not None:
                        prompt += f"- {category}: {value} ({percentage}%)\n"
                    else:
                        prompt += f"- {category}: {value}\n"
                
                if len(results) > 10:
                    prompt += f"... and {len(results) - 10} more items\n"
            
            elif query_type in ['statistics', 'stats_for_zscore']:
                # Format statistics
                prompt += "Statistical values:\n"
                for result in results:
                    stat_name = result.get('stat_name', result.get('statistic', 'unknown'))
                    column = result.get('column', 'default')
                    value = result.get('value', 0)
                    
                    prompt += f"- {stat_name.capitalize()} ({column}): {value}\n"
            
            elif query_type in ['explicit_comparisons']:
                # Format comparisons
                prompt += "Explicit comparisons:\n"
                for result in results:
                    source = result.get('source', 'unknown')
                    target = result.get('target', 'unknown')
                    relation = result.get('relation', 'RELATED')
                    source_value = result.get('source_value', 0)
                    target_value = result.get('target_value', 0)
                    
                    prompt += f"- {source} ({source_value}) {relation} {target} ({target_value})\n"
            
            elif query_type in ['mean_comparisons']:
                # Format mean comparisons
                prompt += "Comparisons to mean:\n"
                for result in results:
                    item = result.get('item', 'unknown')
                    value = result.get('value', 0)
                    mean_value = result.get('mean_value', 0)
                    relation = result.get('relation', 'UNKNOWN')
                    
                    prompt += f"- {item} ({value}) is {relation} the mean ({mean_value})\n"
            
            elif query_type in ['extremes']:
                # Format extremes
                prompt += "Extreme values:\n"
                for result in results:
                    item = result.get('item', 'unknown')
                    value = result.get('value', 0)
                    extreme_type = result.get('extreme_type', 'unknown')
                    
                    prompt += f"- {item} ({value}) is the {extreme_type} value\n"
            
            elif query_type in ['series_endpoints']:
                # Format series endpoints
                prompt += "Series endpoints:\n"
                for result in results:
                    series = result.get('series', 'main')
                    start_y = result.get('start_y', 0)
                    end_y = result.get('end_y', 0)
                    total_change = result.get('total_change', 0)
                    total_change_pct = result.get('total_change_pct', 0)
                    
                    prompt += f"- Series {series}: Start={start_y}, End={end_y}, Total Change={total_change} ({total_change_pct}%)\n"
            
            elif query_type in ['change_anomalies', 'value_anomalies']:
                # Format anomalies
                prompt += "Detected anomalies:\n"
                for result in results:
                    if 'item' in result:
                        # Value anomalies
                        item = result.get('item', 'unknown')
                        value = result.get('value', 0)
                        mean = result.get('mean', 0)
                        std_dev = result.get('std_dev', 0)
                        z_score = result.get('z_score', 0)
                        
                        prompt += f"- {item} has value {value} (mean={mean}, std_dev={std_dev}, z-score={z_score})\n"
                    else:
                        # Change anomalies
                        series = result.get('series_name', 'main')
                        x1 = result.get('x1', 0)
                        y1 = result.get('y1', 0)
                        x2 = result.get('x2', 0)
                        y2 = result.get('y2', 0)
                        change = result.get('change', 0)
                        change_pct = result.get('change_pct', 0)
                        z_score = result.get('z_score', 0)
                        
                        prompt += f"- Series {series}: Unusual change from {y1} to {y2} at x={x2} (change={change}, z-score={z_score})\n"
            
            elif query_type in ['pattern_breaks']:
                # Format pattern breaks
                prompt += "Pattern breaks:\n"
                for result in results:
                    series = result.get('series_name', 'main')
                    pattern_type = result.get('pattern_type', 'UNKNOWN')
                    x2 = result.get('x2', 0)
                    y2 = result.get('y2', 0)
                    
                    prompt += f"- Series {series}: {pattern_type} detected at x={x2}, y={y2}\n"
            
            elif query_type in ['scatter_correlation', 'series_correlation']:
                # Format data for correlation
                prompt += f"Correlation data points: {len(results)} points\n"
                if len(results) <= 10:
                    for idx, result in enumerate(results):
                        if 'series1' in result:
                            # Series correlation
                            x = result.get('x', idx)
                            y1 = result.get('y1', 0)
                            y2 = result.get('y2', 0)
                            series1 = result.get('series1', 'Series 1')
                            series2 = result.get('series2', 'Series 2')
                            
                            prompt += f"- Point {idx+1}: x={x}, {series1}={y1}, {series2}={y2}\n"
                        else:
                            # Scatter correlation
                            x = result.get('x', 0)
                            y = result.get('y', 0)
                            
                            prompt += f"- Point {idx+1}: x={x}, y={y}\n"
                else:
                    prompt += f"(Too many points to list individually)\n"
            
            elif query_type in ['series_correlation_stats']:
                # Format correlation statistics
                prompt += "Correlation statistics:\n"
                for result in results:
                    series1 = result.get('series1', 'Series 1')
                    series2 = result.get('series2', 'Series 2')
                    correlation = result.get('correlation', 0)
                    corr_type = result.get('correlation_type', 'UNKNOWN')
                    point_count = result.get('point_count', 0)
                    
                    prompt += f"- {series1} vs {series2}: correlation={correlation:.3f} ({corr_type}) based on {point_count} points\n"
            
            elif query_type in ['category_correlation']:
                # Format category correlations
                prompt += "Category correlations:\n"
                for result in results:
                    cat1 = result.get('category1', 'Category 1')
                    val1 = result.get('value1', 0)
                    cat2 = result.get('category2', 'Category 2')
                    val2 = result.get('value2', 0)
                    ratio = result.get('ratio', 0)
                    
                    prompt += f"- {cat1} ({val1}) vs {cat2} ({val2}): ratio={ratio:.2f}\n"
            
            else:
                # Generic formatting for other query types
                if len(results) > 0 and isinstance(results[0], dict):
                    # Get columns from first result
                    columns = list(results[0].keys())
                    prompt += f"Columns: {', '.join(columns)}\n"
                    
                    # Add each row
                    for idx, result in enumerate(results[:10]):  # Limit to 10 rows
                        values = [str(result.get(col, '')) for col in columns]
                        prompt += f"- Row {idx+1}: {', '.join(values)}\n"
                    
                    if len(results) > 10:
                        prompt += f"... and {len(results) - 10} more rows\n"
                else:
                    # Unstructured results
                    prompt += json.dumps(results, indent=2)[:500] + "...\n"
        
        # Add guidance based on insight type
        prompt += self._add_insight_type_guidance(insight_type)
        
        # Limit size
        if len(prompt) > 6000:
            logger.warning(f"Prompt exceeds 6000 chars ({len(prompt)}), truncating")
            prompt = prompt[:6000] + "...\n"
        
        return prompt
    
    def _format_context_by_chart_type(self, chart_type: str, context: Dict[str, Any]) -> str:
        """
        Format context based on chart type.
        
        Args:
            chart_type: Type of chart
            context: Context dictionary from traversal
            
        Returns:
            Formatted context string
        """
        result = "\n## CHART DATA CONTEXT\n"
        
        # Chart metadata
        chart_data = context.get('chart', {})
        if chart_data:
            result += f"Title: {chart_data.get('title', 'Untitled')}\n"
            result += f"X-axis: {chart_data.get('x_label', '')}\n"
            result += f"Y-axis: {chart_data.get('y_label', '')}\n"
        
        # Process different context structures based on chart type
        if chart_type in ['line', 'area']:
            # Time series data
            series_list = context.get('series', [])
            if series_list:
                result += "\nSeries data:\n"
                for series in series_list:
                    series_name = series.get('name', 'unnamed')
                    points = series.get('points', [])
                    
                    result += f"- Series: {series_name}, {len(points)} points\n"
                    
                    if len(points) > 0:
                        # Show first and last few points
                        for i, point in enumerate(points[:3]):
                            result += f"  - Point {i+1}: x={point.get('x', i)}, y={point.get('y', 0)}\n"
                        
                        if len(points) > 6:
                            result += "  - ...\n"
                        
                        for i, point in enumerate(points[-3:]):
                            idx = len(points) - 3 + i
                            result += f"  - Point {idx+1}: x={point.get('x', idx)}, y={point.get('y', 0)}\n"
        
        elif chart_type in ['bar', 'pie']:
            # Categorical data
            categories = context.get('categories', [])
            if categories:
                result += "\nCategories:\n"
                for i, category in enumerate(categories):
                    name = category.get('name', f'Category {i+1}')
                    value = category.get('value', 0)
                    percentage = category.get('percentage', None)
                    
                    if percentage is not None:
                        result += f"- {name}: {value} ({percentage}%)\n"
                    else:
                        result += f"- {name}: {value}\n"
        
        elif chart_type == 'scatter':
            # Scatter plot data
            series_pairs = context.get('series_pairs', [])
            if series_pairs and len(series_pairs) > 0:
                result += "\nScatter plot data:\n"
                pair = series_pairs[0]  # Usually just one pair for scatter plots
                
                x_series = pair.get('x_series', 'X')
                y_series = pair.get('y_series', 'Y')
                points = pair.get('points', [])
                
                result += f"- X-axis: {x_series}, Y-axis: {y_series}, {len(points)} points\n"
                
                # Show sample points
                for i, point in enumerate(points[:5]):
                    result += f"  - Point {i+1}: x={point.get('x', 0)}, y={point.get('y', 0)}\n"
                
                if len(points) > 5:
                    result += f"  - ... and {len(points) - 5} more points\n"
        
        # Add statistics if available
        stats = context.get('statistics', {})
        if stats:
            result += "\nStatistics:\n"
            for stat_name, stat_values in stats.items():
                result += f"- {stat_name.capitalize()}:\n"
                
                if isinstance(stat_values, dict):
                    for column, value in stat_values.items():
                        result += f"  - {column}: {value}\n"
                else:
                    result += f"  - {stat_values}\n"
        
        return result
    
    def _add_insight_type_guidance(self, insight_type: str) -> str:
        """
        Add specific guidance based on insight type.
        
        Args:
            insight_type: Type of insight
            
        Returns:
            Guidance string
        """
        result = "\n## GUIDANCE FOR INSIGHT GENERATION\n"
        
        if insight_type == 'trend':
            result += """
For TREND insights, focus on:
- Overall direction of data over time (increasing, decreasing, stable)
- Rate of change (acceleration, deceleration)
- Seasonality or cyclical patterns
- Turning points where trends change direction
- Comparison of trends between different series

Your insights should highlight important patterns in how the data changes over time.
"""
        
        elif insight_type == 'comparison':
            result += """
For COMPARISON insights, focus on:
- Differences between categories or data points
- Relative proportions or percentages
- Ranking of values (highest to lowest)
- Identification of outliers compared to others
- Differences between actual values and statistical measures (mean, median)

Your insights should highlight significant differences between data elements.
"""
        
        elif insight_type == 'anomaly':
            result += """
For ANOMALY insights, focus on:
- Data points that deviate significantly from the norm
- Sudden changes or spikes in otherwise stable data
- Values that are statistically unlikely (beyond 2 standard deviations)
- Pattern breaks or unexpected behavior
- Missing or unusual values

Your insights should identify and explain unusual observations in the data.
"""
        
        elif insight_type == 'correlation':
            result += """
For CORRELATION insights, focus on:
- Relationships between different variables or series
- Patterns where variables move together or in opposite directions
- Strength of relationships (strong, moderate, weak)
- Potential causal relationships (though be careful not to claim causation)
- Areas where expected correlations are absent

Your insights should highlight meaningful relationships between different aspects of the data.
"""
        
        else:
            result += """
For your insights:
- Prioritize accuracy over speculation
- Base insights directly on the data provided
- Explain the significance of what you observe
- Consider the broader context of the chart
- Highlight the most important patterns or findings

Aim for 2-4 high-quality insights that provide real value to someone analyzing this chart.
"""
        
        return result
