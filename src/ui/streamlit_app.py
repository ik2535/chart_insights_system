"""
Streamlit UI for Chart Insights System.
Provides a web interface for analyzing charts and viewing insights.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yaml
from PIL import Image
import io
import json
from typing import Dict, List, Any, Optional, Tuple

# Import chart insights modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.chart_analysis import ChartAnalyzer
from src.knowledge_graph.builder import ChartKnowledgeGraphBuilder
from src.graph_rag import ChartGraphRAG
from src.insights_generation import InsightGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chart_insights.log')
    ]
)

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Return default configuration
        return {
            'chart_analysis': {
                'ocr_config': '--psm 6',
                'supported_types': ['bar', 'line', 'pie', 'scatter']
            }
        }

def get_chart_type_options():
    """Get chart type options."""
    return ['bar', 'line', 'pie', 'scatter']

def load_sample_data(chart_type):
    """Load sample data for specified chart type."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'samples')
    
    # Define default data for each chart type
    if chart_type == 'bar':
        data = pd.DataFrame({
            'Category': ['A', 'B', 'C', 'D', 'E'],
            'Value': [10, 25, 15, 30, 20]
        })
    elif chart_type == 'line':
        data = pd.DataFrame({
            'Month': range(1, 13),
            'Sales': [10, 15, 13, 17, 20, 25, 30, 35, 30, 25, 20, 15]
        })
    elif chart_type == 'pie':
        data = pd.DataFrame({
            'Category': ['A', 'B', 'C', 'D'],
            'Value': [30, 20, 25, 25]
        })
    elif chart_type == 'scatter':
        np.random.seed(42)
        x = np.random.rand(50) * 10
        y = 2 * x + np.random.randn(50) * 2
        data = pd.DataFrame({
            'X': x,
            'Y': y
        })
    else:
        data = pd.DataFrame()
    
    return data

def create_metadata(chart_type, title=None, x_label=None, y_label=None):
    """Create chart metadata from inputs."""
    metadata = {
        'title': title or f"{chart_type.capitalize()} Chart",
        'x_axis_label': x_label or ('Category' if chart_type in ['bar', 'pie'] else 'X'),
        'y_axis_label': y_label or ('Value' if chart_type in ['bar', 'pie'] else 'Y'),
    }
    return metadata

def display_insights(insights):
    """Display insights in a formatted way."""
    if not insights:
        st.warning("No insights generated.")
        return
    
    # Display total insights
    st.subheader(f"Generated {len(insights)} Insights")
    
    # Group insights by type
    insight_types = {}
    for insight in insights:
        insight_type = insight.get('type', 'Other')
        if insight_type not in insight_types:
            insight_types[insight_type] = []
        insight_types[insight_type].append(insight)
    
    # Display insights by type
    for insight_type, type_insights in insight_types.items():
        with st.expander(f"{insight_type.capitalize()} Insights ({len(type_insights)})", expanded=True):
            for i, insight in enumerate(type_insights, 1):
                confidence = insight.get('confidence', 0)
                confidence_str = f"{confidence:.1%}" if isinstance(confidence, float) else f"{confidence}"
                
                # Use different colors based on confidence
                if confidence >= 0.8:
                    confidence_color = "green"
                elif confidence >= 0.5:
                    confidence_color = "orange"
                else:
                    confidence_color = "red"
                
                st.markdown(f"### Insight {i}: _{insight.get('text', 'No text')}_ ")
                st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{confidence_str}</span>", unsafe_allow_html=True)
                
                if 'explanation' in insight:
                    st.markdown(f"**Explanation:** {insight['explanation']}")
                
                if 'supporting_data' in insight and insight['supporting_data']:
                    with st.expander("Supporting Data"):
                        for data_point in insight['supporting_data']:
                            st.text(data_point)

def visualize_knowledge_graph(graph, chart_type):
    """Visualize knowledge graph using NetworkX and matplotlib."""
    if not graph or not graph.nodes():
        st.warning("No graph to visualize.")
        return
    
    try:
        import networkx as nx
        
        # Create a spring layout
        pos = nx.spring_layout(graph, k=0.3, iterations=50)
        
        # Create a figure
        plt.figure(figsize=(10, 6))
        
        # Node colors based on type
        node_colors = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'unknown')
            if node_type == 'chart':
                node_colors.append('royalblue')
            elif node_type in ['category', 'segment']:
                node_colors.append('green')
            elif node_type == 'data_point':
                node_colors.append('orange')
            elif node_type == 'statistics':
                node_colors.append('purple')
            elif node_type == 'statistic':
                node_colors.append('mediumpurple')
            else:
                node_colors.append('gray')
        
        # Draw the graph
        nx.draw(graph, pos, with_labels=False, node_color=node_colors, 
                node_size=100, alpha=0.8, linewidths=0.5, width=0.5)
        
        # Add node labels to only important nodes
        labels = {}
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', '')
            if node_type in ['chart', 'statistics']:
                labels[node] = node_type
            elif node_type in ['category', 'segment']:
                labels[node] = graph.nodes[node].get('name', node)[:10]
        
        nx.draw_networkx_labels(graph, pos, labels, font_size=8)
        
        # Add a title
        plt.title(f"Knowledge Graph for {chart_type.capitalize()} Chart")
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Display in Streamlit
        st.image(buf, caption=f"Knowledge Graph - {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Close plot to avoid memory issues
        plt.close()
        
    except Exception as e:
        st.error(f"Error visualizing graph: {str(e)}")

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Chart Insights System",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Chart Insights Generation System")
    st.markdown("Upload chart data or images to generate automated insights using GraphRAG.")
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    if 'insights' not in st.session_state:
        st.session_state.insights = None
    if 'chart_data' not in st.session_state:
        st.session_state.chart_data = None
    if 'chart_type' not in st.session_state:
        st.session_state.chart_type = None
    if 'chart_image' not in st.session_state:
        st.session_state.chart_image = None
    if 'knowledge_graph' not in st.session_state:
        st.session_state.knowledge_graph = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Chart Configuration")
        
        # Input method selection
        input_method = st.radio("Input Method", ["Generate Sample Data", "Upload Data", "Upload Chart Image"])
        
        # Chart type selection
        chart_types = get_chart_type_options()
        chart_type = st.selectbox("Chart Type", chart_types, disabled=(input_method == "Upload Chart Image"))
        
        # Chart metadata
        st.subheader("Chart Metadata")
        chart_title = st.text_input("Chart Title", value="")
        x_label = st.text_input("X-Axis Label", value="")
        y_label = st.text_input("Y-Axis Label", value="")
        
        # Insight type selection
        st.subheader("Insight Types")
        insight_types = st.multiselect(
            "Select Insight Types",
            ["trend", "comparison", "anomaly", "correlation"],
            default=["trend", "comparison"]
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            graph_depth = st.slider("Graph Traversal Depth", 1, 5, 3)
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
            max_insights = st.slider("Max Insights Per Type", 1, 10, 3)
        
        # Update configuration
        st.session_state.config['graph_rag']['max_hops'] = graph_depth
        st.session_state.config['insights']['confidence_threshold'] = confidence_threshold
        st.session_state.config['insights']['max_insights_per_chart'] = max_insights
        st.session_state.config['insights']['types'] = insight_types
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    # Handle input methods
    with col1:
        st.header("Chart Data & Visualization")
        
        chart_data = None
        if input_method == "Generate Sample Data":
            chart_data = load_sample_data(chart_type)
            st.dataframe(chart_data, use_container_width=True)
            st.session_state.chart_type = chart_type
            st.session_state.chart_data = chart_data
            st.session_state.chart_image = None
            
        elif input_method == "Upload Data":
            uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        chart_data = pd.read_csv(uploaded_file)
                    else:
                        chart_data = pd.read_excel(uploaded_file)
                    
                    st.dataframe(chart_data, use_container_width=True)
                    st.session_state.chart_type = chart_type
                    st.session_state.chart_data = chart_data
                    st.session_state.chart_image = None
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            
        elif input_method == "Upload Chart Image":
            uploaded_image = st.file_uploader("Upload Chart Image", type=["png", "jpg", "jpeg"])
            
            if uploaded_image is not None:
                try:
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Chart", use_column_width=True)
                    st.session_state.chart_image = image
                    st.session_state.chart_data = None
                    # Chart type will be detected by the analyzer
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
        
        # Create chart visualization if data is available
        if st.session_state.chart_data is not None and st.session_state.chart_type is not None:
            chart_type = st.session_state.chart_type
            chart_data = st.session_state.chart_data
            
            st.subheader("Chart Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == 'bar':
                if len(chart_data.columns) >= 2:
                    x_col, y_col = chart_data.columns[0], chart_data.columns[1]
                    chart_data.plot(kind='bar', x=x_col, y=y_col, ax=ax)
            
            elif chart_type == 'line':
                if len(chart_data.columns) >= 2:
                    x_col, y_col = chart_data.columns[0], chart_data.columns[1]
                    chart_data.plot(kind='line', x=x_col, y=y_col, ax=ax)
            
            elif chart_type == 'pie':
                if len(chart_data.columns) >= 2:
                    labels = chart_data[chart_data.columns[0]]
                    values = chart_data[chart_data.columns[1]]
                    ax.pie(values, labels=labels, autopct='%1.1f%%')
                    ax.axis('equal')
            
            elif chart_type == 'scatter':
                if len(chart_data.columns) >= 2:
                    x_col, y_col = chart_data.columns[0], chart_data.columns[1]
                    chart_data.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
            
            # Set labels
            if chart_title:
                ax.set_title(chart_title)
            if x_label:
                ax.set_xlabel(x_label)
            if y_label:
                ax.set_ylabel(y_label)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    # Generate insights
    analyze_button = st.button("Generate Insights")
    
    if analyze_button:
        with st.spinner("Analyzing chart and generating insights..."):
            try:
                config = st.session_state.config
                
                # Create metadata
                metadata = create_metadata(
                    chart_type=st.session_state.chart_type if st.session_state.chart_type else "unknown",
                    title=chart_title,
                    x_label=x_label,
                    y_label=y_label
                )
                
                # Initialize components
                analyzer = ChartAnalyzer(config)
                kg_builder = ChartKnowledgeGraphBuilder(config)
                graph_rag = ChartGraphRAG(config)
                insight_generator = InsightGenerator(config)
                
                if st.session_state.chart_data is not None:
                    # Process data
                    chart_data = st.session_state.chart_data
                    chart_type = st.session_state.chart_type
                    
                    # Build knowledge graph
                    graph = kg_builder.build_graph(chart_data, chart_type, metadata)
                    st.session_state.knowledge_graph = graph
                    
                    # Generate insights
                    insights = insight_generator.generate_insights(chart_data, chart_type, metadata)
                    st.session_state.insights = insights
                
                elif st.session_state.chart_image is not None:
                    # Process image
                    image_array = np.array(st.session_state.chart_image)
                    
                    # Analyze chart
                    result = analyzer.analyze_chart(chart_image=image_array)
                    
                    # Extract data and metadata
                    chart_data = result['data']
                    chart_type = result['chart_type']
                    extracted_metadata = result['metadata']
                    
                    # Merge with user-provided metadata
                    merged_metadata = {**extracted_metadata}
                    if chart_title:
                        merged_metadata['title'] = chart_title
                    if x_label:
                        merged_metadata['x_axis_label'] = x_label
                    if y_label:
                        merged_metadata['y_axis_label'] = y_label
                    
                    # Build knowledge graph
                    graph = kg_builder.build_graph(chart_data, chart_type, merged_metadata)
                    st.session_state.knowledge_graph = graph
                    
                    # Generate insights
                    insights = insight_generator.generate_insights(chart_data, chart_type, merged_metadata)
                    st.session_state.insights = insights
                    
                    # Update session state
                    st.session_state.chart_type = chart_type
                    st.session_state.chart_data = chart_data
                
                else:
                    st.error("No data or image provided.")
            
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                logger.error(f"Error generating insights: {e}", exc_info=True)
    
    # Display insights and knowledge graph
    with col2:
        st.header("Generated Insights")
        
        if st.session_state.insights:
            display_insights(st.session_state.insights)
        else:
            st.info("Click 'Generate Insights' to analyze the chart.")
        
        st.header("Knowledge Graph Visualization")
        if st.session_state.knowledge_graph:
            visualize_knowledge_graph(
                st.session_state.knowledge_graph, 
                st.session_state.chart_type or "unknown"
            )
        else:
            st.info("Knowledge graph will be displayed here after analysis.")

if __name__ == "__main__":
    main()
