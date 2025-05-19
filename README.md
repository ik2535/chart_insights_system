# Chart Insights Generation System

The Chart Insights Generation System is an advanced analytics tool that automatically generates meaningful insights from chart data using Knowledge Graphs and Graph-based Retrieval Augmented Generation (GraphRAG).

## Features

- **Chart Analysis**: Extract data from chart images using computer vision or process raw data directly
- **Knowledge Graph Creation**: Build rich knowledge graphs representing chart entities and relationships
- **GraphRAG**: Use graph-based retrieval augmented generation to create contextual insights
- **Insight Generation**: Automatically generate trend, comparison, anomaly, and correlation insights
- **User Interface**: Interactive Streamlit UI for easy visualization and exploration

## Getting Started

### Prerequisites

- Python 3.10+
- Neo4j 5.x
- LLM API access (OpenAI, Anthropic, etc.)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chart_insights_system
   ```

2. **Set up a virtual environment** (see [Environment Setup](docs/ENVIRONMENT_SETUP.md) for detailed instructions)
   ```bash
   # Create a virtual environment outside the project directory (recommended)
   python -m venv ~/venvs/chart_insights_venv
   
   # Activate the virtual environment
   # On Windows
   ~/venvs/chart_insights_venv/Scripts/activate
   
   # On macOS/Linux
   source ~/venvs/chart_insights_venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Neo4j (see below)

5. Configure environment variables:
   ```bash
   # Copy the example env file
   cp .env.example .env
   
   # Edit with your settings
   # Edit .env with your preferred editor
   ```

## Architecture

The system implements a sophisticated knowledge graph-based architecture that combines the power of traditional data visualization with advanced graph-based retrieval augmented generation:

```
┌───────────────────────┐
┌───────────────┐                    │ Knowledge Graph Layer │
│ Data Sources  │                    │                       │
│ - CSV/Excel   │                    │ ┌─────────────────┐   │
│ - Databases   │◄───────────────────┼─┤Entity Extraction│   │
│ - APIs        │                    │ └─────────────────┘   │
└───────┬───────┘                    │         │             │
        │                            │         ▼             │
        ▼                            │ ┌─────────────────┐   │
┌───────────────┐                    │ │Relationship     │   │
│ Data          │                    │ │Mapping          │   │
│ Processing    │───────────────────►│ └─────────────────┘   │
└───────┬───────┘                    │         │             │
        │                            │         ▼             │
        ▼                            │ ┌─────────────────┐   │
┌───────────────┐                    │ │Graph Database   │   │
│ Chart         │                    │ │(Neo4j/FalkorDB) │   │
│ Generation    │───────────────────►│ └─────────────────┘   │
└───────┬───────┘                    └───────────┬───────────┘
        │                                        │
        ▼                                        ▼
┌───────────────────────────────────────────────────────────┐
│                     GraphRAG Engine                        │
│                                                            │
│  ┌─────────────────┐       ┌─────────────────────────┐    │
│  │Chart Analysis   │◄─────►│Knowledge Graph Traversal │    │
│  └─────────────────┘       └─────────────────────────┘    │
│            │                            │                  │
│            ▼                            ▼                  │
│  ┌─────────────────┐       ┌─────────────────────────┐    │
│  │Pattern Detection│       │Context Augmentation     │    │
│  └─────────────────┘       └─────────────────────────┘    │
│            │                            │                  │
│            └────────────────┬───────────┘                  │
│                             │                              │
└─────────────────────────────┼──────────────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │Insights Generation  │
                   │with LLM             │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │Client-Facing        │
                   │Interactive Reports  │
                   └─────────────────────┘
```

## Neo4j Setup

The system uses Neo4j for persistent graph storage. Follow these steps to set up Neo4j:

### Option 1: Neo4j Desktop (Recommended)

1. Download Neo4j Desktop from https://neo4j.com/download/
2. Install and launch Neo4j Desktop
3. Click "New" to create a new project
4. Click "Add" → "Local DBMS"
5. Set:
   - Name: `chart-insights`
   - Password: `password` (or update config.yaml with your password)
   - Version: Latest 5.x version
6. Click "Create"
7. Click the play button to start the database
8. The connection URI will be `bolt://localhost:7687`

### Option 2: Neo4j Community Server

1. Download Neo4j Community Server from https://neo4j.com/download/
2. Extract and run:
   ```bash
   # Linux/Mac
   ./bin/neo4j start
   
   # Windows
   bin\neo4j.bat start
   ```
3. Set password:
   ```bash
   # Navigate to http://localhost:7474
   # Login with neo4j/neo4j
   # Set new password (e.g., 'password')
   ```

### Initialize the Database

After setting up Neo4j, initialize the database:

```bash
# Test connection
python test_neo4j.py

# Initialize database with constraints and indexes
python init_neo4j.py
```

### Configuration

Update `config/config.yaml` with your Neo4j credentials:

```yaml
graph_db:
  provider: "neo4j"
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "your_password_here"
  database: "chart_insights"
  require_connection: true
```

## Usage

### Command Line Interface

Analyze data from a CSV file:
```
python main.py --data data/samples/sales_data.csv --chart-type line
```

Analyze a chart image:
```
python main.py --image path/to/chart.png
```

Generate specific insight types:
```
python main.py --data data/samples/sales_data.csv --chart-type line --insight-types trend anomaly
```

Save results to a file:
```
python main.py --data data/samples/sales_data.csv --chart-type bar --output results/insights.json
```

### Streamlit UI

Launch the interactive UI:
```
python main.py --ui
```

Or directly using Streamlit:
```
streamlit run src/ui/streamlit_app.py
```

## Configuration

Configuration is stored in `config/config.yaml`. Key settings include:

- LLM provider and model
- Neo4j connection details
- Chart analysis parameters
- GraphRAG settings
- Insight generation preferences

## Modules

- **Chart Analysis**: Extracts data from chart images or processes raw data
- **Knowledge Graph**: Builds a graph representation of chart data
- **GraphRAG**: Implements graph-based retrieval augmented generation
- **Insights Generation**: Generates insights from knowledge graphs
- **UI**: Provides interactive user interfaces

## Insight Types

The system generates four types of insights:

1. **Trend Insights**: Patterns of change over time
2. **Comparison Insights**: Relationships between different data points
3. **Anomaly Insights**: Unusual or unexpected data points
4. **Correlation Insights**: Relationships between different variables

## Development

For development setup, testing, and contributing guidelines, please see:

- [Environment Setup](docs/ENVIRONMENT_SETUP.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Roadmap](docs/ROADMAP.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
