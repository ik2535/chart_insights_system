# Development Environment Setup

This document provides instructions for setting up a development environment for the Chart Insights System.

## Virtual Environment

It's recommended to create your virtual environment **outside** the project directory to avoid accidentally committing it to the repository.

### Option 1: Create a virtual environment in a dedicated directory

```bash
# Create a dedicated directory for virtual environments if you don't have one
mkdir -p ~/venvs

# Create a virtual environment for this project
python -m venv ~/venvs/chart_insights_venv

# Activate the virtual environment
# On Windows
~/venvs/chart_insights_venv/Scripts/activate

# On macOS/Linux
source ~/venvs/chart_insights_venv/bin/activate
```

### Option 2: Create a virtual environment in the project directory (not recommended for Git)

If you must create the virtual environment inside the project directory, make sure it's named `venv` so that it's ignored by Git:

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

## Installing Dependencies

Once your virtual environment is activated, install the project dependencies:

```bash
# Install dependencies
pip install -r requirements.txt

# For development, you might want to install the project in development mode
pip install -e .
```

## Setting Up Neo4j

The Chart Insights System requires Neo4j for storing knowledge graphs. Follow these steps to set up Neo4j:

1. See the instructions in the main README.md for Neo4j setup
2. Run the initialization script:
   ```bash
   python init_neo4j.py
   ```

## Running Tests

To run the tests, make sure you have the development dependencies installed:

```bash
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## Starting the UI

To start the Streamlit UI:

```bash
# Using the convenience script
# On Windows
run_ui.bat

# On macOS/Linux
./run_ui.sh

# Or directly
python main.py --ui
```

## Development Tools

It's recommended to use the following tools for development:

1. **Black**: For code formatting
   ```bash
   pip install black
   black .
   ```

2. **Flake8**: For linting
   ```bash
   pip install flake8
   flake8 .
   ```

3. **MyPy**: For type checking
   ```bash
   pip install mypy
   mypy src
   ```

4. **Pre-commit**: For automating checks before commits
   ```bash
   pip install pre-commit
   pre-commit install
   ```
