#!/bin/bash
# Setup script for Chart Insights System
# This script creates a virtual environment outside the project directory
# and installs the required dependencies

echo "=== Chart Insights System Setup ==="

# Create directory for virtual environments if it doesn't exist
if [ ! -d "$HOME/venvs" ]; then
    echo "Creating virtual environments directory..."
    mkdir -p "$HOME/venvs"
fi

# Create virtual environment for Chart Insights
echo "Creating virtual environment..."
python3 -m venv "$HOME/venvs/chart_insights_venv"

# Activate virtual environment
echo "Activating virtual environment..."
source "$HOME/venvs/chart_insights_venv/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov black flake8 mypy

# Install the package in development mode
echo "Installing in development mode..."
pip install -e .

echo "=== Setup complete ==="
echo
echo "To activate the environment, run:"
echo "source $HOME/venvs/chart_insights_venv/bin/activate"
echo
echo "To test the Neo4j connection, run:"
echo "python test_neo4j.py"
echo
echo "To initialize the Neo4j database, run:"
echo "python init_neo4j.py"
echo
echo "To start the UI, run:"
echo "python main.py --ui"
echo
