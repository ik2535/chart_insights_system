#!/bin/bash
echo "Starting Chart Insights System UI..."

# Check if the venv directory exists in the project
if [ -f "venv/bin/activate" ]; then
    # Use the local virtual environment
    echo "Using local virtual environment..."
    source venv/bin/activate
else
    # Check if the external virtual environment exists
    if [ -f "$HOME/venvs/chart_insights_venv/bin/activate" ]; then
        echo "Using external virtual environment..."
        source "$HOME/venvs/chart_insights_venv/bin/activate"
    else
        echo "No virtual environment found. Please run setup.sh first."
        exit 1
    fi
fi

# Start the UI
python main.py --ui
