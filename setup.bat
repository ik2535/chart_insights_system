@echo off
REM Setup script for Chart Insights System
REM This script creates a virtual environment outside the project directory
REM and installs the required dependencies

echo === Chart Insights System Setup ===

REM Create directory for virtual environments if it doesn't exist
if not exist "%USERPROFILE%\venvs" (
    echo Creating virtual environments directory...
    mkdir "%USERPROFILE%\venvs"
)

REM Create virtual environment for Chart Insights
echo Creating virtual environment...
python -m venv "%USERPROFILE%\venvs\chart_insights_venv"

REM Activate virtual environment
echo Activating virtual environment...
call "%USERPROFILE%\venvs\chart_insights_venv\Scripts\activate.bat"

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Install development dependencies
echo Installing development dependencies...
pip install pytest pytest-cov black flake8 mypy

REM Install the package in development mode
echo Installing in development mode...
pip install -e .

echo === Setup complete ===
echo.
echo To activate the environment, run:
echo call "%USERPROFILE%\venvs\chart_insights_venv\Scripts\activate.bat"
echo.
echo To test the Neo4j connection, run:
echo python test_neo4j.py
echo.
echo To initialize the Neo4j database, run:
echo python init_neo4j.py
echo.
echo To start the UI, run:
echo python main.py --ui
echo.
