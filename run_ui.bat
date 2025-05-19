@echo off
echo Starting Chart Insights System UI...

REM Check if the venv directory exists in the project
if exist "venv\Scripts\activate.bat" (
    REM Use the local virtual environment
    echo Using local virtual environment...
    call venv\Scripts\activate.bat
) else (
    REM Check if the external virtual environment exists
    if exist "%USERPROFILE%\venvs\chart_insights_venv\Scripts\activate.bat" (
        echo Using external virtual environment...
        call "%USERPROFILE%\venvs\chart_insights_venv\Scripts\activate.bat"
    ) else (
        echo No virtual environment found. Please run setup.bat first.
        exit /b 1
    )
)

REM Start the UI
python main.py --ui
