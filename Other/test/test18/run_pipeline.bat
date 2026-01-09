@echo off
echo Test18 - GeoJSON Area Processing Pipeline
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or later and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking Python dependencies...
python -c "import ee; import numpy; import matplotlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required Python packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error: Failed to install required packages
        pause
        exit /b 1
    )
)

REM Run the pipeline
echo Starting pipeline processing...
echo.
python main.py

echo.
echo Pipeline processing completed.
pause