@echo off
echo GEE Sensor Data Downloader
echo ==========================

REM Check if virtual environment exists
if not exist "..\agro\.venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv ..\agro\.venv
)

echo Activating virtual environment...
call ..\agro\.venv\Scripts\activate.bat

echo Installing required packages...
pip install -r requirements.txt

echo Running sensor data downloader...
python main.py

echo Pipeline execution completed.
pause