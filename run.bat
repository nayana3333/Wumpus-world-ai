@echo off
REM Wumpus World AI - one-click launcher for Windows
cd /d "%~dp0"

if not exist ".venv" (
  echo Creating virtual environment...
  python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --quiet --disable-pip-version-check -r requirements.txt

echo.
echo   Wumpus World AI  ->  http://127.0.0.1:5000
echo   Press Ctrl+C to stop.
echo.
python app.py
pause
