@echo off
REM bootstrap.bat — full setup + run on a fresh Windows machine
REM Usage: bootstrap.bat

echo.
echo   RAG Retrieval Platform — Bootstrap
echo   ------------------------------------

REM 1. Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo   ✗  Python not found. Install Python 3.11+ from python.org
    exit /b 1
)
echo   ✓  Python found

REM 2. Create venv if missing
if not exist ".venv" (
    echo   →  Creating .venv...
    python -m venv .venv
)

REM 3. Activate
call .venv\Scripts\activate.bat
echo   ✓  venv activated

REM 4. Install dependencies
echo   →  Installing dependencies (first run: ~2 min)...
pip install -e ".[dev]" --quiet
echo   ✓  Dependencies installed

REM 5. Run the full pipeline
echo.
python run.py %*
