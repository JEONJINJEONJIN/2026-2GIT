@echo off
setlocal
chcp 65001 >nul

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [ERROR] Could not find .venv Python:
  echo %PYTHON%
  echo.
  echo Create/restore the project virtual environment first.
  pause
  exit /b 1
)

if not exist "%ROOT%dev_harness.py" (
  echo [ERROR] Could not find dev_harness.py in:
  echo %ROOT%
  pause
  exit /b 1
)

echo Starting Slider Demo Dev Harness...
echo.
echo Browser should open automatically at:
echo http://localhost:8501
echo.
echo Keep this window open while using the demo.
echo Press Ctrl+C here to stop the server.
echo.

set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_SERVER_FILE_WATCHER_TYPE=none"
set "TRANSFORMERS_VERBOSITY=error"

"%PYTHON%" -m streamlit run "%ROOT%dev_harness.py" --server.headless=false --server.port=8501 --browser.gatherUsageStats=false --server.fileWatcherType=none

echo.
echo Streamlit stopped.
pause
