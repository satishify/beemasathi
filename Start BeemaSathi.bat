@echo off
rem BeemaSathi launcher for Windows. Double-click to open the menu,
rem or run: Start BeemaSathi.bat start
setlocal
cd /d "%~dp0"

set "PY_CMD="
py -3 -c "import sys" >nul 2>nul && set "PY_CMD=py -3"
if not defined PY_CMD (
  python -c "import sys" >nul 2>nul && set "PY_CMD=python"
)
if not defined PY_CMD (
  python3 -c "import sys" >nul 2>nul && set "PY_CMD=python3"
)
if not defined PY_CMD (
  echo Python 3.11 or newer is required.
  echo Install it from https://www.python.org/downloads/ and tick the box
  echo "Add python.exe to PATH" during setup.
  pause
  exit /b 1
)

%PY_CMD% "%~dp0launcher.py" %*
set "RC=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %RC%
