@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe scripts\bootstrap.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe app.py
exit /b %errorlevel%
:failed
echo Startup failed. Install Python 3.12 or use Docker.
pause
exit /b 1
