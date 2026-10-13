@echo off
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=python
:loop
%PY% -m recorder --config config.toml
echo Recorder exited with code %errorlevel%. Restarting in 10 seconds. Close this window to stop.
timeout /t 10 /nobreak >nul
goto loop
