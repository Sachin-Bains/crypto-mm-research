@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo This folder has no .venv, so it is not your project folder.
  echo Run run_recorder.bat from %USERPROFILE%\Projects\crypto-mm-research instead.
  echo.
  pause
  exit /b 1
)
:loop
.venv\Scripts\python.exe -m recorder --config config.toml
echo Recorder exited with code %errorlevel%. Restarting in 10 seconds. Close this window to stop.
timeout /t 10 /nobreak >nul
goto loop
