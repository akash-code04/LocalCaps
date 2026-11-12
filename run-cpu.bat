@echo off
setlocal
cd /d "%~dp0"
if not exist venv (
    echo Run run.bat once first to create the environment.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
set "WHISPER_DEVICE=cpu"
python app.py
pause
endlocal
