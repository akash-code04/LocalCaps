@echo off
setlocal
cd /d "%~dp0"
echo ========================================
echo             LocalCaps Setup
echo ========================================
echo.
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON=py -3.12"
    ) else (
        py -3.11 --version >nul 2>nul
        if not errorlevel 1 (set "PYTHON=py -3.11") else (set "PYTHON=python")
    )
) else (
    set "PYTHON=python"
)
if not exist venv (
    echo Creating Python virtual environment...
    %PYTHON% -m venv venv
    if errorlevel 1 goto failed
)
call venv\Scripts\activate.bat
if errorlevel 1 goto failed
python -m pip install --upgrade pip
if errorlevel 1 goto failed
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
set "WHISPER_DEVICE=auto"
set "LOCALCAPS_CUDA_LIB_DIR=D:\LocalCaps-Dependencies\cuBLAS.and.cuDNN_CUDA12_win_v3"
if exist "%LOCALCAPS_CUDA_LIB_DIR%\cublas64_12.dll" (
    set "PATH=%LOCALCAPS_CUDA_LIB_DIR%;%PATH%"
    echo CUDA runtime libraries found. GPU acceleration will be enabled when available.
) else (
    echo CUDA runtime bundle not found. LocalCaps will try GPU detection, then CPU fallback.
)
echo.
echo LocalCaps will use NVIDIA CUDA when available, otherwise CPU.
echo For GPU acceleration, see GPU_SETUP.md.
echo.
python app.py
if errorlevel 1 goto failed
goto end
:failed
echo.
echo Setup or startup failed. Read the error above and see README.md.
pause
:end
endlocal
