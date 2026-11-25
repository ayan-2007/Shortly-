@echo off
setlocal EnableExtensions
title Shortly Studio v1.2
cd /d "%~dp0"

if not exist "gui.py" (
    if exist "ShortlyStudio.exe" (
        start "" "ShortlyStudio.exe" %*
        exit /b 0
    )
    echo.
    echo [Shortly] Could not find gui.py in this folder.
    echo If you opened this from inside a .zip file, please EXTRACT ALL files
    echo to a normal folder first, then double-click "Start Shortly.bat".
    echo.
    pause
    exit /b 1
)

REM High-throughput UTF-8 & quiet C++ logging environment
set "PYTHONUTF8=1"
set "OPENCV_LOG_LEVEL=ERROR"
if defined NUMBER_OF_PROCESSORS set "OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%"

REM Prepend bundled bin\ and WinGet Links to PATH for instant FFmpeg/FFprobe lookup
if exist "%~dp0bin" set "PATH=%~dp0bin;%PATH%"
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"

if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
    )
)

set "READY_STAMP=.venv\.shortly_ready"
set "STAMP_ID=%COMPUTERNAME%::%~dp0::READY_V2"
set "SAVED_STAMP="

REM Instant zero-subprocess fast-path check when this PC + folder is already verified
if exist ".venv\Scripts\python.exe" (
    if exist "%READY_STAMP%" (
        set /p SAVED_STAMP=<"%READY_STAMP%"
    )
)
if /i "%SAVED_STAMP%"=="%STAMP_ID%" goto :fast_launch

:verify_env
if not exist ".venv\Scripts\python.exe" goto :create_venv

".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 goto :check_deps

echo [Shortly] Existing .venv is from another PC or broken. Rebuilding for this machine...
rmdir /s /q ".venv" 2>nul

:create_venv
echo [Shortly] First-time setup: creating Python virtual environment...
set "PY_CMD="

call :check_py "py -3.12"
if not defined PY_CMD call :check_py "py -3.11"
if not defined PY_CMD call :check_py "py -3.10"
if not defined PY_CMD call :check_py "py -3"
if not defined PY_CMD call :check_py "python"
if not defined PY_CMD call :check_py "python3"

if defined PY_CMD goto :run_venv

where uv >nul 2>&1
if not errorlevel 1 goto :run_uv_venv

echo.
echo [Shortly] Python 3.10-3.12 was not found on this PC.
echo Please install Python 3.12 from https://www.python.org/downloads/
echo Make sure to check "Add python.exe to PATH" during installation,
echo then double-click "Start Shortly.bat" again.
echo.
pause
exit /b 1

:run_venv
echo [Shortly] Using Python: %PY_CMD%
%PY_CMD% -m venv .venv
if errorlevel 1 (
    echo [Shortly] Failed to create virtual environment.
    pause
    exit /b 1
)
goto :check_deps

:run_uv_venv
echo [Shortly] Using uv to create Python 3.12 virtual environment...
uv venv --seed --python 3.12 .venv
if errorlevel 1 (
    echo [Shortly] Failed to create virtual environment with uv.
    pause
    exit /b 1
)

:check_deps
".venv\Scripts\python.exe" -c "import importlib.util; mods=('yaml','pydantic','faster_whisper','cv2','yt_dlp','PIL','rich','numpy'); raise SystemExit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)" >nul 2>&1
if not errorlevel 1 goto :mark_ready

echo [Shortly] Installing dependencies ^(this only runs once and may take a few minutes^)...
where uv >nul 2>&1
if not errorlevel 1 (
    uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
    if not errorlevel 1 goto :mark_ready
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [Shortly] Dependency installation failed. Check the output above.
    echo Tip: Python 3.11 or 3.12 is recommended ^(Python 3.13+ may lack wheels for some packages^).
    pause
    exit /b 1
)

:mark_ready
>"%READY_STAMP%" echo %STAMP_ID%
goto :launch

:fast_launch
echo [Shortly] Starting Shortly Studio v1.2 ^(http://127.0.0.1:8780^)...
".venv\Scripts\python.exe" -O gui.py %*
if errorlevel 1 (
    del /f /q "%READY_STAMP%" 2>nul
    echo [Shortly] Re-verifying Python environment after unexpected exit...
    goto :verify_env
)
exit /b 0

:launch
echo [Shortly] Starting Shortly Studio v1.2 ^(http://127.0.0.1:8780^)...
".venv\Scripts\python.exe" -O gui.py %*
if errorlevel 1 (
    del /f /q "%READY_STAMP%" 2>nul
    echo.
    echo [Shortly] Could not start. Check the error output above.
    pause
    exit /b 1
)
exit /b 0

:check_py
%~1 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY_CMD=%~1"
exit /b 0
