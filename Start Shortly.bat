@echo off
title Shortly Studio
cd /d "%~dp0"

if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [Shortly] First-time setup: creating Python virtual environment...
    where python >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [Shortly] Python was not found on PATH.
        echo Please install Python 3.11 or 3.12 from https://www.python.org/downloads/
        echo Make sure to check "Add python.exe to PATH" during installation.
        pause
        exit /b 1
    )
    python -m venv .venv
    if errorlevel 1 (
        echo [Shortly] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [Shortly] Installing dependencies (this only runs once)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [Shortly] Dependency installation failed. Check the output above.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" gui.py

if errorlevel 1 (
    echo.
    echo [Shortly] Could not start. Check Python / FFmpeg installation above.
    pause
)
