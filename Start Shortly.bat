@echo off
title Shortly Studio
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" gui.py
) else (
    python gui.py
)

if errorlevel 1 (
    echo.
    echo [Shortly] Could not start. Check Python / .venv installation above.
    pause
)
