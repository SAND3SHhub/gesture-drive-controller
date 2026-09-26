@echo off

cd /d "%~dp0"

echo ========================================
echo       GESTURE DRIVE CONTROLLER
echo ========================================
echo.
echo Starting controller...
echo.

call venv\Scripts\activate.bat

python main.py

echo.
echo Controller stopped.
pause