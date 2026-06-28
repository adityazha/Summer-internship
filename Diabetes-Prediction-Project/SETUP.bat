@echo off
title Diabetes Prediction - Setup
color 0A

echo ============================================
echo   Diabetes Prediction Project - First Setup
echo ============================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed!
    echo.
    echo Download Python from: https://www.python.org/downloads/
    echo During install, check "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/2] Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)

if not exist "artifacts\diabetes_model.pkl" (
    echo.
    echo [2/2] Training model (first time only, ~2 min)...
    python train_model.py
) else (
    echo.
    echo [2/2] Model already exists. Skipping training.
)

echo.
echo ============================================
echo   Setup Complete! Now double-click RUN.bat
echo ============================================
pause
