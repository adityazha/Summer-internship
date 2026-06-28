@echo off
title Diabetes Prediction App
color 0B

echo Starting Diabetes Prediction App...
echo Browser will open at http://localhost:8501
echo.
echo To stop the app, close this window or press Ctrl+C
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Run SETUP.bat first.
    pause
    exit /b 1
)

if not exist "artifacts\diabetes_model.pkl" (
    echo Model not found. Running setup first...
    call SETUP.bat
)

streamlit run app.py
