@echo off
echo ===========================================
echo   DEEPFAKE THREAT DETECTOR INITIALIZATION
echo ===========================================
echo.

if not exist venv (
    echo [1/3] Creating Python Virtual Environment...
    python -m venv venv
    echo [2/3] Installing Dependencies...
    call venv\Scripts\activate.bat
    pip install -r backend\requirements.txt opencv-python-headless
) else (
    echo [1/3] Found existing environment.
    call venv\Scripts\activate.bat
)

echo.
echo [3/3] Launching Neural Network Backend...
start "Deepfake Backend" cmd /c "call venv\Scripts\activate.bat && uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo.
echo Waiting for AI models to load into memory...
timeout /t 8 /nobreak >nul

echo Opening Security Dashboard in your Web Browser...
start http://localhost:8000

echo.
echo Done! Please keep the black server terminal open while using the app.
pause
