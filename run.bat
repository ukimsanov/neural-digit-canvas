@echo off
REM MNIST Classifier - Startup Script for Windows
REM This script starts both the FastAPI backend and Next.js frontend

echo Starting MNIST Classifier...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    echo Virtual environment found
    call venv\Scripts\activate
)

REM Check if node_modules exists in frontend
if not exist "frontend\node_modules" (
    echo Node modules not found. Installing...
    cd frontend
    npm install
    cd ..
) else (
    echo Node modules found
)

REM Start FastAPI backend
echo Starting FastAPI backend on http://localhost:8000
start /B cmd /c "venv\Scripts\activate && python api.py"

REM Wait for API to start
timeout /t 2 /nobreak >nul

REM Start Next.js frontend
echo Starting Next.js frontend on http://localhost:3000
cd frontend
start /B cmd /c "npm run dev"
cd ..

REM Wait for services to start
timeout /t 3 /nobreak >nul

echo.
echo MNIST Classifier is ready!
echo.
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press CTRL+C to stop all services
echo.

REM Keep the window open
pause