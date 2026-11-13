
@echo off
title Climate Digital Twin

cd /d "%~dp0"

echo Starting Climate Digital Twin...
echo.

if not exist "backend\.venv\Scripts\python.exe" (
    echo ERROR: Backend virtual environment not found!
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo ERROR: Frontend dependencies not installed!
    echo Run npm install inside the frontend folder first.
    pause
    exit /b 1
)

start "Climate Backend" cmd /k "cd /d "%~dp0" && backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload"

timeout /t 3 /nobreak >nul

start "Climate Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Both servers are starting.
echo Open http://localhost:5173
pause
