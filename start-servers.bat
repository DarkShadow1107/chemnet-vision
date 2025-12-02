@echo off
echo ========================================
echo    ChemNet-Vision Server Launcher
echo ========================================
echo.

:: Get the directory where this script is located
cd /d "%~dp0"

echo Select an option:
echo   1. Start both servers (Backend + Frontend)
echo   2. Train the AI model
echo   3. Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" goto start_servers
if "%choice%"=="2" goto train_model
if "%choice%"=="3" goto end
goto start_servers

:start_servers
echo.
echo [1/2] Starting Flask Backend (port 5000)...
start "ChemNet Backend" cmd /k ".venv\Scripts\python.exe backend\app.py"

:: Wait a moment for backend to initialize
timeout /t 3 /nobreak >nul

echo [2/2] Starting Next.js Frontend (port 3000)...
start "ChemNet Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo    Both servers are starting!
echo ========================================
echo.
echo    Backend:  http://localhost:5000
echo    Frontend: http://localhost:3000
echo.
echo    Close the server windows to stop them.
echo ========================================
goto end

:train_model
echo.
echo ========================================
echo    Starting Model Training
echo ========================================
echo.
echo This will train the ChemNet-Vision neural network.
echo Training may take a long time depending on your hardware.
echo.
pause

.venv\Scripts\python.exe ai_model\train_model.py

echo.
echo ========================================
echo    Training Complete!
echo ========================================
echo.
goto end

:end
pause
