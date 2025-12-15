@echo off
echo ============================================
echo MedScan Backend - Gemini AI Setup
echo ============================================
echo.

REM Check if .env exists
if exist .env (
    echo [✓] .env file found
) else (
    echo [!] Creating .env file from template...
    copy .env.example .env
    echo.
    echo [!] IMPORTANT: Edit .env file and add your Gemini API key
    echo     Get your key from: https://makersuite.google.com/app/apikey
    echo.
    pause
)

echo.
echo Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [✗] Installation failed. Please check the errors above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo [✓] Setup Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Edit .env file and add your GEMINI_API_KEY
echo 2. Run: python app.py
echo.
echo For detailed instructions, see GEMINI_SETUP.md
echo.
pause
