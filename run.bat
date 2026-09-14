@echo off
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
python main.py
if errorlevel 1 (
    echo.
    echo 실행 중 오류가 발생했습니다.
    pause
)
