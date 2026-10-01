@echo off
title CityWatch AI Local Server
echo ==========================================
echo Starting CityWatch AI Server...
echo ==========================================

:: Activate virtual environment
call venv\Scripts\activate

:: Automatically open browser after 2 seconds
start "" http://127.0.0.1:8000

:: Start FastAPI backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

pause