@echo off
echo Starting Python app...

:: Run the Python file
start "" python image.py

:: Wait a few seconds to ensure the server starts
timeout /t 3 /nobreak >nul

:: Open localhost in default web browser
start http://127.0.0.1:5050

exit