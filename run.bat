@echo off
cd /d "%~dp0"
echo ====================================================================
echo Starting PathSnap GPS Map Matching Server (Python 3.14)
echo ====================================================================
"%LOCALAPPDATA%\Programs\Python\Python314\python.exe" -u run_server.py
pause
