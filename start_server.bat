@echo off
title BoneSense AI - Diagnostic Station
cd /d "%~dp0"
echo ========================================================
echo Starting BoneSense AI PACS Diagnostic Station...
echo Open your browser at: http://localhost:5000/
echo ========================================================
python src/app.py
pause
