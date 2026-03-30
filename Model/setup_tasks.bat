@echo off
:: ============================================================
:: ClutchSheets F5 — Task Scheduler Setup
:: Run as Administrator once. Creates two daily tasks.
:: ============================================================

set SCRIPT=D:\Desktop\F5 Baseball Forecaster\Model\f5_pipeline.py

:: Remove old tasks
schtasks /delete /tn "ClutchSheets F5 Nightly" /f >nul 2>&1
schtasks /delete /tn "ClutchSheets F5 Morning" /f >nul 2>&1
schtasks /delete /tn "ClutchSheets F5 Daily"   /f >nul 2>&1

:: Nightly 12:30 AM — rebuild xRA while you sleep
schtasks /create /tn "ClutchSheets F5 Nightly" /tr "python \"%SCRIPT%\" --no-excel" /sc daily /st 00:30 /ru "%USERNAME%" /f

:: Morning 9:00 AM — fresh slate with today's probable starters + Excel
schtasks /create /tn "ClutchSheets F5 Morning" /tr "python \"%SCRIPT%\"" /sc daily /st 09:00 /ru "%USERNAME%" /f

echo.
echo Tasks created:
echo   ClutchSheets F5 Nightly  12:30 AM  xRA rebuild + CSVs no Excel
echo   ClutchSheets F5 Morning  9:00 AM   full run xRA + slate + Excel
echo.
echo Script: %SCRIPT%
echo.
pause
