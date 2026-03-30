@echo off
:: Run as Administrator — registers both scheduled tasks

:: Nightly at 12:30 AM
schtasks /create ^
  /tn "ClutchSheets F5 Nightly" ^
  /tr "python D:\Desktop\f5_tracker_v2.py --nightly" ^
  /sc daily /st 00:30 ^
  /ru "%USERNAME%" /f

:: Morning at 10:00 AM
schtasks /create ^
  /tn "ClutchSheets F5 Morning" ^
  /tr "python D:\Desktop\f5_tracker_v2.py --morning" ^
  /sc daily /st 10:00 ^
  /ru "%USERNAME%" /f

echo.
echo Tasks created:
echo   ClutchSheets F5 Nightly  — 12:30 AM daily
echo   ClutchSheets F5 Morning  — 10:00 AM daily
echo.
echo Check Task Scheduler to confirm both tasks are registered.
pause
