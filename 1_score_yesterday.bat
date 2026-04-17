@echo off
cd /d "D:\Desktop\F5 Baseball Forecaster"

echo ========================================
echo  ClutchSheets F5 -- STEP 1: Score Games
echo  %date% %time%
echo ========================================
echo.
echo  Scores yesterday's games from daily_forecast.csv
echo  grades them against Pinnacle F5 close
echo  builds Edge CLV summaries
echo  then pushes mlb_f5_forecasts.db to GitHub.
echo  Run this BEFORE you overwrite daily_forecast.csv
echo  with today's model output.
echo ========================================
echo.

echo [1/4] Scoring forecasts...
python score_forecasts.py
if errorlevel 1 (
    echo [ERROR] score_forecasts.py failed. Check output above.
    pause
    exit /b 1
)

echo.
echo [2/4] Backfilling / grading Pinnacle closing odds...
python grade_closing.py
if errorlevel 1 (
    echo [ERROR] grade_closing.py failed. Check output above.
    pause
    exit /b 1
)

echo.
echo [3/4] Building Edge CLV summaries...
python grade_edge_clv.py
if errorlevel 1 (
    echo [ERROR] grade_edge_clv.py failed. Check output above.
    pause
    exit /b 1
)

echo.
echo [4/4] Pushing DB to GitHub...
python push_scores.py
if errorlevel 1 (
    echo [ERROR] push_scores.py failed. Push manually.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Done! %time%
echo  Railway is redeploying with new scores.
echo ========================================
pause
