@echo off
cd /d "D:\Desktop\F5 Baseball Forecaster"

echo ========================================
echo  ClutchSheets F5 -- STEP 2: Push Forecasts
echo  %date% %time%
echo ========================================
echo.
echo  Run this AFTER dropping today's daily_forecast.csv
echo  in the folder from your model run.
echo  fetch_odds.py now archives the 6h snapshot into SQLite
echo  before writing daily_odds.json.
echo ========================================
echo.

echo [1/3] Fetching live odds from The Odds API...
python fetch_odds.py
if errorlevel 1 (
    echo [WARNING] fetch_odds.py failed -- pushing without odds.
    echo           Check your ODDS_API_KEY in fetch_odds.py
)

echo.
echo [2/3] Committing forecasts + odds to GitHub...
git add daily_forecast.csv daily_odds.json mlb_f5_forecasts.db
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "forecasts + odds %date%"
    git push
    echo  Git: pushed -- Railway redeploying.
) else (
    echo  Git: no changes to commit.
)

echo.
echo ========================================
echo  Done! %time%
echo ========================================
pause
