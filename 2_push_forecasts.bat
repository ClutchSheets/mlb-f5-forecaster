\
@echo off
cd /d "D:\Desktop\F5 Baseball Forecaster"

echo === Fetching forecast odds + archiving 6h snapshot ===
python fetch_odds.py
if errorlevel 1 goto :err

echo === Git add / commit / push forecast files ===
git add daily_forecast.csv daily_odds.json mlb_f5_forecasts.db app.py fetch_odds.py grade_edge_clv.py 1_score_yesterday.bat 2_push_forecasts.bat
git commit -m "forecast push"
git push

echo Done.
goto :eof

:err
echo ERROR: push pipeline failed.
pause
