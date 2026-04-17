\
@echo off
cd /d "D:\Desktop\F5 Baseball Forecaster"

echo === Scoring yesterday's forecasts ===
python score_forecasts.py
if errorlevel 1 goto :err

echo === Grading devigged closing line ===
python grade_closing.py
if errorlevel 1 goto :err

echo === Grading edge CLV thresholds ===
python grade_edge_clv.py
if errorlevel 1 goto :err

echo === Pushing updated DB ===
python push_scores.py
if errorlevel 1 goto :err

echo Done.
goto :eof

:err
echo ERROR: pipeline failed.
pause
