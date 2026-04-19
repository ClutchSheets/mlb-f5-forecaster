\
@echo off
cd /d "D:\Desktop\F5 Baseball Forecaster"

echo === Scoring yesterday's forecasts ===
python score_forecasts.py

echo === Grading devigged closing line ===
python grade_closing.py

echo === Building edge/CLV input ===
python build_edge_clv_input.py

echo === Grading edge CLV thresholds ===
python grade_edge_clv.py

echo === Pushing updated DB ===
python push_scores.py

echo Done.
goto :eof

:err
echo ERROR: pipeline failed.
pause
