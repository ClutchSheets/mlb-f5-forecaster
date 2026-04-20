@echo off
cd /d "D:\Desktop\F5 Baseball Forecaster"

echo ========================================
echo  ClutchSheets F5 Daily Score + Push
echo  %date% %time%
echo ========================================

echo.
echo [1/5] Scoring yesterday's games...
python score_forecasts.py
if errorlevel 1 (
    echo [ERROR] score_forecasts.py failed. Stopping.
    pause
    exit /b 1
)

echo.
echo [2/5] Fetching Pinnacle closing lines...
python grade_closing.py
if errorlevel 1 (
    echo [ERROR] grade_closing.py failed. Stopping.
    pause
    exit /b 1
)

echo.
echo [3/5] Building edge/CLV input...
python build_edge_clv_input.py
if errorlevel 1 (
    echo [ERROR] build_edge_clv_input.py failed. Stopping.
    pause
    exit /b 1
)

echo.
echo [4/5] Loading edge bets into DB...
python grade_edge_clv.py
if errorlevel 1 (
    echo [ERROR] grade_edge_clv.py failed. Stopping.
    pause
    exit /b 1
)

echo.
echo [5/5] Pushing to GitHub...
python push_scores.py
if errorlevel 1 (
    echo [ERROR] push_scores.py failed. Stopping.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Done! %time%
echo  Performance tab will update in ~60s
echo ========================================
pause
