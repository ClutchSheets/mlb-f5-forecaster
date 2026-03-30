"""
push_from_slate.py — ClutchSheets F5 Slate → daily_forecast.csv
================================================================
Reads the Forecast sheet of f5_review_YYYY-MM-DD.xlsx
(which pulls its values from the Slate sheet you've edited),
writes daily_forecast.csv to the repo root, and git pushes.

Usage:
    python push_from_slate.py                    # today
    python push_from_slate.py --date 2026-04-01
"""

import argparse
import subprocess
from datetime import date, datetime
from pathlib import Path

import openpyxl
import pandas as pd

MODEL_DIR    = Path(r"D:\Desktop\F5 Baseball Forecaster\Model")
FORECAST_DIR = Path(r"D:\Desktop\F5 Baseball Forecaster")


def git_push(message, files):
    try:
        for f in files:
            subprocess.run(["git", "-C", str(FORECAST_DIR), "add", str(f)],
                           check=True, capture_output=True)
        r = subprocess.run(["git", "-C", str(FORECAST_DIR), "diff", "--cached", "--quiet"],
                           capture_output=True)
        if r.returncode == 0:
            print("  Git: nothing new to commit")
            return
        subprocess.run(["git", "-C", str(FORECAST_DIR), "commit", "-m", message],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(FORECAST_DIR), "push"],
                       check=True, capture_output=True)
        print(f"  Git pushed: {message}")
    except subprocess.CalledProcessError as e:
        print(f"  Git WARNING: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--no-push", action="store_true", help="Skip git push")
    args = parser.parse_args()

    gd       = (date.today() if not args.date
                else datetime.strptime(args.date, "%Y-%m-%d").date())
    date_str = gd.strftime("%Y-%m-%d")

    # Always read from static workbook
    review_file = MODEL_DIR / "F5_Daily.xlsx"
    if not review_file.exists():
        print(f"ERROR: F5_Daily.xlsx not found in {MODEL_DIR}")
        print("Run build_f5_daily.py first")
        return

    # Read the Forecast sheet with data_only=True so we get computed values
    wb = openpyxl.load_workbook(review_file, data_only=True)

    if "Forecast" not in wb.sheetnames:
        print("ERROR: 'Forecast' sheet not found in workbook")
        return

    ws = wb["Forecast"]

    rows = []
    for row in ws.iter_rows(min_row=3, values_only=True):
        if not row[0] and not row[3]:   # stop when both date and home_team blank
            break
        date_val, away_team, away_sp, home_team, home_sp, \
            lambda_away, lambda_home, total_proj, winprob_home, model_ver = row[:10]

        # Skip if home team is blank
        if not home_team:
            continue

        rows.append({
            "date":          str(date_val or date_str).strip(),
            "away_team":     str(away_team or "").strip(),
            "away_sp":       str(away_sp  or "").strip(),
            "home_team":     str(home_team or "").strip(),
            "home_sp":       str(home_sp  or "").strip(),
            "lambda_away":   round(float(lambda_away  or 0), 3),
            "lambda_home":   round(float(lambda_home  or 0), 3),
            "total_proj":    round(float(total_proj   or 0), 3),
            "winprob_home":  round(float(winprob_home or 0.5), 3),
            "model_version": str(model_ver or "v3.0"),
        })

    if not rows:
        print("ERROR: no data rows found in Forecast sheet")
        return

    df = pd.DataFrame(rows)

    # Write daily_forecast.csv (repo root — picked up by Railway/app.py)
    daily_csv = FORECAST_DIR / "daily_forecast.csv"
    df.to_csv(daily_csv, index=False)
    print(f"Wrote {len(df)} games → {daily_csv}")

    # Also write dated copy
    dated_csv = FORECAST_DIR / f"f5_forecast_{date_str}.csv"
    df.to_csv(dated_csv, index=False)
    print(f"Wrote dated copy → {dated_csv}")

    # Print summary
    print(f"\n{'GAME':<36} {'λ Away':>7} {'λ Home':>7} {'H Win%':>7}")
    print("-" * 62)
    for _, r in df.iterrows():
        g = f"{r['away_team']} @ {r['home_team']}"
        print(f"{g:<36} {r['lambda_away']:>7.3f} {r['lambda_home']:>7.3f} "
              f"{r['winprob_home']*100:>6.1f}%")

    if not args.no_push:
        print("\nPushing to GitHub...")
        git_push(f"forecast {date_str} (from slate)", [daily_csv])

    print("\nDone.")


if __name__ == "__main__":
    main()
