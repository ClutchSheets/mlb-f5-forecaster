"""
push_scores.py
==============
Call this at the end of your nightly scoring script to push
mlb_f5_forecasts.db to GitHub, triggering a Railway redeploy.

Usage (standalone):
    python push_scores.py

Or import and call from your scoring script:
    from push_scores import push_scores
    push_scores("2026-03-28")
"""

import subprocess
from datetime import date
from pathlib import Path

GIT_REPO = Path(r"D:\Desktop\F5 Baseball Forecaster")
DB_FILE  = GIT_REPO / "mlb_f5_forecasts.db"


def push_scores(score_date: str = None) -> None:
    if score_date is None:
        score_date = date.today().strftime("%Y-%m-%d")

    print(f"Pushing scores to GitHub ({score_date})...")

    if not DB_FILE.exists():
        print(f"  WARNING: {DB_FILE} not found — skipping push")
        return

    try:
        subprocess.run(["git", "-C", str(GIT_REPO), "add", str(DB_FILE)],
                       check=True, capture_output=True)

        result = subprocess.run(
            ["git", "-C", str(GIT_REPO), "diff", "--cached", "--quiet"],
            capture_output=True)
        if result.returncode == 0:
            print("  Git: no scoring changes to commit")
            return

        subprocess.run(
            ["git", "-C", str(GIT_REPO), "commit", "-m", f"scores {score_date}"],
            check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(GIT_REPO), "push"],
            check=True, capture_output=True)
        print(f"  Git: pushed scores {score_date} → Railway redeploying")

    except subprocess.CalledProcessError as e:
        print(f"  Git WARNING: push failed — {e}")
        print("  Run manually: git add mlb_f5_forecasts.db && git commit -m 'scores' && git push")


if __name__ == "__main__":
    push_scores()
