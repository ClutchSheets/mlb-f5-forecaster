import subprocess
from datetime import date
from pathlib import Path

GIT_REPO = Path(r"D:\Desktop\F5 Baseball Forecaster")

def run(cmd, **kwargs):
    return subprocess.run(["git", "-C", str(GIT_REPO)] + cmd, **kwargs)

def push_scores(score_date: str = None) -> None:
    if score_date is None:
        score_date = date.today().strftime("%Y-%m-%d")

    # Force git to see DB as changed by removing from index and re-adding
    run(["rm", "--cached", "mlb_f5_forecasts.db"], check=False)
    run(["add", "-f", "mlb_f5_forecasts.db"], check=True)

    # Add other tracked files
    other_files = [
        "app.py", "build_edge_clv_input.py", "grade_edge_clv.py",
        "push_scores.py", "fetch_odds.py", "score_forecasts.py",
        "grade_closing.py", "edge_clv_input.csv",
    ]
    run(["add"] + other_files, check=False)  # check=False so missing files don't abort

    result = run(["diff", "--cached", "--quiet"])
    if result.returncode == 0:
        print("No staged changes.")
        return

    run(["commit", "-m", f"deploy {score_date}"], check=True)

    # Pull remote changes first to avoid rejection
    run(["pull", "--rebase", "--autostash"], check=True)
    run(["push"], check=True)
    print("Pushed successfully.")

if __name__ == "__main__":
    push_scores()
