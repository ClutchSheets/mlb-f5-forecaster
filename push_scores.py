import subprocess
from datetime import date
from pathlib import Path
GIT_REPO = Path(r"D:\Desktop\F5 Baseball Forecaster")
def push_scores(score_date: str = None) -> None:
    if score_date is None:
        score_date = date.today().strftime("%Y-%m-%d")
    subprocess.run(["git","-C",str(GIT_REPO),"add","app.py","build_edge_clv_input.py","grade_edge_clv.py","inspect_db.py","reset_edge_clv_tables.py","push_scores.py","mlb_f5_forecasts.db"], check=True)
    result = subprocess.run(["git","-C",str(GIT_REPO),"diff","--cached","--quiet"])
    if result.returncode == 0:
        print("No staged changes.")
        return
    subprocess.run(["git","-C",str(GIT_REPO),"commit","-m",f"deploy {score_date}"], check=True)
    subprocess.run(["git","-C",str(GIT_REPO),"push"], check=True)
    print("Pushed app + DB.")
if __name__ == "__main__":
    push_scores()
