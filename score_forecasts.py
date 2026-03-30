# score_forecasts.py
# Reads daily_forecast.csv (Sheets export), matches MLB games by date + teams,
# pulls score through 5 innings, computes MAE/Brier/LogLoss, and appends to SQLite.
# Includes away_sp/home_sp in the stored snapshot.

import pandas as pd
import requests
import sqlite3
import math
from datetime import datetime, timezone

FORECAST_FILE = "daily_forecast.csv"
DB_FILE = "mlb_f5_forecasts.db"

TEAM_ID_MAP = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CHW": 145,   # both codes map to White Sox
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KCR": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147,
    "ATH": 133, "OAK": 133,   # both codes map to Athletics
    "PHI": 143, "PIT": 134, "SDP": 135, "SEA": 136,
    "SFG": 137, "STL": 138, "TBR": 139, "TEX": 140, "TOR": 141, "WSN": 120
}

def get_schedule(date_str: str) -> dict:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def get_game_feed(game_pk: int) -> dict:
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def flatten_games(schedule_json: dict) -> list:
    games = []
    for d in schedule_json.get("dates", []):
        games.extend(d.get("games", []))
    return games

def find_game_pk(games: list, away_id: int, home_id: int):
    matches = []
    for g in games:
        a_id = g["teams"]["away"]["team"]["id"]
        h_id = g["teams"]["home"]["team"]["id"]
        if a_id == away_id and h_id == home_id:
            matches.append(g["gamePk"])
    return matches[0] if matches else None

def compute_runs_thru5(game_feed: dict):
    innings = game_feed.get("liveData", {}).get("linescore", {}).get("innings", [])
    if not innings or len(innings) < 5:
        return None, None, 0
    away = sum(inn.get("away", {}).get("runs", 0) for inn in innings[:5])
    home = sum(inn.get("home", {}).get("runs", 0) for inn in innings[:5])
    return away, home, 1

def brier(p: float, y: float) -> float:
    return (p - y) ** 2

def logloss(p: float, y: float) -> float:
    # supports y in {0,1,0.5}
    p = max(min(p, 1 - 1e-15), 1e-15)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))

def ensure_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_archive (
            date TEXT,
            away_team TEXT,
            away_sp TEXT,
            home_team TEXT,
            home_sp TEXT,
            lambda_away REAL,
            lambda_home REAL,
            total_proj REAL,
            winprob_home REAL,
            model_version TEXT,
            runs_away_thru5 INTEGER,
            runs_home_thru5 INTEGER,
            completed5 INTEGER,
            total_actual INTEGER,
            home_win_f5 REAL,
            mae_total REAL,
            brier REAL,
            logloss REAL,
            game_pk INTEGER,
            created_at TEXT
        )
        """
    )
    conn.commit()

def validate_row(row: pd.Series):
    required = ["date", "away_team", "home_team", "lambda_away", "lambda_home", "winprob_home"]
    for c in required:
        if c not in row or pd.isna(row[c]):
            raise ValueError(f"Missing required field: {c}")

    away = str(row["away_team"]).strip()
    home = str(row["home_team"]).strip()
    if away not in TEAM_ID_MAP:
        raise ValueError(f"Unknown away_team code: {away}")
    if home not in TEAM_ID_MAP:
        raise ValueError(f"Unknown home_team code: {home}")

    p = float(row["winprob_home"])
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"winprob_home must be in [0,1], got {p}")

def main():
    forecasts = pd.read_csv(FORECAST_FILE, encoding="latin-1")

    # normalize
    forecasts["date"] = forecasts["date"].astype(str).str.strip()
    forecasts["away_team"] = forecasts["away_team"].astype(str).str.strip()
    forecasts["home_team"] = forecasts["home_team"].astype(str).str.strip()

    # optional starter cols
    if "away_sp" not in forecasts.columns:
        forecasts["away_sp"] = ""
    if "home_sp" not in forecasts.columns:
        forecasts["home_sp"] = ""

    forecasts["away_sp"] = forecasts["away_sp"].astype(str).fillna("").str.strip()
    forecasts["home_sp"] = forecasts["home_sp"].astype(str).fillna("").str.strip()

    forecasts["lambda_away"] = pd.to_numeric(forecasts["lambda_away"], errors="coerce")
    forecasts["lambda_home"] = pd.to_numeric(forecasts["lambda_home"], errors="coerce")
    forecasts["winprob_home"] = pd.to_numeric(forecasts["winprob_home"], errors="coerce")
    forecasts["total_proj"] = forecasts["lambda_away"] + forecasts["lambda_home"]

    conn = sqlite3.connect(DB_FILE)
    ensure_table(conn)

    inserted = 0
    skipped_invalid = 0
    skipped_no_schedule = 0
    skipped_no_match = 0
    skipped_not_thru5 = 0

    for _, row in forecasts.iterrows():
        try:
            validate_row(row)
        except Exception as e:
            print(f"[SKIP INVALID ROW] {e}")
            skipped_invalid += 1
            continue

        date = row["date"]
        away_code = row["away_team"]
        home_code = row["home_team"]
        away_id = TEAM_ID_MAP[away_code]
        home_id = TEAM_ID_MAP[home_code]

        schedule = get_schedule(date)
        games = flatten_games(schedule)
        if not games:
            print(f"[NO SCHEDULE] {date} {away_code}@{home_code}")
            skipped_no_schedule += 1
            continue

        game_pk = find_game_pk(games, away_id, home_id)
        if not game_pk:
            print(f"[NO MATCH] {date} {away_code}@{home_code}")
            skipped_no_match += 1
            continue

        feed = get_game_feed(game_pk)
        away_runs, home_runs, completed5 = compute_runs_thru5(feed)
        if not completed5:
            print(f"[NOT THRU 5] {date} {away_code}@{home_code} gamePk={game_pk}")
            skipped_not_thru5 += 1
            continue

        total_actual = int(away_runs + home_runs)

        if home_runs > away_runs:
            y = 1.0
        elif home_runs < away_runs:
            y = 0.0
        else:
            y = 0.5

        p = float(row["winprob_home"])
        mae_total = abs(total_actual - float(row["total_proj"]))
        b = brier(p, y)
        ll = logloss(p, y)
        created_at = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            INSERT INTO forecast_archive (
                date, away_team, away_sp, home_team, home_sp,
                lambda_away, lambda_home, total_proj, winprob_home, model_version,
                runs_away_thru5, runs_home_thru5, completed5, total_actual, home_win_f5,
                mae_total, brier, logloss, game_pk, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                away_code,
                str(row.get("away_sp", "")),
                home_code,
                str(row.get("home_sp", "")),
                float(row["lambda_away"]),
                float(row["lambda_home"]),
                float(row["total_proj"]),
                p,
                str(row.get("model_version", "v1.0")),
                int(away_runs),
                int(home_runs),
                int(completed5),
                total_actual,
                y,
                float(mae_total),
                float(b),
                float(ll),
                int(game_pk),
                created_at,
            )
        )

        inserted += 1
        print(f"[OK] {date} {away_code}@{home_code} thru5 {away_runs}-{home_runs} gamePk={game_pk}")

    conn.commit()
    conn.close()

    print("\n--- SUMMARY ---")
    print(f"Inserted: {inserted}")
    print(f"Skipped (invalid): {skipped_invalid}")
    print(f"Skipped (no schedule): {skipped_no_schedule}")
    print(f"Skipped (no match): {skipped_no_match}")
    print(f"Skipped (not thru 5): {skipped_not_thru5}")

if __name__ == "__main__":
    main()
    # Auto-push scores to GitHub → triggers Railway redeploy
    from push_scores import push_scores
    push_scores()
