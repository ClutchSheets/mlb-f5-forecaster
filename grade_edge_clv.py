import os
import sqlite3
from datetime import datetime, UTC
import pandas as pd

DB_PATH = "mlb_f5_forecasts.db"
CSV_PATH = "edge_clv_input.csv"
BET_COLUMNS = ["threshold","date","away_team","home_team","pick_side","edge","odds_taken_american","odds_taken_decimal","close_prob_devig","clv","beat_close","result","updated_at"]
SUMMARY_COLUMNS = ["threshold","bets","avg_clv","beat_close","updated_at"]

def ensure_tables(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS edge_clv_bets (threshold REAL,date TEXT,away_team TEXT,home_team TEXT,pick_side TEXT,edge REAL,odds_taken_american INTEGER,odds_taken_decimal REAL,close_prob_devig REAL,clv REAL,beat_close INTEGER,result REAL,updated_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS edge_clv_summary (threshold REAL,bets INTEGER,avg_clv REAL,beat_close REAL,updated_at TEXT)")
    conn.commit()

def load_input(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = [c for c in BET_COLUMNS if c != "updated_at"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if df.empty:
        df["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        return df[BET_COLUMNS]
    df = df.copy()
    df["pick_side"] = df["pick_side"].astype(str).str.upper().str.strip()
    df = df[df["pick_side"].isin(["AWAY", "HOME"])].copy()
    if df.empty:
        raise ValueError("No valid rows after normalizing pick_side. Expected AWAY/HOME values.")
    df["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    return df[BET_COLUMNS]

def replace_edge_clv(conn, df):
    conn.execute("DELETE FROM edge_clv_bets")
    conn.execute("DELETE FROM edge_clv_summary")

    if not df.empty:
        placeholders = ", ".join(["?"] * len(BET_COLUMNS))
        conn.executemany(
            f"INSERT INTO edge_clv_bets ({', '.join(BET_COLUMNS)}) VALUES ({placeholders})",
            df.itertuples(index=False, name=None),
        )

    updated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    if df.empty:
        conn.execute(
            "INSERT INTO edge_clv_summary (threshold, bets, avg_clv, beat_close, updated_at) VALUES (?, ?, ?, ?, ?)",
            (0.05, 0, None, None, updated_at),
        )
        conn.commit()
        return

    summary = (
        df.groupby("threshold", dropna=False)
          .agg(bets=("threshold","size"), avg_clv=("clv","mean"), beat_close=("beat_close","mean"))
          .reset_index()
    )
    summary["updated_at"] = updated_at
    summary = summary[SUMMARY_COLUMNS]

    summary_placeholders = ", ".join(["?"] * len(SUMMARY_COLUMNS))
    conn.executemany(
        f"INSERT INTO edge_clv_summary ({', '.join(SUMMARY_COLUMNS)}) VALUES ({summary_placeholders})",
        summary.itertuples(index=False, name=None),
    )
    conn.commit()

def main():
    print(f"Using DB: {os.path.abspath(DB_PATH)}")
    print(f"Using CSV: {os.path.abspath(CSV_PATH)}")
    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_tables(conn)
        df = load_input(CSV_PATH)
        replace_edge_clv(conn, df)
        print(f"Loaded {len(df)} rows into edge_clv_bets.")
        print("Rebuilt edge_clv_summary.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
