import argparse
import glob
import os
import sqlite3
from datetime import datetime, timezone
import pandas as pd

DB_FILE = "mlb_f5_forecasts.db"

REQUIRED = ["date", "away_team", "home_team", "book", "side", "price_american"]

def american_to_decimal(o):
    return (o/100)+1 if o > 0 else (100/abs(o))+1

def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshot (
            date TEXT,
            away_team TEXT,
            home_team TEXT,
            book TEXT,
            side TEXT,
            price_american INTEGER,
            price_decimal REAL,
            snapshot_type TEXT,
            captured_at TEXT
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_odds_snapshot_unique
        ON odds_snapshot(date, away_team, home_team, book, side, snapshot_type)
    """)
    conn.commit()

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["price_american"] = pd.to_numeric(df["price_american"], errors="coerce")
    df = df.dropna(subset=["date", "away_team", "home_team", "book", "side", "price_american"]).copy()
    df["price_american"] = df["price_american"].astype(int)
    df["price_decimal"] = df["price_american"].apply(american_to_decimal)
    return df

def main():
    parser = argparse.ArgumentParser(description="Import historical 6h odds CSVs into odds_snapshot.")
    parser.add_argument("paths", nargs="*", help="CSV files to import. Defaults to f5_6h_odds*.csv in current folder.")
    parser.add_argument("--snapshot-type", default="6h")
    args = parser.parse_args()

    paths = args.paths or sorted(glob.glob("f5_6h_odds*.csv"))
    if not paths:
        print("[SKIP] No CSV files found.")
        return

    conn = sqlite3.connect(DB_FILE)
    ensure_table(conn)

    total_rows = 0
    captured_at = datetime.now(timezone.utc).isoformat()

    for path in paths:
        df = load_csv(path)
        rows = []
        for _, r in df.iterrows():
            rows.append((
                r["date"],
                str(r["away_team"]).strip(),
                str(r["home_team"]).strip(),
                str(r["book"]).strip(),
                str(r["side"]).strip(),
                int(r["price_american"]),
                float(r["price_decimal"]),
                args.snapshot_type,
                captured_at,
            ))
        conn.executemany("""
            INSERT OR REPLACE INTO odds_snapshot
            (date, away_team, home_team, book, side, price_american, price_decimal, snapshot_type, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        total_rows += len(rows)
        print(f"[OK] Imported {len(rows)} rows from {os.path.basename(path)}")

    conn.close()
    print(f"[DONE] Imported {total_rows} total rows into odds_snapshot.")

if __name__ == "__main__":
    main()
