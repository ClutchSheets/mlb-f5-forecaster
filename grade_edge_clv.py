import os
import sqlite3
from datetime import datetime, UTC

import pandas as pd

DB_PATH = "mlb_f5_forecasts.db"
CSV_PATH = "edge_clv_input.csv"

BET_COLUMNS = [
    "threshold",
    "date",
    "away_team",
    "home_team",
    "pick_side",
    "edge",
    "odds_taken_american",
    "taken_implied_prob",
    "close_prob_devig",
    "close_fair_odds_american",
    "clv",
    "beat_close",
    "result",
    "updated_at",
]

SUMMARY_COLUMNS = [
    "threshold",
    "bets",
    "avg_clv",
    "beat_close",
    "updated_at",
]


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edge_clv_bets (
            threshold REAL,
            date TEXT,
            away_team TEXT,
            home_team TEXT,
            pick_side TEXT,
            edge REAL,
            odds_taken_american INTEGER,
            taken_implied_prob REAL,
            close_prob_devig REAL,
            close_fair_odds_american INTEGER,
            clv REAL,
            beat_close INTEGER,
            result REAL,
            updated_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edge_clv_summary (
            threshold REAL,
            bets INTEGER,
            avg_clv REAL,
            beat_close REAL,
            updated_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_edge_clv_bets_unique
        ON edge_clv_bets (
            date,
            away_team,
            home_team,
            pick_side,
            odds_taken_american
        )
        """
    )

    conn.commit()


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [row[1] for row in rows]


def validate_schema(conn: sqlite3.Connection) -> None:
    bet_cols = get_table_columns(conn, "edge_clv_bets")
    summary_cols = get_table_columns(conn, "edge_clv_summary")

    if bet_cols != BET_COLUMNS:
        raise RuntimeError(
            "edge_clv_bets schema does not match expected columns.\n"
            f"Expected: {BET_COLUMNS}\nActual:   {bet_cols}"
        )

    if summary_cols != SUMMARY_COLUMNS:
        raise RuntimeError(
            "edge_clv_summary schema does not match expected columns.\n"
            f"Expected: {SUMMARY_COLUMNS}\nActual:   {summary_cols}"
        )


def load_input(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = [c for c in BET_COLUMNS if c != "updated_at"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()
    df["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    df = df[BET_COLUMNS]
    return df


def append_new_bets(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    placeholders = ", ".join(["?"] * len(BET_COLUMNS))

    before = conn.execute("SELECT COUNT(*) FROM edge_clv_bets").fetchone()[0]

    conn.executemany(
        f"""
        INSERT OR IGNORE INTO edge_clv_bets ({', '.join(BET_COLUMNS)})
        VALUES ({placeholders})
        """,
        df.itertuples(index=False, name=None),
    )
    conn.commit()

    after = conn.execute("SELECT COUNT(*) FROM edge_clv_bets").fetchone()[0]
    return after - before


def rebuild_summary(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM edge_clv_summary")

    updated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    summary_rows = conn.execute(
        """
        SELECT
            threshold,
            COUNT(*) AS bets,
            AVG(clv) AS avg_clv,
            AVG(beat_close) AS beat_close
        FROM edge_clv_bets
        GROUP BY threshold
        ORDER BY threshold
        """
    ).fetchall()

    conn.executemany(
        """
        INSERT INTO edge_clv_summary (
            threshold, bets, avg_clv, beat_close, updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        [(row[0], row[1], row[2], row[3], updated_at) for row in summary_rows],
    )
    conn.commit()


def main() -> None:
    print(f"Using DB: {os.path.abspath(DB_PATH)}")
    print(f"Using CSV: {os.path.abspath(CSV_PATH)}")

    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_tables(conn)
        validate_schema(conn)
        df = load_input(CSV_PATH)
        inserted = append_new_bets(conn, df)
        rebuild_summary(conn)
        print(f"New rows inserted: {inserted}")
        print("edge_clv_summary rebuilt from all stored bets.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
