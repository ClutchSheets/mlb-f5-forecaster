import argparse
import sqlite3
import pandas as pd

DB_FILE = "mlb_f5_forecasts.db"

NAME_TO_CODE = {
    "Pittsburgh Pirates":"PIT","Chicago Cubs":"CHC","Oakland Athletics":"ATH","Athletics":"ATH",
    "New York Mets":"NYM","Texas Rangers":"TEX","Los Angeles Dodgers":"LAD","Los Angeles Angels":"LAA",
    "Cincinnati Reds":"CIN","Washington Nationals":"WSN","Milwaukee Brewers":"MIL",
    "Arizona Diamondbacks":"ARI","Philadelphia Phillies":"PHI","Minnesota Twins":"MIN",
    "Toronto Blue Jays":"TOR","Colorado Rockies":"COL","San Diego Padres":"SDP","Miami Marlins":"MIA",
    "Detroit Tigers":"DET","New York Yankees":"NYY","Tampa Bay Rays":"TBR","Chicago White Sox":"CHW",
    "Kansas City Royals":"KCR","San Francisco Giants":"SFG","Baltimore Orioles":"BAL","Boston Red Sox":"BOS",
    "St. Louis Cardinals":"STL","Cleveland Guardians":"CLE","Atlanta Braves":"ATL","Houston Astros":"HOU",
    "Seattle Mariners":"SEA"
}

def american_to_decimal(o):
    o = int(o)
    return o/100 + 1 if o > 0 else 100/abs(o) + 1

def norm(x):
    x = str(x).strip()
    return NAME_TO_CODE.get(x, x)

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
            captured_at TEXT,
            UNIQUE(date, away_team, home_team, book, side, snapshot_type)
        )
    """)
    conn.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--snapshot-type", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required = {"date","away_team","home_team","book","side","price_american"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["away_team"] = df["away_team"].map(norm)
    df["home_team"] = df["home_team"].map(norm)
    df["side"] = df["side"].map(norm)
    df["book"] = df["book"].astype(str).str.lower().str.strip()

    df["price_american"] = pd.to_numeric(df["price_american"], errors="coerce")
    df = df.dropna(subset=["price_american"])
    df["price_american"] = df["price_american"].astype(int)
    df["price_decimal"] = df["price_american"].map(american_to_decimal)

    df["snapshot_type"] = args.snapshot_type

    if "timestamp_utc" in df.columns:
        df["captured_at"] = df["timestamp_utc"].astype(str)
    else:
        df["captured_at"] = None

    # Deduplicate
    df = df.sort_values(["date","away_team","home_team","book","side","captured_at"])
    df = df.drop_duplicates(
        subset=["date","away_team","home_team","book","side","snapshot_type"],
        keep="last"
    )

    conn = sqlite3.connect(DB_FILE)
    ensure_table(conn)

    # IMPORTANT: only delete same snapshot_type (so 6h stays intact)
    dates = df["date"].unique()
    for d in dates:
        conn.execute(
            "DELETE FROM odds_snapshot WHERE date=? AND snapshot_type=?",
            (d, args.snapshot_type)
        )
    conn.commit()

    rows = list(df[[
        "date","away_team","home_team","book","side",
        "price_american","price_decimal","snapshot_type","captured_at"
    ]].itertuples(index=False, name=None))

    conn.executemany("""
        INSERT OR REPLACE INTO odds_snapshot
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()

    print(f"[OK] Imported {len(rows)} rows as {args.snapshot_type}")

if __name__ == "__main__":
    main()
