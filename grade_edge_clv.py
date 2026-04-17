import sqlite3
from datetime import datetime, timezone
import math
import pandas as pd

DB_FILE = "mlb_f5_forecasts.db"
SOFT_BOOKS = ["betonlineag", "fanduel", "draftkings", "betmgm", "betrivers"]
THRESHOLDS = [3.0, 5.0, 7.0]

TEAM_CODE_TO_NAME = {
    "PIT":"Pittsburgh Pirates","CHC":"Chicago Cubs","ATH":"Athletics","OAK":"Athletics",
    "NYM":"New York Mets","TEX":"Texas Rangers","LAD":"Los Angeles Dodgers","LAA":"Los Angeles Angels",
    "CIN":"Cincinnati Reds","WSN":"Washington Nationals","MIL":"Milwaukee Brewers",
    "ARI":"Arizona Diamondbacks","PHI":"Philadelphia Phillies","MIN":"Minnesota Twins","TOR":"Toronto Blue Jays",
    "COL":"Colorado Rockies","SDP":"San Diego Padres","MIA":"Miami Marlins","DET":"Detroit Tigers",
    "NYY":"New York Yankees","TBR":"Tampa Bay Rays","CHW":"Chicago White Sox","CWS":"Chicago White Sox",
    "KCR":"Kansas City Royals","SFG":"San Francisco Giants","BAL":"Baltimore Orioles","BOS":"Boston Red Sox",
    "STL":"St. Louis Cardinals","CLE":"Cleveland Guardians","ATL":"Atlanta Braves",
    "HOU":"Houston Astros","SEA":"Seattle Mariners"
}

def american_to_prob(o):
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)

def american_to_decimal(o):
    return (o/100)+1 if o > 0 else (100/abs(o))+1

def devig_pair(a, b):
    pa = american_to_prob(a)
    pb = american_to_prob(b)
    s = pa + pb
    return pa/s, pb/s

def ensure_tables(conn):
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
        CREATE TABLE IF NOT EXISTS edge_clv_summary (
            threshold_pct REAL,
            bets INTEGER,
            avg_clv REAL,
            beat_close_pct REAL,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_clv_bets (
            threshold_pct REAL,
            game_date TEXT,
            matchup TEXT,
            pick_side TEXT,
            edge_pct REAL,
            odds_taken_american INTEGER,
            odds_taken_decimal REAL,
            close_devig_prob REAL,
            clv REAL,
            beat_close INTEGER,
            result TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()

def result_for_side(home_win_f5, side):
    if pd.isna(home_win_f5):
        return "NA"
    if float(home_win_f5) == 0.5:
        return "Push"
    if side == "home":
        return "Win" if float(home_win_f5) == 1.0 else "Loss"
    return "Win" if float(home_win_f5) == 0.0 else "Loss"

def main():
    conn = sqlite3.connect(DB_FILE)
    ensure_tables(conn)

    forecasts = pd.read_sql("""
        SELECT date, away_team, home_team, winprob_home, home_win_f5, close_home_prob_devig
        FROM forecast_archive
        WHERE completed5 = 1
          AND close_home_prob_devig IS NOT NULL
    """, conn)

    odds = pd.read_sql("""
        SELECT date, away_team, home_team, book, side, price_american, price_decimal
        FROM odds_snapshot
        WHERE snapshot_type = '6h'
    """, conn)

    if forecasts.empty or odds.empty:
        print("[SKIP] Missing forecast rows or odds snapshots.")
        conn.close()
        return

    forecasts["away_name"] = forecasts["away_team"].map(TEAM_CODE_TO_NAME)
    forecasts["home_name"] = forecasts["home_team"].map(TEAM_CODE_TO_NAME)
    forecasts = forecasts.dropna(subset=["away_name", "home_name"]).copy()

    odds = odds[odds["book"].isin(SOFT_BOOKS)].copy()

    conn.execute("DELETE FROM edge_clv_summary")
    conn.execute("DELETE FROM edge_clv_bets")
    conn.commit()

    all_picks = []

    for _, row in forecasts.iterrows():
        game_odds = odds[
            (odds["date"] == row["date"]) &
            (odds["away_team"] == row["away_name"]) &
            (odds["home_team"] == row["home_name"])
        ].copy()
        if game_odds.empty:
            continue

        book_pairs = {}
        for book, grp in game_odds.groupby("book"):
            away_row = grp[grp["side"] == "away"]
            home_row = grp[grp["side"] == "home"]
            if away_row.empty or home_row.empty:
                continue
            away_price = int(away_row.iloc[0]["price_american"])
            home_price = int(home_row.iloc[0]["price_american"])
            away_dec = float(away_row.iloc[0]["price_decimal"])
            home_dec = float(home_row.iloc[0]["price_decimal"])
            away_nv, home_nv = devig_pair(away_price, home_price)
            book_pairs[book] = {
                "away_price": away_price,
                "home_price": home_price,
                "away_dec": away_dec,
                "home_dec": home_dec,
                "away_nv": away_nv,
                "home_nv": home_nv,
            }

        if not book_pairs:
            continue

        model_home = float(row["winprob_home"])
        model_away = 1.0 - model_home

        median_home_prob = pd.Series([v["home_nv"] for v in book_pairs.values()]).median()
        median_away_prob = pd.Series([v["away_nv"] for v in book_pairs.values()]).median()
        edge_home = (model_home - median_home_prob) * 100.0
        edge_away = (model_away - median_away_prob) * 100.0

        best_home = max(book_pairs.items(), key=lambda x: x[1]["home_dec"])
        best_away = max(book_pairs.items(), key=lambda x: x[1]["away_dec"])

        matchup = f'{row["away_team"]} @ {row["home_team"]}'
        close_home = float(row["close_home_prob_devig"])
        close_away = 1.0 - close_home

        for threshold in THRESHOLDS:
            qualifies_home = edge_home >= threshold
            qualifies_away = edge_away >= threshold
            if not qualifies_home and not qualifies_away:
                continue

            if qualifies_home and qualifies_away:
                side = "home" if edge_home >= edge_away else "away"
            else:
                side = "home" if qualifies_home else "away"

            if side == "home":
                book, take = best_home
                edge_pct = edge_home
                close_prob = close_home
            else:
                book, take = best_away
                edge_pct = edge_away
                close_prob = close_away

            odds_taken_american = int(take[f"{side}_price"])
            odds_taken_decimal = float(take[f"{side}_dec"])
            implied_taken_prob = 1.0 / odds_taken_decimal
            clv = implied_taken_prob - close_prob
            beat_close = 1 if clv > 0 else 0
            result = result_for_side(row["home_win_f5"], side)

            all_picks.append({
                "threshold_pct": threshold,
                "game_date": row["date"],
                "matchup": matchup,
                "pick_side": side.upper(),
                "edge_pct": edge_pct,
                "odds_taken_american": odds_taken_american,
                "odds_taken_decimal": odds_taken_decimal,
                "close_devig_prob": close_prob,
                "clv": clv,
                "beat_close": beat_close,
                "result": result,
            })

    if not all_picks:
        print("[SKIP] No qualifying bets for any threshold.")
        conn.close()
        return

    picks = pd.DataFrame(all_picks)
    updated_at = datetime.now(timezone.utc).isoformat()

    summary_rows = []
    for threshold in THRESHOLDS:
        sub = picks[picks["threshold_pct"] == threshold].copy()
        if sub.empty:
            summary_rows.append((threshold, 0, None, None, updated_at))
            continue
        summary_rows.append((
            threshold,
            int(len(sub)),
            float(sub["clv"].mean()),
            float(sub["beat_close"].mean()),
            updated_at
        ))

    conn.executemany(
        "INSERT INTO edge_clv_summary (threshold_pct, bets, avg_clv, beat_close_pct, updated_at) VALUES (?, ?, ?, ?, ?)",
        summary_rows
    )
    picks["updated_at"] = updated_at
    picks.to_sql("edge_clv_bets", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print("[OK] edge_clv_summary and edge_clv_bets refreshed.")

if __name__ == "__main__":
    main()
