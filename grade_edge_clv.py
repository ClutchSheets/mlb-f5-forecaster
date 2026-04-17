
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DB_FILE = "mlb_f5_forecasts.db"
THRESHOLDS = [3.0, 5.0, 7.0]
SNAPSHOT_TYPE = "6h"

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


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_clv_summary (
            threshold REAL,
            bets INTEGER,
            wins INTEGER,
            losses INTEGER,
            pushes INTEGER,
            win_rate_ex_push REAL,
            avg_clv REAL,
            median_clv REAL,
            beat_close_pct REAL,
            updated_at TEXT,
            UNIQUE(threshold)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_clv_detail (
            threshold REAL,
            date TEXT,
            away_team TEXT,
            home_team TEXT,
            pick_side TEXT,
            pick_team TEXT,
            edge_pct REAL,
            odds_taken_american INTEGER,
            odds_taken_decimal REAL,
            close_prob_devig REAL,
            implied_prob_taken REAL,
            clv REAL,
            beat_close INTEGER,
            result REAL,
            updated_at TEXT
        )
    """)
    conn.commit()


def no_vig_prob(odds_a: int, odds_b: int):
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total


def american_to_prob(o: int) -> float:
    return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)


def american_to_decimal(o: int) -> float:
    if o > 0:
        return (o / 100) + 1
    return (100 / abs(o)) + 1


def load_data(conn: sqlite3.Connection) -> pd.DataFrame:
    forecast = pd.read_sql("""
        SELECT date, away_team, home_team, winprob_home, home_win_f5, close_home_prob_devig
        FROM forecast_archive
        WHERE completed5 = 1
          AND close_home_prob_devig IS NOT NULL
    """, conn)

    if forecast.empty:
        return forecast

    forecast["away_name"] = forecast["away_team"].map(TEAM_CODE_TO_NAME)
    forecast["home_name"] = forecast["home_team"].map(TEAM_CODE_TO_NAME)
    forecast = forecast.dropna(subset=["away_name", "home_name"]).copy()

    odds = pd.read_sql("""
        SELECT date, away_team, home_team, book, side, price_american, price_decimal
        FROM odds_snapshot
        WHERE snapshot_type = ?
    """, conn, params=[SNAPSHOT_TYPE])

    if odds.empty:
        return pd.DataFrame()

    return forecast, odds


def build_game_metrics(forecast: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = odds.groupby(["date", "away_team", "home_team"])
    for (date, away_name, home_name), grp in grouped:
        f = forecast[
            (forecast["date"] == date) &
            (forecast["away_name"] == away_name) &
            (forecast["home_name"] == home_name)
        ]
        if f.empty:
            continue

        fr = f.iloc[0]
        model_home = float(fr["winprob_home"])
        model_away = 1 - model_home
        close_home = float(fr["close_home_prob_devig"])
        close_away = 1 - close_home
        result_home = fr["home_win_f5"]
        result_away = np.nan if pd.isna(result_home) else (1 - result_home if result_home in (0, 1) else 0.5)

        # best prices
        best_home = grp[grp["side"] == "home"].sort_values(["price_decimal", "price_american"], ascending=False).head(1)
        best_away = grp[grp["side"] == "away"].sort_values(["price_decimal", "price_american"], ascending=False).head(1)
        if best_home.empty or best_away.empty:
            continue

        best_home = best_home.iloc[0]
        best_away = best_away.iloc[0]

        # median soft devig fair probs from books with both sides
        fair_home_probs = []
        fair_away_probs = []
        for book, bg in grp.groupby("book"):
            sides = set(bg["side"].tolist())
            if "home" not in sides or "away" not in sides:
                continue
            home_odds = int(bg.loc[bg["side"] == "home", "price_american"].iloc[0])
            away_odds = int(bg.loc[bg["side"] == "away", "price_american"].iloc[0])
            away_p, home_p = no_vig_prob(away_odds, home_odds)
            fair_home_probs.append(home_p)
            fair_away_probs.append(away_p)

        if not fair_home_probs or not fair_away_probs:
            continue

        market_home = float(np.median(fair_home_probs))
        market_away = float(np.median(fair_away_probs))

        edge_home = (model_home - market_home) * 100
        edge_away = (model_away - market_away) * 100

        implied_home = 1 / float(best_home["price_decimal"])
        implied_away = 1 / float(best_away["price_decimal"])

        clv_home = implied_home - close_home
        clv_away = implied_away - close_away

        rows.append({
            "date": date,
            "away_team": fr["away_team"],
            "home_team": fr["home_team"],
            "away_name": away_name,
            "home_name": home_name,
            "edge_home": edge_home,
            "edge_away": edge_away,
            "odds_home_american": int(best_home["price_american"]),
            "odds_away_american": int(best_away["price_american"]),
            "odds_home_decimal": float(best_home["price_decimal"]),
            "odds_away_decimal": float(best_away["price_decimal"]),
            "close_home_prob_devig": close_home,
            "close_away_prob_devig": close_away,
            "implied_home": implied_home,
            "implied_away": implied_away,
            "clv_home": clv_home,
            "clv_away": clv_away,
            "result_home": result_home,
            "result_away": result_away,
        })

    return pd.DataFrame(rows)


def select_bets(game_metrics: pd.DataFrame, threshold: float) -> pd.DataFrame:
    picks = []
    for _, r in game_metrics.iterrows():
        home_ok = r["edge_home"] >= threshold
        away_ok = r["edge_away"] >= threshold

        if not home_ok and not away_ok:
            continue

        if home_ok and away_ok:
            side = "home" if r["edge_home"] >= r["edge_away"] else "away"
        else:
            side = "home" if home_ok else "away"

        if side == "home":
            picks.append({
                "threshold": threshold,
                "date": r["date"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "pick_side": "home",
                "pick_team": r["home_team"],
                "edge_pct": r["edge_home"],
                "odds_taken_american": r["odds_home_american"],
                "odds_taken_decimal": r["odds_home_decimal"],
                "close_prob_devig": r["close_home_prob_devig"],
                "implied_prob_taken": r["implied_home"],
                "clv": r["clv_home"],
                "beat_close": int(r["clv_home"] > 0),
                "result": r["result_home"],
            })
        else:
            picks.append({
                "threshold": threshold,
                "date": r["date"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "pick_side": "away",
                "pick_team": r["away_team"],
                "edge_pct": r["edge_away"],
                "odds_taken_american": r["odds_away_american"],
                "odds_taken_decimal": r["odds_away_decimal"],
                "close_prob_devig": r["close_away_prob_devig"],
                "implied_prob_taken": r["implied_away"],
                "clv": r["clv_away"],
                "beat_close": int(r["clv_away"] > 0),
                "result": r["result_away"],
            })

    return pd.DataFrame(picks)


def summarize(df: pd.DataFrame, threshold: float) -> dict:
    if df.empty:
        return {
            "threshold": threshold,
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate_ex_push": None,
            "avg_clv": None,
            "median_clv": None,
            "beat_close_pct": None,
        }

    wins = int((df["result"] == 1).sum())
    losses = int((df["result"] == 0).sum())
    pushes = int((df["result"] == 0.5).sum())
    decisions = wins + losses
    return {
        "threshold": threshold,
        "bets": int(len(df)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate_ex_push": (wins / decisions) if decisions else None,
        "avg_clv": float(df["clv"].mean()),
        "median_clv": float(df["clv"].median()),
        "beat_close_pct": float((df["clv"] > 0).mean()),
    }


def main():
    conn = sqlite3.connect(DB_FILE)
    ensure_tables(conn)

    loaded = load_data(conn)
    if isinstance(loaded, pd.DataFrame) and loaded.empty:
        print("No qualifying source data found.")
        conn.close()
        return

    forecast, odds = loaded
    game_metrics = build_game_metrics(forecast, odds)
    if game_metrics.empty:
        print("No matched forecast/odds rows.")
        conn.close()
        return

    updated_at = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM edge_clv_summary")
    conn.execute("DELETE FROM edge_clv_detail")

    summaries = []
    all_details = []

    for threshold in THRESHOLDS:
        detail = select_bets(game_metrics, threshold)
        if not detail.empty:
            detail = detail.copy()
            detail["updated_at"] = updated_at
            all_details.append(detail)
        summaries.append(summarize(detail, threshold))

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df["updated_at"] = updated_at
        summary_df.to_sql("edge_clv_summary", conn, if_exists="append", index=False)

    if all_details:
        detail_df = pd.concat(all_details, ignore_index=True)
        detail_df.to_sql("edge_clv_detail", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    print("Edge CLV grading complete.")
    for s in summaries:
        print(
            f"Edge >= {s['threshold']:.0f}% | bets={s['bets']} | "
            f"avg_clv={'' if s['avg_clv'] is None else round(s['avg_clv']*100, 4)}% | "
            f"beat_close={'' if s['beat_close_pct'] is None else round(s['beat_close_pct']*100, 2)}%"
        )


if __name__ == "__main__":
    main()
