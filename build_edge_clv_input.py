import sqlite3
import pandas as pd

DB_PATH = "mlb_f5_forecasts.db"
OUTPUT_CSV = "edge_clv_input.csv"
EDGE_THRESHOLD = 0.05

OUTPUT_COLUMNS = [
    "threshold", "date", "away_team", "home_team", "pick_side",
    "edge", "odds_taken_american", "odds_taken_decimal",
    "close_prob_devig", "clv", "beat_close", "result",
]

def american_to_prob(odds):
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def write_empty():
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_CSV, index=False)
    print("No valid rows. Wrote empty edge_clv_input.csv")

def main():
    conn = sqlite3.connect(DB_PATH)

    model_df = pd.read_sql("""
        SELECT date, away_team, home_team, winprob_home, home_win_f5, close_home_prob_devig
        FROM forecast_archive WHERE completed5 = 1
    """, conn)

    odds_df = pd.read_sql("""
        SELECT date, away_team, home_team, side, price_american, price_decimal, book, captured_at
        FROM odds_snapshot WHERE snapshot_type = '6h'
    """, conn)

    conn.close()

    if model_df.empty or odds_df.empty:
        write_empty()
        return

    model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    odds_df["date"] = pd.to_datetime(odds_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    odds_df["captured_at"] = pd.to_datetime(odds_df["captured_at"], errors="coerce", utc=True)

    model_df["home_win_prob"] = pd.to_numeric(model_df["winprob_home"], errors="coerce")
    model_df["away_win_prob"] = 1 - model_df["home_win_prob"]
    model_df["home_result"] = pd.to_numeric(model_df["home_win_f5"], errors="coerce")
    model_df["away_result"] = 1 - model_df["home_result"]
    model_df["close_home_prob"] = pd.to_numeric(model_df["close_home_prob_devig"], errors="coerce")
    model_df["close_away_prob"] = 1 - model_df["close_home_prob"]

    odds_df["price_american"] = pd.to_numeric(odds_df["price_american"], errors="coerce")
    odds_df["price_decimal"] = pd.to_numeric(odds_df["price_decimal"], errors="coerce")

    # Resolve side to away/home regardless of whether it's a code or word
    def resolve_side(row):
        s = str(row["side"]).strip().lower()
        if s == "away":
            return "away"
        if s == "home":
            return "home"
        # It's a team code — compare against away/home team codes
        s_upper = s.upper()
        if s_upper == str(row["away_team"]).upper():
            return "away"
        if s_upper == str(row["home_team"]).upper():
            return "home"
        return None

    odds_df["side_clean"] = odds_df.apply(resolve_side, axis=1)
    odds_df = odds_df.dropna(subset=["side_clean", "price_american", "price_decimal", "captured_at"]).copy()

    if odds_df.empty:
        write_empty()
        return

    model_df = model_df.dropna(
        subset=["date", "away_team", "home_team", "home_win_prob", "close_home_prob", "home_result"]
    ).copy()

    # Merge on date + team codes
    df = model_df.merge(
        odds_df,
        on=["date", "away_team", "home_team"],
        how="inner"
    ).copy()

    print(f"Model rows: {len(model_df)} | Odds rows: {len(odds_df)} | Merged: {len(df)}")

    if df.empty:
        write_empty()
        return

    df["pick_side"] = df["side_clean"].map({"away": "AWAY", "home": "HOME"})
    df["model_prob"] = df.apply(
        lambda r: r["away_win_prob"] if r["side_clean"] == "away" else r["home_win_prob"], axis=1
    )
    df["close_prob_devig"] = df.apply(
        lambda r: r["close_away_prob"] if r["side_clean"] == "away" else r["close_home_prob"], axis=1
    )
    df["result"] = df.apply(
        lambda r: r["away_result"] if r["side_clean"] == "away" else r["home_result"], axis=1
    )

    df["taken_implied_prob"] = df["price_american"].apply(american_to_prob)
    df["edge"] = df["model_prob"] - df["taken_implied_prob"]
    df["clv"] = df["close_prob_devig"] - (1 / df["price_decimal"])
    df["beat_close"] = (df["clv"] > 0).astype(int)

    df = df.dropna(subset=["pick_side", "edge", "clv", "close_prob_devig", "result"]).copy()

    if df.empty:
        write_empty()
        return

    df["game_key"] = df["date"] + "_" + df["away_team"] + "_" + df["home_team"]
    df_best = (
        df.sort_values(["edge", "captured_at"], ascending=[False, False])
        .drop_duplicates("game_key")
        .copy()
    )

    bets = df_best[df_best["edge"] >= EDGE_THRESHOLD].copy()
    print(f"Unique games: {df['game_key'].nunique()} | Qualified bets (>=5%): {len(bets)}")

    if bets.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_CSV, index=False)
        print("No bets found.")
        return

    bets["threshold"] = EDGE_THRESHOLD
    bets["odds_taken_american"] = bets["price_american"].astype(int)
    bets["odds_taken_decimal"] = bets["price_decimal"]

    out = bets[["threshold", "date", "away_team", "home_team", "pick_side",
                "edge", "odds_taken_american", "odds_taken_decimal",
                "close_prob_devig", "clv", "beat_close", "result"]].copy()

    out = out.sort_values(["date", "away_team", "home_team"], ascending=[False, True, True]).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} bets -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
