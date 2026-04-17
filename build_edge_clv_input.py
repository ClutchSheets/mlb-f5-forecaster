import sqlite3
import pandas as pd

DB_PATH = "mlb_f5_forecasts.db"
OUTPUT_CSV = "edge_clv_input.csv"
EDGE_THRESHOLD = 0.05


def prob_to_american(p: float) -> int:
    p = max(min(float(p), 0.9999), 0.0001)
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    return round((1 - p) / p * 100)


def main():
    conn = sqlite3.connect(DB_PATH)

    model_df = pd.read_sql(
        """
        SELECT
            date,
            away_team,
            home_team,
            winprob_home,
            home_win_f5,
            close_home_prob_devig
        FROM forecast_archive
        WHERE completed5 = 1
        """,
        conn,
    )

    odds_df = pd.read_sql(
        """
        SELECT
            date,
            away_team,
            home_team,
            side,
            price_american,
            price_decimal,
            book,
            captured_at
        FROM odds_snapshot
        WHERE snapshot_type = '6h'
        """,
        conn,
    )

    conn.close()

    model_df["date"] = pd.to_datetime(model_df["date"]).dt.strftime("%Y-%m-%d")
    odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.strftime("%Y-%m-%d")

    model_df["home_win_prob"] = pd.to_numeric(model_df["winprob_home"], errors="coerce")
    model_df["away_win_prob"] = 1 - model_df["home_win_prob"]

    model_df["home_result"] = pd.to_numeric(model_df["home_win_f5"], errors="coerce")
    model_df["away_result"] = 1 - model_df["home_result"]

    model_df["close_home_prob_devig"] = pd.to_numeric(model_df["close_home_prob_devig"], errors="coerce")
    model_df["close_away_prob_devig"] = 1 - model_df["close_home_prob_devig"]

    odds_df["price_american"] = pd.to_numeric(odds_df["price_american"], errors="coerce")
    odds_df["price_decimal"] = pd.to_numeric(odds_df["price_decimal"], errors="coerce")
    odds_df["side"] = odds_df["side"].astype(str).str.lower().str.strip()

    df = model_df.merge(
        odds_df,
        on=["date", "away_team", "home_team"],
        how="inner"
    ).copy()

    print(f"Model rows: {len(model_df)}")
    print(f"Odds rows: {len(odds_df)}")
    print(f"Merged rows: {len(df)}")

    df["pick_side"] = df["side"].map({"away": "AWAY", "home": "HOME"})

    df["model_prob"] = df.apply(
        lambda r: r["away_win_prob"] if r["side"] == "away" else r["home_win_prob"],
        axis=1,
    )

    df["close_prob_devig_side"] = df.apply(
        lambda r: r["close_away_prob_devig"] if r["side"] == "away" else r["close_home_prob_devig"],
        axis=1,
    )

    df["result"] = df.apply(
        lambda r: r["away_result"] if r["side"] == "away" else r["home_result"],
        axis=1,
    )

    df["taken_implied_prob"] = 1 / df["price_decimal"]

    # Edge = model probability - implied probability of the price taken
    df["edge"] = df["model_prob"] - df["taken_implied_prob"]

    # CLV = devigged closing implied probability - implied probability of the price taken
    df["clv"] = df["close_prob_devig_side"] - df["taken_implied_prob"]
    df["beat_close"] = (df["clv"] > 0).astype(int)

    df["game_key"] = df["date"] + "_" + df["away_team"] + "_" + df["home_team"]

    # One best bet per game: highest edge
    df_best = (
        df.sort_values(["edge", "captured_at"], ascending=[False, False])
          .drop_duplicates("game_key")
          .copy()
    )

    print(f"Unique games: {df['game_key'].nunique()}")
    print(f"Best bets (1 per game): {len(df_best)}")

    bets = df_best[df_best["edge"] >= EDGE_THRESHOLD].copy()
    print(f"Qualified bets (≥5%): {len(bets)}")

    if bets.empty:
        bets = pd.DataFrame(columns=[
            "threshold", "date", "away_team", "home_team", "pick_side", "edge",
            "odds_taken_american", "close_prob_devig", "clv", "beat_close", "result"
        ])
        bets.to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote empty {OUTPUT_CSV}")
        return

    bets["threshold"] = EDGE_THRESHOLD
    bets["odds_taken_american"] = bets["price_american"].astype(int)
    bets["close_prob_devig"] = bets["close_prob_devig_side"]
    bets["close_fair_odds_american"] = bets["close_prob_devig"].apply(prob_to_american)
    bets["taken_implied_prob"] = bets["taken_implied_prob"]

    out = bets[
        [
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
        ]
    ].copy()

    out = out.sort_values(["date", "away_team", "home_team"], ascending=[False, True, True]).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} bets -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
