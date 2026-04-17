import sqlite3
import pandas as pd

DB_PATH = "mlb_f5_forecasts.db"
OUTPUT_CSV = "edge_clv_input.csv"
EDGE_THRESHOLD = 0.05


def main() -> None:
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
          AND winprob_home IS NOT NULL
          AND home_win_f5 IS NOT NULL
          AND close_home_prob_devig IS NOT NULL
        """,
        conn,
    )

    odds_df = pd.read_sql(
        """
        SELECT
            date,
            away_team,
            home_team,
            book,
            side,
            price_american,
            price_decimal
        FROM odds_snapshot
        WHERE snapshot_type = '6h'
          AND price_american IS NOT NULL
          AND price_decimal IS NOT NULL
        """,
        conn,
    )

    conn.close()

    model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    odds_df["date"] = pd.to_datetime(odds_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for col in ["away_team", "home_team"]:
        model_df[col] = model_df[col].astype(str).str.strip().str.upper()
        odds_df[col] = odds_df[col].astype(str).str.strip().str.upper()

    odds_df["side"] = odds_df["side"].astype(str).str.strip().str.lower()

    merged = model_df.merge(
        odds_df,
        on=["date", "away_team", "home_team"],
        how="inner",
    ).copy()

    print(f"Model rows: {len(model_df)}")
    print(f"Odds rows: {len(odds_df)}")
    print(f"Merged rows: {len(merged)}")

    merged["home_win_prob"] = pd.to_numeric(merged["winprob_home"], errors="coerce")
    merged["away_win_prob"] = 1.0 - merged["home_win_prob"]

    merged["home_result"] = pd.to_numeric(merged["home_win_f5"], errors="coerce")
    merged["away_result"] = 1.0 - merged["home_result"]

    merged["close_home_prob_devig"] = pd.to_numeric(merged["close_home_prob_devig"], errors="coerce")
    merged["close_away_prob_devig"] = 1.0 - merged["close_home_prob_devig"]

    merged["model_prob"] = merged.apply(
        lambda r: r["away_win_prob"] if r["side"] == "away" else r["home_win_prob"],
        axis=1,
    )
    merged["close_prob_devig"] = merged.apply(
        lambda r: r["close_away_prob_devig"] if r["side"] == "away" else r["close_home_prob_devig"],
        axis=1,
    )
    merged["result"] = merged.apply(
        lambda r: r["away_result"] if r["side"] == "away" else r["home_result"],
        axis=1,
    )

    implied_prob = 1.0 / pd.to_numeric(merged["price_decimal"], errors="coerce")
    merged["edge"] = merged["model_prob"] - implied_prob
    merged["clv"] = merged["close_prob_devig"] - implied_prob
    merged["beat_close"] = (merged["clv"] > 0).astype(int)

    merged["game_key"] = (
        merged["date"].astype(str) + "_" + merged["away_team"] + "_" + merged["home_team"]
    )

    best_per_game = (
        merged.sort_values(["game_key", "edge"], ascending=[True, False])
        .drop_duplicates(subset=["game_key"], keep="first")
        .copy()
    )

    print(f"Unique games: {best_per_game['game_key'].nunique()}")

    bets = best_per_game[best_per_game["edge"] >= EDGE_THRESHOLD].copy()
    print(f"Qualified bets (>=5%): {len(bets)}")

    if bets.empty:
        pd.DataFrame(
            columns=[
                "threshold",
                "date",
                "away_team",
                "home_team",
                "pick_side",
                "edge",
                "odds_taken_american",
                "odds_taken_decimal",
                "close_prob_devig",
                "clv",
                "beat_close",
                "result",
            ]
        ).to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote empty file -> {OUTPUT_CSV}")
        return

    bets["threshold"] = EDGE_THRESHOLD
    bets["pick_side"] = bets["side"].str.upper()
    bets["odds_taken_american"] = bets["price_american"].astype(int)
    bets["odds_taken_decimal"] = bets["price_decimal"].astype(float)

    out = bets[
        [
            "threshold",
            "date",
            "away_team",
            "home_team",
            "pick_side",
            "edge",
            "odds_taken_american",
            "odds_taken_decimal",
            "close_prob_devig",
            "clv",
            "beat_close",
            "result",
        ]
    ].sort_values(["date", "away_team", "home_team"]).reset_index(drop=True)

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} bets -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()