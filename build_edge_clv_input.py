import sqlite3
import pandas as pd

DB_PATH = "mlb_f5_forecasts.db"
OUTPUT_CSV = "edge_clv_input.csv"
EDGE_THRESHOLD = 0.05

NAME_TO_CODE = {
    "Pittsburgh Pirates": "PIT",
    "Chicago Cubs": "CHC",
    "Athletics": "ATH",
    "Oakland Athletics": "ATH",
    "New York Mets": "NYM",
    "Texas Rangers": "TEX",
    "Los Angeles Dodgers": "LAD",
    "Los Angeles Angels": "LAA",
    "Cincinnati Reds": "CIN",
    "Washington Nationals": "WSN",
    "Milwaukee Brewers": "MIL",
    "Arizona Diamondbacks": "ARI",
    "Philadelphia Phillies": "PHI",
    "Minnesota Twins": "MIN",
    "Toronto Blue Jays": "TOR",
    "Colorado Rockies": "COL",
    "San Diego Padres": "SDP",
    "Miami Marlins": "MIA",
    "Detroit Tigers": "DET",
    "New York Yankees": "NYY",
    "Tampa Bay Rays": "TBR",
    "Chicago White Sox": "CHW",
    "Kansas City Royals": "KCR",
    "San Francisco Giants": "SFG",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "St. Louis Cardinals": "STL",
    "Cleveland Guardians": "CLE",
    "Atlanta Braves": "ATL",
    "Houston Astros": "HOU",
    "Seattle Mariners": "SEA",
}

OUTPUT_COLUMNS = [
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


def american_to_prob(odds):
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def write_empty():
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_CSV, index=False)
    print("No valid source rows after cleaning. Wrote empty edge_clv_input.csv with headers")


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
    model_df["close_home_prob_devig"] = pd.to_numeric(model_df["close_home_prob_devig"], errors="coerce")
    model_df["close_away_prob_devig"] = 1 - model_df["close_home_prob_devig"]

    odds_df["price_american"] = pd.to_numeric(odds_df["price_american"], errors="coerce")
    odds_df["price_decimal"] = pd.to_numeric(odds_df["price_decimal"], errors="coerce")

    odds_df["away_team_code"] = odds_df["away_team"].map(NAME_TO_CODE).fillna(odds_df["away_team"])
    odds_df["home_team_code"] = odds_df["home_team"].map(NAME_TO_CODE).fillna(odds_df["home_team"])
    odds_df["side_raw"] = odds_df["side"].astype(str).str.strip()
    odds_df["side_upper"] = odds_df["side_raw"].str.upper()
    odds_df["side_lower"] = odds_df["side_raw"].str.lower()

    odds_df["side_clean"] = None
    odds_df.loc[odds_df["side_lower"] == "away", "side_clean"] = "away"
    odds_df.loc[odds_df["side_lower"] == "home", "side_clean"] = "home"
    odds_df.loc[odds_df["side_upper"] == odds_df["away_team_code"], "side_clean"] = "away"
    odds_df.loc[odds_df["side_upper"] == odds_df["home_team_code"], "side_clean"] = "home"

    odds_df = odds_df.dropna(
        subset=["date", "away_team_code", "home_team_code", "side_clean", "price_american", "price_decimal", "captured_at"]
    ).copy()

    if odds_df.empty:
        write_empty()
        return

    model_df = model_df.dropna(
        subset=["date", "away_team", "home_team", "home_win_prob", "close_home_prob_devig", "home_result"]
    ).copy()

    if model_df.empty:
        write_empty()
        return

    df = model_df.merge(
        odds_df,
        left_on=["away_team", "home_team"],
        right_on=["away_team_code", "home_team_code"],
        how="inner",
        suffixes=("_game", "_odds"),
    ).copy()

    if df.empty:
        write_empty()
        return

    df["pick_side"] = df["side_clean"].map({"away": "AWAY", "home": "HOME"})
    df["model_prob"] = df.apply(
        lambda r: r["away_win_prob"] if r["side_clean"] == "away" else r["home_win_prob"],
        axis=1,
    )
    df["close_prob_devig"] = df.apply(
        lambda r: r["close_away_prob_devig"] if r["side_clean"] == "away" else r["close_home_prob_devig"],
        axis=1,
    )
    df["result"] = df.apply(
        lambda r: r["away_result"] if r["side_clean"] == "away" else r["home_result"],
        axis=1,
    )

    df["taken_implied_prob"] = df["price_american"].apply(american_to_prob)
    df["edge"] = df["model_prob"] - df["taken_implied_prob"]
    df["clv"] = df["close_prob_devig"] - (1 / df["price_decimal"])
    df["beat_close"] = (df["clv"] > 0).astype(int)

    df = df.dropna(
        subset=["pick_side", "edge", "clv", "close_prob_devig", "result"]
    ).copy()

    if df.empty:
        write_empty()
        return

    df["game_key"] = df["date_game"] + "_" + df["away_team"] + "_" + df["home_team"]
    df_best = (
        df.sort_values(["edge", "captured_at"], ascending=[False, False])
        .drop_duplicates("game_key")
        .copy()
    )

    bets = df_best[df_best["edge"] >= EDGE_THRESHOLD].copy()

    if bets.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_CSV, index=False)
        print("No bets found. Wrote empty edge_clv_input.csv with headers")
        return

    bets["threshold"] = EDGE_THRESHOLD
    bets["odds_taken_american"] = bets["price_american"].astype(int)
    bets["odds_taken_decimal"] = bets["price_decimal"]

    out = bets[
        [
            "threshold",
            "date_game",
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
    ].copy()

    out = out.rename(columns={"date_game": "date"})
    out = out.sort_values(["date", "away_team", "home_team"], ascending=[False, True, True]).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(out)} bets -> {OUTPUT_CSV}")
    print("Pick side values:", sorted(out["pick_side"].dropna().unique().tolist()))


if __name__ == "__main__":
    main()
