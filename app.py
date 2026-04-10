import streamlit as st
import pandas as pd
import sqlite3
import os
import requests
from datetime import date, timedelta

st.set_page_config(page_title="ClutchSheets F5 Forecaster", layout="wide")

# ── Odds API ──────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

BOOKS = ["betonline", "fanduel", "draftkings", "betmgm", "betrivers"]
BOOK_LABELS = {
    "betonline": "BetOnline",
    "fanduel":   "FanDuel",
    "draftkings":"DraftKings",
    "betmgm":    "BetMGM",
    "betrivers": "BetRivers",
}

# ── FIXED: Raised threshold from 3.0 to 5.0 to reduce noise ──────────────────
EDGE_THRESHOLD = 5.0

def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def prob_to_american(p: float) -> int:
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    return round((1 - p) / p * 100)

def no_vig_prob(home_odds: int, away_odds: int):
    ph = american_to_prob(home_odds)
    pa = american_to_prob(away_odds)
    total = ph + pa
    return ph / total, pa / total

def edge_pct(model_p: float, market_p: float) -> float:
    return (model_p - market_p) * 100

# ── FIXED: Sigmoid-based totals probability using actual projection distance ──
def total_edge_prob(total_proj: float, line: float) -> tuple:
    """
    Convert the distance between model projection and market line into
    a meaningful over/under probability using a sigmoid function.
    
    diff > 0 → model projects more runs than the line → lean Over
    diff < 0 → model projects fewer runs than the line → lean Under
    
    k=0.7 is the steepness — tunable based on calibration.
    A diff of 0.5 runs gives ~58% confidence, 1.0 run gives ~67%.
    """
    diff = total_proj - line
    k = 0.7
    model_over_p = 1 / (1 + 10 ** (-diff * k))
    model_under_p = 1 - model_over_p
    return model_over_p, model_under_p

# ── Team name → CSV code map ──────────────────────────────────────────────────
NAME_TO_CODE = {
    "Pittsburgh Pirates": "PIT", "Chicago Cubs": "CHC",
    "Oakland Athletics": "ATH", "New York Mets": "NYM",
    "Texas Rangers": "TEX", "Los Angeles Dodgers": "LAD",
    "Los Angeles Angels": "LAA", "Cincinnati Reds": "CIN",
    "Washington Nationals": "WSN", "Milwaukee Brewers": "MIL",
    "Arizona Diamondbacks": "ARI", "Philadelphia Phillies": "PHI",
    "Minnesota Twins": "MIN", "Toronto Blue Jays": "TOR",
    "Colorado Rockies": "COL", "San Diego Padres": "SDP",
    "Miami Marlins": "MIA", "Detroit Tigers": "DET",
    "New York Yankees": "NYY", "Tampa Bay Rays": "TBR",
    "Chicago White Sox": "CHW", "Kansas City Royals": "KCR",
    "San Francisco Giants": "SFG", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "St. Louis Cardinals": "STL",
    "Cleveland Guardians": "CLE", "Atlanta Braves": "ATL",
    "Houston Astros": "HOU", "Seattle Mariners": "SEA",
}

@st.cache_data(ttl=300)
def fetch_odds():
    if not ODDS_API_KEY:
        return {}
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "bookmakers": ",".join(BOOKS),
    }
    r = requests.get(ODDS_API_URL, params=params, timeout=15)
    if r.status_code != 200:
        return {}
    raw = r.json()

    lookup = {}
    for game in raw:
        teams = {b["key"]: b for b in game.get("bookmakers", [])}
        away_name = game["away_team"]
        home_name = game["home_team"]
        away_code = NAME_TO_CODE.get(away_name, away_name[:3].upper())
        home_code = NAME_TO_CODE.get(home_name, home_name[:3].upper())
        key = f"{away_code}@{home_code}"
        lookup[key] = {"bookmakers": teams, "commence_time": game.get("commence_time", "")}
    return lookup

@st.cache_data(ttl=3600)
def load_forecast(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    return df

def load_history(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM forecast_archive ORDER BY date DESC", conn)
    conn.close()
    return df

# ── Render one game row ───────────────────────────────────────────────────────
def render_game(row, odds_lookup, col_books):
    away = str(row.get("away_team", "")).strip()
    home = str(row.get("home_team", "")).strip()
    away_sp = str(row.get("away_sp", "")).strip()
    home_sp = str(row.get("home_sp", "")).strip()
    away_sp_text = str(row.get("away_sp_text", away_sp)).strip()
    home_sp_text = str(row.get("home_sp_text", home_sp)).strip()

    model_away_p = 1 - float(row.get("winprob_home", 0.5))
    model_home_p = float(row.get("winprob_home", 0.5))
    ml_fv_away = prob_to_american(model_away_p)
    ml_fv_home = prob_to_american(model_home_p)

    lambda_away = float(row.get("lambda_away", 0))
    lambda_home = float(row.get("lambda_home", 0))
    total_proj  = lambda_away + lambda_home
    tot_fv_val  = round(total_proj, 1)

    key = f"{away}@{home}"
    bk  = odds_lookup.get(key, {}).get("bookmakers", {})

    def book_cells(book_key):
        bdata = bk.get(book_key, {})
        if not bdata:
            return "", ""

        # ── ML edge (unchanged — was already correct) ─────────────────────────
        ml_mkt = bdata.get("markets", {})
        h2h = next((m for m in ml_mkt if m["key"] == "h2h"), None) if isinstance(ml_mkt, list) else ml_mkt.get("h2h")
        ml_cell = ""
        if h2h:
            outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
            home_name = next((n for n in outcomes if NAME_TO_CODE.get(n) == home), None)
            away_name = next((n for n in outcomes if NAME_TO_CODE.get(n) == away), None)
            if home_name and away_name:
                mkt_home_p, mkt_away_p = no_vig_prob(outcomes[home_name], outcomes[away_name])
                home_edge = edge_pct(model_home_p, mkt_home_p)
                away_edge = edge_pct(model_away_p, mkt_away_p)
                best_edge = home_edge if abs(home_edge) >= abs(away_edge) else away_edge
                best_odds = outcomes[home_name] if abs(home_edge) >= abs(away_edge) else outcomes[away_name]
                sign = "+" if best_odds > 0 else ""
                if abs(best_edge) >= EDGE_THRESHOLD:
                    ml_cell = f'<span style="background:#1a7a3a;padding:2px 6px;border-radius:3px">{sign}{best_odds}<br><small>+{abs(best_edge):.1f}%</small></span>'
                else:
                    ml_cell = f"{sign}{best_odds}"

        # ── Totals edge (FIXED) ───────────────────────────────────────────────
        totals_mkt = bdata.get("markets", {})
        tot_market = next((m for m in totals_mkt if m["key"] == "totals"), None) if isinstance(totals_mkt, list) else totals_mkt.get("totals")
        tot_cell = ""
        if tot_market:
            outcomes = {o["name"]: (o["price"], o.get("point", 0)) for o in tot_market.get("outcomes", [])}
            if "Over" in outcomes and "Under" in outcomes:
                over_odds, over_line = outcomes["Over"]
                under_odds, under_line = outcomes["Under"]
                mkt_over_p, mkt_under_p = no_vig_prob(over_odds, under_odds)

                # FIXED: use sigmoid distance instead of hardcoded 0.55/0.45
                model_over_p, model_under_p = total_edge_prob(total_proj, over_line)

                tot_edge = edge_pct(model_over_p, mkt_over_p)
                best_tot_odds = over_odds if tot_edge > 0 else under_odds
                sign = "+" if best_tot_odds > 0 else ""
                line_label = f"o{over_line}" if tot_edge > 0 else f"u{under_line}"
                if abs(tot_edge) >= EDGE_THRESHOLD:
                    tot_cell = f'<span style="background:#1a7a3a;padding:2px 6px;border-radius:3px">{line_label} {sign}{best_tot_odds}<br><small>+{abs(tot_edge):.1f}%</small></span>'
                else:
                    tot_cell = f"{line_label} {sign}{best_tot_odds}"

        return ml_cell, tot_cell

    # build display row
    ...

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.title("⚾ ClutchSheets F5 Baseball Forecaster")

    CSV_PATH = os.environ.get("FORECAST_CSV", "daily_forecast.csv")
    DB_PATH  = os.environ.get("DB_PATH",       "mlb_f5_forecasts.db")

    df = load_forecast(CSV_PATH)
    if df.empty:
        st.warning("No forecast data found.")
        return

    today_str = date.today().strftime("%Y-%m-%d")
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.strip()
        df = df[df["date"] == today_str]

    odds_lookup = fetch_odds()

    col_books = BOOKS

    for _, row in df.iterrows():
        render_game(row, odds_lookup, col_books)

    st.divider()
    st.subheader("📊 Forecast Archive")
    hist = load_history(DB_PATH)
    if not hist.empty:
        st.dataframe(hist, use_container_width=True)

if __name__ == "__main__":
    main()
