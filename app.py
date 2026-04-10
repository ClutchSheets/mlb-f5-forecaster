import streamlit as st
import pandas as pd
import sqlite3
import os
import requests
import numpy as np
import altair as alt
from datetime import date

st.set_page_config(page_title="ClutchSheets F5 Forecaster", layout="wide")

# ── Odds API ──────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "d31971f20a2edb069b228e828826815a")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

BOOKS = ["betonline", "fanduel", "draftkings", "betmgm", "betrivers"]
BOOK_LABELS = {
    "betonline":  "BetOnline",
    "fanduel":    "FanDuel",
    "draftkings": "DraftKings",
    "betmgm":     "BetMGM",
    "betrivers":  "BetRivers",
}

EDGE_THRESHOLD = 5.0  # raised from 3.0 to reduce noise

# ── Helpers ───────────────────────────────────────────────────────────────────
def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def prob_to_american(p: float) -> int:
    p = max(min(p, 0.9999), 0.0001)
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    return round((1 - p) / p * 100)

def no_vig_prob(odds_a: int, odds_b: int):
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total

def edge_pct(model_p: float, market_p: float) -> float:
    return (model_p - market_p) * 100

def total_edge_prob(total_proj: float, line: float):
    """Sigmoid-based probability from projection distance vs line."""
    diff = total_proj - line
    k = 0.7
    over_p = 1 / (1 + 10 ** (-diff * k))
    return over_p, 1 - over_p

def fmt_american(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)

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

# ── Data fetching ─────────────────────────────────────────────────────────────
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
        bmakers = {b["key"]: b for b in game.get("bookmakers", [])}
        away_name = game["away_team"]
        home_name = game["home_team"]
        away_code = NAME_TO_CODE.get(away_name, away_name[:3].upper())
        home_code = NAME_TO_CODE.get(home_name, home_name[:3].upper())
        key = f"{away_code}@{home_code}"
        lookup[key] = {"bookmakers": bmakers, "commence_time": game.get("commence_time", "")}
    return lookup

@st.cache_data(ttl=300)
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

# ── Book cell builder (ML only) ───────────────────────────────────────────────
def get_book_cells(book_key, bk, away, home, model_away_p, model_home_p, total_proj):
    bdata = bk.get(book_key, {})
    if not bdata:
        return "", ""

    markets = bdata.get("markets", {})

    if isinstance(markets, list):
        h2h_market = next((m for m in markets if m["key"] == "h2h"), None)
    else:
        h2h_market = markets.get("h2h")

    ml_cell = ""
    if h2h_market:
        outcomes = {o["name"]: o["price"] for o in h2h_market.get("outcomes", [])}
        home_name = next((n for n in outcomes if NAME_TO_CODE.get(n) == home), None)
        away_name = next((n for n in outcomes if NAME_TO_CODE.get(n) == away), None)
        if home_name and away_name:
            mkt_home_p, mkt_away_p = no_vig_prob(outcomes[home_name], outcomes[away_name])
            home_edge = edge_pct(model_home_p, mkt_home_p)
            away_edge = edge_pct(model_away_p, mkt_away_p)
            if abs(home_edge) >= abs(away_edge):
                best_edge = home_edge
                best_odds = outcomes[home_name]
            else:
                best_edge = away_edge
                best_odds = outcomes[away_name]
            sign = fmt_american(best_odds)
            if abs(best_edge) >= EDGE_THRESHOLD:
                ml_cell = (
                    f'<span style="background:#1a7a3a;padding:2px 6px;'
                    f'border-radius:3px;white-space:nowrap">'
                    f'{sign}<br><small>+{abs(best_edge):.1f}%</small></span>'
                )
            else:
                ml_cell = sign

    return ml_cell, ""  # second value kept for unpacking compatibility

# ── Render one game block ─────────────────────────────────────────────────────
def render_game(game_num, row, odds_lookup):
    away    = str(row.get("away_team", "")).strip()
    home    = str(row.get("home_team", "")).strip()
    away_sp = str(row.get("away_sp",   "")).strip()
    home_sp = str(row.get("home_sp",   "")).strip()

    model_home_p = float(row.get("winprob_home", 0.5))
    model_away_p = 1 - model_home_p
    ml_fv_away   = fmt_american(prob_to_american(model_away_p))
    ml_fv_home   = fmt_american(prob_to_american(model_home_p))

    lambda_away = float(row.get("lambda_away", 0))
    lambda_home = float(row.get("lambda_home", 0))
    total_proj  = lambda_away + lambda_home
    tot_fv      = round(total_proj, 1)

    key = f"{away}@{home}"
    bk  = odds_lookup.get(key, {}).get("bookmakers", {})

    # build per-book cells
    book_data = {}
    for bk_key in BOOKS:
        book_data[bk_key] = get_book_cells(
            bk_key, bk, away, home, model_away_p, model_home_p, total_proj
        )

    # column proportions: #, team/pitcher, win%, ml fv, 5 books (ml only)
    col_spec = [0.25, 0.7, 0.4, 0.4] + [0.55] * len(BOOKS)

    # ── Column headers (only on first game) ───────────────────────────────────
    if game_num == 1:
        hdr = st.columns(col_spec)
        headers = ["#", "TEAM / PITCHER", "WIN %", "ML FV"] + \
                  [BOOK_LABELS[b] for b in BOOKS]
        for col, h in zip(hdr, headers):
            col.markdown(f"<b style='font-size:0.8em;color:#aaa'>{h}</b>",
                         unsafe_allow_html=True)
        st.markdown("<hr style='margin:2px 0 6px 0'>", unsafe_allow_html=True)

    # ── Away row ──────────────────────────────────────────────────────────────
    row_away = st.columns(col_spec)
    row_away[0].markdown(f"**{game_num}**")
    row_away[1].markdown(
        f"**{away}**<br><span style='color:gray;font-size:0.8em'>{away_sp}</span>",
        unsafe_allow_html=True)
    row_away[2].write(f"{model_away_p*100:.1f}%")
    row_away[3].write(ml_fv_away)
    for i, bk_key in enumerate(BOOKS):
        ml_cell, _ = book_data[bk_key]
        row_away[4 + i].markdown(ml_cell if ml_cell else "—", unsafe_allow_html=True)

    # ── Home row ──────────────────────────────────────────────────────────────
    row_home = st.columns(col_spec)
    row_home[0].write("")
    row_home[1].markdown(
        f"**{home}**<br><span style='color:gray;font-size:0.8em'>{home_sp}</span>",
        unsafe_allow_html=True)
    row_home[2].write(f"{model_home_p*100:.1f}%")
    row_home[3].write(ml_fv_home)
    for i, bk_key in enumerate(BOOKS):
        _, tot_cell = book_data[bk_key]
        row_home[4 + i].markdown(tot_cell if tot_cell else "—", unsafe_allow_html=True)

    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

# ── Performance tab ───────────────────────────────────────────────────────────
def render_performance(hist: pd.DataFrame):
    st.subheader("📊 Model Performance")

    scored = hist[hist["completed5"] == 1].copy() if not hist.empty else pd.DataFrame()
    if scored.empty:
        st.info("No scored games yet.")
        return

    scored = scored.sort_values("date")
    total   = len(scored)
    mae     = scored["mae_total"].mean()
    brier   = scored["brier"].mean()
    logloss = scored["logloss"].mean()
    scored["model_pick"] = (scored["winprob_home"] >= 0.5).astype(int)
    scored["actual_win"] = (scored["home_win_f5"] == 1.0).astype(int)
    acc = (scored["model_pick"] == scored["actual_win"]).mean() * 100

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games Scored",  total)
    c2.metric("MAE (Runs)",    f"{mae:.3f}")
    c3.metric("Brier Score",   f"{brier:.4f}", help="Lower is better. Coin-flip baseline = 0.25")
    c4.metric("Log Loss",      f"{logloss:.4f}", help="Lower is better.")
    c5.metric("Direction Acc", f"{acc:.1f}%")

    st.markdown("---")

    # ── Rolling metric charts ─────────────────────────────────────────────────
    st.markdown("#### 📈 Metrics Over Time")

    scored["game_num"] = range(1, total + 1)
    scored["roll_mae"]     = scored["mae_total"].expanding().mean()
    scored["roll_brier"]   = scored["brier"].expanding().mean()
    scored["roll_logloss"] = scored["logloss"].expanding().mean()

    col_a, col_b = st.columns(2)

    with col_a:
        mae_base = alt.Chart(scored).encode(
            x=alt.X("game_num:Q", title="Game #")
        )
        mae_bars = mae_base.mark_bar(color="#378AD3", opacity=0.4).encode(
            y=alt.Y("mae_total:Q", title="MAE"),
            tooltip=["date", "away_team", "home_team",
                     alt.Tooltip("mae_total:Q", format=".2f", title="MAE")]
        )
        mae_line = mae_base.mark_line(color="#1a7a3a", size=2).encode(
            y=alt.Y("roll_mae:Q", title="MAE")
        )
        st.altair_chart(
            (mae_bars + mae_line).properties(title="MAE — Runs Through 5", height=280),
            use_container_width=True
        )

    with col_b:
        bll_df = scored[["game_num", "date", "away_team", "home_team",
                          "roll_brier", "roll_logloss"]].melt(
            id_vars=["game_num", "date", "away_team", "home_team"],
            value_vars=["roll_brier", "roll_logloss"],
            var_name="metric", value_name="value"
        )
        bll_df["metric"] = bll_df["metric"].map(
            {"roll_brier": "Brier", "roll_logloss": "LogLoss"}
        )
        bll_line = alt.Chart(bll_df).mark_line(point=True).encode(
            x=alt.X("game_num:Q", title="Game #"),
            y=alt.Y("value:Q", title="Score"),
            color=alt.Color("metric:N", scale=alt.Scale(
                domain=["Brier", "LogLoss"],
                range=["#1a7a3a", "#d4730a"]
            )),
            strokeDash=alt.condition(
                alt.datum.metric == "LogLoss",
                alt.value([4, 4]),
                alt.value([0])
            ),
            tooltip=["metric", alt.Tooltip("value:Q", format=".4f")]
        )
        baseline = alt.Chart(pd.DataFrame({"y": [0.25]})).mark_rule(
            strokeDash=[4, 4], color="gray"
        ).encode(y="y:Q")
        st.altair_chart(
            (bll_line + baseline).properties(title="Brier Score & Log Loss (cumulative avg)", height=280),
            use_container_width=True
        )

    st.markdown("---")

    # ── Calibration chart ─────────────────────────────────────────────────────
    st.markdown("#### 🎯 Win Probability Calibration")
    st.caption("Dots should sit on the diagonal. Above = model underconfident (good). Below = overconfident.")

    bins = np.arange(0.30, 0.80, 0.10)
    cal_rows = []
    for b in bins:
        mask = (scored["winprob_home"] >= b) & (scored["winprob_home"] < b + 0.10)
        subset = scored[mask]
        if len(subset) > 0:
            cal_rows.append({
                "pred":   subset["winprob_home"].mean(),
                "actual": subset["actual_win"].mean(),
                "n":      len(subset),
                "label":  f"{b:.0%}–{b+0.10:.0%}"
            })

    if cal_rows:
        cal_df = pd.DataFrame(cal_rows)
        diagonal = pd.DataFrame({"x": [0.30, 0.75], "y": [0.30, 0.75]})

        diag_line = alt.Chart(diagonal).mark_line(
            strokeDash=[4, 4], color="gray"
        ).encode(
            x=alt.X("x:Q", title="Predicted win prob",
                    scale=alt.Scale(domain=[0.25, 0.80]),
                    axis=alt.Axis(format=".0%")),
            y=alt.Y("y:Q", title="Actual win rate",
                    scale=alt.Scale(domain=[-0.05, 1.05]),
                    axis=alt.Axis(format=".0%"))
        )

        dots = alt.Chart(cal_df).mark_circle(
            color="#378AD3", opacity=0.7, stroke="#185FA5", strokeWidth=1.5
        ).encode(
            x=alt.X("pred:Q", scale=alt.Scale(domain=[0.25, 0.80]),
                    axis=alt.Axis(format=".0%"), title="Predicted win prob"),
            y=alt.Y("actual:Q", scale=alt.Scale(domain=[-0.05, 1.05]),
                    axis=alt.Axis(format=".0%"), title="Actual win rate"),
            size=alt.Size("n:Q", scale=alt.Scale(range=[80, 400]), legend=None),
            tooltip=[
                alt.Tooltip("label:N", title="Bucket"),
                alt.Tooltip("pred:Q",   format=".3f", title="Predicted"),
                alt.Tooltip("actual:Q", format=".3f", title="Actual"),
                alt.Tooltip("n:Q",      title="Games"),
            ]
        )

        labels = dots.mark_text(dy=-14, fontSize=11, color="gray").encode(
            text=alt.Text("n:Q", format="d")
        )

        st.altair_chart(
            (diag_line + dots + labels).properties(height=360),
            use_container_width=True
        )
    else:
        st.info("Not enough scored games for calibration buckets yet.")

    st.markdown("---")

    # ── Archive table ─────────────────────────────────────────────────────────
    st.markdown("#### 📋 Forecast Archive")
    display_cols = ["date", "away_team", "away_sp", "home_team", "home_sp",
                    "total_proj", "winprob_home", "runs_away_thru5",
                    "runs_home_thru5", "total_actual", "mae_total", "brier", "logloss"]
    available = [c for c in display_cols if c in scored.columns]
    st.dataframe(scored[available].sort_values("date", ascending=False),
                 use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        "<h1 style='color:#1a7a3a'>⚾ ClutchSheets F5 Baseball Forecaster</h1>",
        unsafe_allow_html=True
    )

    CSV_PATH = os.environ.get("FORECAST_CSV", "daily_forecast.csv")
    DB_PATH  = os.environ.get("DB_PATH",      "mlb_f5_forecasts.db")

    tab1, tab2 = st.tabs(["📅 Today's Games", "📊 Performance"])

    with tab1:
        df = load_forecast(CSV_PATH)
        if df.empty:
            st.warning("No forecast data found.")
            return

        today_str = date.today().strftime("%Y-%m-%d")
        if "date" in df.columns:
            df["date"] = df["date"].astype(str).str.strip()
            df = df[df["date"] == today_str]

        if df.empty:
            st.warning(f"No games found for {today_str}.")
            return

        odds_lookup = fetch_odds()
        if not odds_lookup:
            st.warning("⚠️ No odds data — check ODDS_API_KEY or API quota.")

        for game_num, (_, row) in enumerate(df.iterrows(), start=1):
            render_game(game_num, row, odds_lookup)

    with tab2:
        hist = load_history(DB_PATH)
        render_performance(hist)


if __name__ == "__main__":
    main()
