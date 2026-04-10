import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sqlite3
import os
import requests
import numpy as np
import altair as alt
from datetime import date

st.set_page_config(page_title="ClutchSheets F5 Forecaster", layout="wide")

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

BOOKS = ["betonline", "fanduel", "draftkings", "betmgm", "betrivers"]
BOOK_LABELS = {
    "betonline": "BetOnline",
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "betmgm": "BetMGM",
    "betrivers": "BetRivers",
}

EDGE_THRESHOLD = 5.0


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


def fmt_american(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


NAME_TO_CODE = {
    "Pittsburgh Pirates": "PIT", "Chicago Cubs": "CHC",
    "Oakland Athletics": "ATH", "Athletics": "ATH", "New York Mets": "NYM",
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


def get_book_odds(book_key, bk, away, home):
    bdata = bk.get(book_key, {})
    if not bdata:
        return None, None
    markets = bdata.get("markets", {})
    h2h = next((m for m in markets if m["key"] == "h2h"), None) if isinstance(markets, list) else markets.get("h2h")
    if not h2h:
        return None, None
    outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
    home_name = next((n for n in outcomes if NAME_TO_CODE.get(n) == home), None)
    away_name = next((n for n in outcomes if NAME_TO_CODE.get(n) == away), None)
    if home_name and away_name:
        return outcomes[away_name], outcomes[home_name]
    return None, None


def render_games_table(df, odds_lookup):
    GREEN = "#1a7a3a"
    BG_DARK = "#0e1117"
    BG_ROW = "#1a1d24"
    BG_ALT = "#14171e"
    BORDER = "#2a2d35"
    SUBTEXT = "#888"

    book_headers = "".join(
        f"<th style='padding:4px 12px;text-align:center;color:{SUBTEXT};font-size:0.75em;letter-spacing:0.05em;border-bottom:1px solid {BORDER}'>{BOOK_LABELS[b]}</th>"
        for b in BOOKS
    )

    css = f"""
      <style>
        body{{margin:0;padding:0;background:transparent}}
        table{{width:100%;border-collapse:collapse;font-family:sans-serif;background:{BG_DARK};border-radius:8px;overflow:hidden}}
        th{{background:#111418;color:#ffffff;font-size:0.72em;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;padding:10px 12px;border-bottom:2px solid #3a4050}}
        tr.game-odd td{{background:{BG_ROW}}}
        tr.game-even td{{background:{BG_ALT}}}
        tr.game-odd:hover td,tr.game-even:hover td{{filter:brightness(1.2)}}
        .edge{{background:{GREEN};border-radius:3px;padding:1px 6px;white-space:nowrap;display:inline-block;font-weight:600;color:#fff}}
        .edge small{{display:block;font-size:0.75em;font-weight:400;opacity:0.9}}
        td{{padding:4px 12px;border-bottom:1px solid {BORDER}}}
      </style>
    """

    TH = "padding:10px 12px;font-size:0.72em;font-weight:700;color:#fff;letter-spacing:0.08em;text-transform:uppercase;background:#111418;border-bottom:2px solid #3a4050"
    html = css + f"""<table>
      <thead>
        <tr>
          <th style='{TH};text-align:center;width:30px'>#</th>
          <th style='{TH};text-align:left'>TEAM / PITCHER</th>
          <th style='{TH};text-align:center'>WIN %</th>
          <th style='{TH};text-align:center'>ML FV</th>
          {book_headers}
        </tr>
      </thead>
      <tbody>
    """

    for game_num, (_, row) in enumerate(df.iterrows(), start=1):
        away = str(row.get("away_team", "")).strip()
        home = str(row.get("home_team", "")).strip()
        away_sp = str(row.get("away_sp", "")).strip()
        home_sp = str(row.get("home_sp", "")).strip()

        model_home_p = float(row.get("winprob_home", 0.5))
        model_away_p = 1 - model_home_p
        ml_fv_away = fmt_american(prob_to_american(model_away_p))
        ml_fv_home = fmt_american(prob_to_american(model_home_p))

        key = f"{away}@{home}"
        bk = odds_lookup.get(key, {}).get("bookmakers", {})

        away_cells = ""
        home_cells = ""
        td_style = f"padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}"

        for bk_key in BOOKS:
            away_odds, home_odds = get_book_odds(bk_key, bk, away, home)
            if away_odds is None:
                away_cells += f"<td style='{td_style}'><span style='color:#888'>—</span></td>"
                home_cells += f"<td style='{td_style}'><span style='color:#888'>—</span></td>"
                continue

            mkt_away_p, mkt_home_p = no_vig_prob(away_odds, home_odds)
            away_edge = edge_pct(model_away_p, mkt_away_p)
            home_edge = edge_pct(model_home_p, mkt_home_p)

            away_sign = fmt_american(away_odds)
            if away_edge >= EDGE_THRESHOLD:
                away_cells += f"<td style='{td_style}'><span class='edge'>{away_sign}<small>+{away_edge:.1f}%</small></span></td>"
            else:
                away_cells += f"<td style='{td_style}'><span style='color:#e0e0e0;font-size:0.9em'>{away_sign}</span></td>"

            home_sign = fmt_american(home_odds)
            if home_edge >= EDGE_THRESHOLD:
                home_cells += f"<td style='{td_style}'><span class='edge'>{home_sign}<small>+{home_edge:.1f}%</small></span></td>"
            else:
                home_cells += f"<td style='{td_style}'><span style='color:#e0e0e0;font-size:0.9em'>{home_sign}</span></td>"

        game_class = 'game-odd' if game_num % 2 == 1 else 'game-even'
        html += f"""
        <tr class='{game_class}'>
          <td style='padding:4px 6px;text-align:center;border-bottom:1px solid {BORDER};vertical-align:bottom' rowspan='2'>
            <span style='color:#888;font-size:0.8em'>{game_num}</span>
          </td>
          <td style='padding:4px 12px;border-bottom:1px solid {BORDER}'>
            <span style='font-weight:600;color:#e0e0e0'>{away}</span><br>
            <span style='color:#888;font-size:0.78em'>{away_sp}</span>
          </td>
          <td style='padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}'>
            <span style='color:#e0e0e0;font-size:0.9em'>{model_away_p*100:.1f}%</span>
          </td>
          <td style='padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}'>
            <span style='color:#7ec8e3;font-size:0.9em;font-weight:600'>{ml_fv_away}</span>
          </td>
          {away_cells}
        </tr>
        <tr class='{game_class}'>
          <td style='padding:4px 12px;border-bottom:2px solid {BORDER}'>
            <span style='font-weight:600;color:#e0e0e0'>{home}</span><br>
            <span style='color:#888;font-size:0.78em'>{home_sp}</span>
          </td>
          <td style='padding:4px 12px;text-align:center;border-bottom:2px solid {BORDER}'>
            <span style='color:#e0e0e0;font-size:0.9em'>{model_home_p*100:.1f}%</span>
          </td>
          <td style='padding:4px 12px;text-align:center;border-bottom:2px solid {BORDER}'>
            <span style='color:#7ec8e3;font-size:0.9em;font-weight:600'>{ml_fv_home}</span>
          </td>
          {home_cells}
        </tr>
        """

    html += "</tbody></table>"
    height = len(df) * 64 + 50
    components.html(html, height=height, scrolling=False)


def render_performance(hist: pd.DataFrame):
    st.subheader("Pricing Performance")

    if hist.empty or "completed5" not in hist.columns:
        st.info("No scored games yet.")
        return

    scored = hist[hist["completed5"] == 1].copy()
    if scored.empty:
        st.info("No scored games yet.")
        return

    scored = scored.sort_values(["date", "game_pk" if "game_pk" in scored.columns else "date"]).reset_index(drop=True)
    scored["game_num"] = range(1, len(scored) + 1)

    for col in [
        "winprob_home", "close_home_prob_devig", "model_vs_close_logloss_devig",
        "close_logloss_devig", "abs_diff_close_devig_home", "logloss", "brier", "mae_total"
    ]:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")

    close_ready = scored.dropna(subset=["close_home_prob_devig"]).copy() if "close_home_prob_devig" in scored.columns else pd.DataFrame()
    ll_ready = scored.dropna(subset=["model_vs_close_logloss_devig", "close_logloss_devig", "logloss"]).copy() if "model_vs_close_logloss_devig" in scored.columns else pd.DataFrame()

    total_scored = len(scored)
    total_with_close = len(close_ready)
    avg_abs_devig = close_ready["abs_diff_close_devig_home"].mean() if "abs_diff_close_devig_home" in close_ready.columns and not close_ready.empty else np.nan
    model_logloss = ll_ready["logloss"].mean() if not ll_ready.empty else np.nan
    close_logloss = ll_ready["close_logloss_devig"].mean() if not ll_ready.empty else np.nan
    delta_vs_close = ll_ready["model_vs_close_logloss_devig"].mean() if not ll_ready.empty else np.nan
    beat_close_rate = (ll_ready["model_vs_close_logloss_devig"] < 0).mean() * 100 if not ll_ready.empty else np.nan

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Games Scored", total_scored)
    c2.metric("Games With Close", total_with_close)
    c3.metric("Avg Abs Devig Diff", "—" if pd.isna(avg_abs_devig) else f"{avg_abs_devig:.4f}")
    c4.metric("Model Log Loss", "—" if pd.isna(model_logloss) else f"{model_logloss:.4f}")
    c5.metric("Pinnacle Close Log Loss", "—" if pd.isna(close_logloss) else f"{close_logloss:.4f}")
    c6.metric("Model - Close", "—" if pd.isna(delta_vs_close) else f"{delta_vs_close:+.4f}")

    st.caption("Avg Abs Devig Diff measures how far your home win probability was from the no-vig Pinnacle close. Model - Close below zero means the model beat the closing market on log loss.")

    c7, c8 = st.columns(2)
    with c7:
        st.metric("Beat Close Rate", "—" if pd.isna(beat_close_rate) else f"{beat_close_rate:.1f}%")
    with c8:
        coverage = (total_with_close / total_scored * 100) if total_scored else np.nan
        st.metric("Close Coverage", "—" if pd.isna(coverage) else f"{coverage:.1f}%")

    st.markdown("---")

    st.markdown("#### Metrics Over Time")

    if not ll_ready.empty:
        ll_ready = ll_ready.sort_values(["date", "game_num"]).copy()
        ll_ready["roll_model_logloss"] = ll_ready["logloss"].expanding().mean()
        ll_ready["roll_close_logloss"] = ll_ready["close_logloss_devig"].expanding().mean()
        ll_ready["roll_delta_vs_close"] = ll_ready["model_vs_close_logloss_devig"].expanding().mean()

        chart_df = ll_ready[["game_num", "roll_model_logloss", "roll_close_logloss"]].melt(
            id_vars=["game_num"],
            var_name="metric",
            value_name="value"
        )
        chart_df["metric"] = chart_df["metric"].map({
            "roll_model_logloss": "Model",
            "roll_close_logloss": "Pinnacle Close",
        })

        line_chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x=alt.X("game_num:Q", title="Game #"),
            y=alt.Y("value:Q", title="Cumulative Avg Log Loss"),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["Model", "Pinnacle Close"], range=["#1f77b4", "#d4730a"])),
            tooltip=["metric", alt.Tooltip("value:Q", format=".4f")]
        ).properties(height=300, title="Model vs Pinnacle Close Log Loss")

        delta_chart = alt.Chart(ll_ready).mark_line(point=True, color="#7a3db8").encode(
            x=alt.X("game_num:Q", title="Game #"),
            y=alt.Y("roll_delta_vs_close:Q", title="Cumulative Avg Delta"),
            tooltip=[alt.Tooltip("roll_delta_vs_close:Q", format="+.4f")]
        ).properties(height=300, title="Model - Pinnacle Close Log Loss")

        zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[5, 5], color="gray").encode(y="y:Q")

        col_a, col_b = st.columns(2)
        with col_a:
            st.altair_chart(line_chart, use_container_width=True)
        with col_b:
            st.altair_chart(delta_chart + zero_rule, use_container_width=True)
    else:
        st.info("Closing-line log-loss metrics will appear after grade_closing.py backfills the DB.")

    st.markdown("---")

    if not close_ready.empty and "abs_diff_close_devig_home" in close_ready.columns:
        close_ready = close_ready.sort_values(["date", "game_num"]).copy()
        close_ready["roll_abs_devig_diff"] = close_ready["abs_diff_close_devig_home"].expanding().mean()

        abs_chart = alt.Chart(close_ready).mark_bar(color="#378AD3", opacity=0.35).encode(
            x=alt.X("game_num:Q", title="Game #"),
            y=alt.Y("abs_diff_close_devig_home:Q", title="Abs Diff"),
            tooltip=["date", "away_team", "home_team", alt.Tooltip("abs_diff_close_devig_home:Q", format=".4f", title="Abs Devig Diff")]
        )
        abs_line = alt.Chart(close_ready).mark_line(color="#1a7a3a", size=2).encode(
            x="game_num:Q",
            y=alt.Y("roll_abs_devig_diff:Q", title="Abs Diff")
        )
        st.markdown("#### Distance From Pinnacle Close")
        st.altair_chart((abs_chart + abs_line).properties(height=300, title="Home Win Probability vs Devigged Pinnacle Close"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Archive")

    archive_cols = [
        "date", "away_team", "home_team", "winprob_home",
        "close_away_odds", "close_home_odds",
        "close_home_prob_devig", "abs_diff_close_devig_home",
        "logloss", "close_logloss_devig", "model_vs_close_logloss_devig"
    ]
    available = [c for c in archive_cols if c in scored.columns]
    if available:
        st.dataframe(scored[available].sort_values("date", ascending=False), use_container_width=True, hide_index=True)

    with st.expander("Result-based diagnostics"):
        result_cols = [c for c in ["date", "away_team", "home_team", "mae_total", "brier", "logloss", "runs_away_thru5", "runs_home_thru5", "total_actual"] if c in scored.columns]
        if result_cols:
            st.dataframe(scored[result_cols].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No result-based diagnostics available.")


def main():
    st.markdown(
        """<style>
        header[data-testid="stHeader"]{display:none!important}
        #MainMenu{display:none!important}
        footer{display:none!important}
        .block-container{padding-top:0.5rem!important;padding-bottom:0!important}
        h2{margin-top:0!important}
        </style>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='color:#1a7a3a;margin:0 0 6px 0'>⚾ ClutchSheets F5 Baseball Forecaster</h2>",
        unsafe_allow_html=True,
    )

    CSV_PATH = os.environ.get("FORECAST_CSV", "daily_forecast.csv")
    DB_PATH = os.environ.get("DB_PATH", "mlb_f5_forecasts.db")

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
            st.warning("No odds data. Check ODDS_API_KEY or API quota.")

        render_games_table(df, odds_lookup)

    with tab2:
        hist = load_history(DB_PATH)
        render_performance(hist)


if __name__ == "__main__":
    main()
