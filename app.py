import os
import sqlite3
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ClutchSheets F5 Forecaster", layout="wide")

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "d31971f20a2edb069b228e828826815a")
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
    try:
        r = requests.get(ODDS_API_URL, params=params, timeout=15)
        if r.status_code != 200:
            return {}
        raw = r.json()
    except Exception:
        return {}

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
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = parsed.dt.strftime("%Y-%m-%d")
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

        game_class = "game-odd" if game_num % 2 == 1 else "game-even"
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


def render_calibration(scored: pd.DataFrame):
    st.markdown("---")
    st.markdown("#### Calibration (10% Buckets)")
    st.caption("This checks how raw model probabilities tracked actual outcomes across all modeled games. It is not restricted to bet-qualified games.")

    calib = scored.dropna(subset=["winprob_home", "home_win_f5"]).copy()
    if calib.empty:
        st.info("Not enough data for calibration.")
        return

    bins = np.linspace(0, 1, 11)
    calib["bucket"] = pd.cut(calib["winprob_home"], bins=bins, include_lowest=True)

    calib_table = (
        calib.groupby("bucket", observed=False)
        .agg(
            avg_pred=("winprob_home", "mean"),
            actual_win=("home_win_f5", "mean"),
            count=("home_win_f5", "count"),
        )
        .reset_index()
    )

    calib_table = calib_table[calib_table["count"] > 0].copy()
    if calib_table.empty:
        st.info("Not enough populated buckets for calibration.")
        return

    diagonal = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    diag_line = alt.Chart(diagonal).mark_line(
        strokeDash=[4, 4], color="gray"
    ).encode(
        x=alt.X("x:Q", title="Predicted win probability", axis=alt.Axis(format=".0%")),
        y=alt.Y("y:Q", title="Actual win rate", axis=alt.Axis(format=".0%")),
    )

    dots = alt.Chart(calib_table).mark_circle(
        color="#378AD3", opacity=0.8, stroke="#185FA5", strokeWidth=1.5
    ).encode(
        x=alt.X("avg_pred:Q", axis=alt.Axis(format=".0%")),
        y=alt.Y("actual_win:Q", axis=alt.Axis(format=".0%")),
        size=alt.Size("count:Q", scale=alt.Scale(range=[80, 450]), legend=None),
        tooltip=[
            alt.Tooltip("bucket:N", title="Bucket"),
            alt.Tooltip("avg_pred:Q", format=".3f", title="Avg predicted"),
            alt.Tooltip("actual_win:Q", format=".3f", title="Actual win rate"),
            alt.Tooltip("count:Q", title="Games"),
        ],
    )

    st.altair_chart((diag_line + dots).properties(height=340), use_container_width=True)

    show_table = calib_table.copy()
    show_table["avg_pred"] = show_table["avg_pred"].round(4)
    show_table["actual_win"] = show_table["actual_win"].round(4)
    st.dataframe(show_table, use_container_width=True)


def render_performance(hist: pd.DataFrame):
    st.subheader("Pricing Performance")
    st.caption("These diagnostics cover all modeled games with scored results and closing-line data. They are not limited to games you would have actually bet.")

    if hist.empty or "completed5" not in hist.columns:
        st.info("No scored games yet.")
        return

    hist = hist.copy()
    hist["completed5"] = pd.to_numeric(hist["completed5"], errors="coerce").fillna(0)
    scored = hist[hist["completed5"] == 1].copy()
    if scored.empty:
        st.info("No scored games yet.")
        return

    # Remove invalid model outputs from evaluation
    if "winprob_home" in scored.columns:
        scored["winprob_home"] = pd.to_numeric(scored["winprob_home"], errors="coerce")
        scored = scored[(scored["winprob_home"] > 0.001) & (scored["winprob_home"] < 0.999)].copy()

    if scored.empty:
        st.info("No valid scored games yet.")
        return

    for col in [
        "winprob_home", "home_win_f5", "logloss", "brier", "mae_total",
        "close_logloss_devig", "model_vs_close_logloss_devig", "home_abs_devig_close_diff"
    ]:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")

    scored = scored.sort_values("date")
    total = len(scored)

    close_coverage = (
        scored["close_logloss_devig"].notna().mean() * 100
        if "close_logloss_devig" in scored.columns else 0.0
    )
    model_logloss = scored["logloss"].mean() if "logloss" in scored.columns else np.nan
    close_logloss = scored["close_logloss_devig"].mean() if "close_logloss_devig" in scored.columns else np.nan
    delta_vs_close = scored["model_vs_close_logloss_devig"].mean() if "model_vs_close_logloss_devig" in scored.columns else np.nan
    beat_close_rate = (
        (scored["model_vs_close_logloss_devig"] < 0).mean() * 100
        if "model_vs_close_logloss_devig" in scored.columns else np.nan
    )
    avg_abs_devig = scored["home_abs_devig_close_diff"].mean() if "home_abs_devig_close_diff" in scored.columns else np.nan

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Games Scored", total)
    c2.metric("Close Coverage", f"{close_coverage:.1f}%")
    c3.metric("Avg Abs Devig Diff", f"{avg_abs_devig:.4f}" if pd.notna(avg_abs_devig) else "—")
    c4.metric("Model Log Loss", f"{model_logloss:.4f}" if pd.notna(model_logloss) else "—")
    c5.metric("Close Log Loss", f"{close_logloss:.4f}" if pd.notna(close_logloss) else "—")
    c6.metric("Model - Close", f"{delta_vs_close:+.4f}" if pd.notna(delta_vs_close) else "—")

    if pd.notna(beat_close_rate):
        st.caption(f"Beat Close Rate: {beat_close_rate:.1f}%")

    st.markdown("---")
    st.markdown("#### Pricing Metrics Over Time")

    scored["game_num"] = range(1, total + 1)

    chart_cols = []
    if "model_vs_close_logloss_devig" in scored.columns:
        scored["roll_delta_vs_close"] = scored["model_vs_close_logloss_devig"].expanding().mean()
        chart_cols.append("roll_delta_vs_close")
    if "home_abs_devig_close_diff" in scored.columns:
        scored["roll_abs_devig_diff"] = scored["home_abs_devig_close_diff"].expanding().mean()
        chart_cols.append("roll_abs_devig_diff")

    if chart_cols:
        plot_df = scored[["game_num", "date", "away_team", "home_team"] + chart_cols].melt(
            id_vars=["game_num", "date", "away_team", "home_team"],
            value_vars=chart_cols,
            var_name="metric",
            value_name="value"
        )
        plot_df["metric"] = plot_df["metric"].map({
            "roll_delta_vs_close": "Model - Close",
            "roll_abs_devig_diff": "Abs Devig Diff",
        })

        line = alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X("game_num:Q", title="Game #"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color("metric:N"),
            tooltip=["metric", alt.Tooltip("value:Q", format=".4f"), "date", "away_team", "home_team"]
        )

        rules = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((line + rules).properties(height=320), use_container_width=True)

    render_calibration(scored)

    with st.expander("Outcome-based diagnostics"):
        st.caption("These are still useful for model quality, but pricing diagnostics above are the primary read for betting purposes.")
        mae = scored["mae_total"].mean() if "mae_total" in scored.columns else np.nan
        brier = scored["brier"].mean() if "brier" in scored.columns else np.nan

        scored["model_pick"] = (scored["winprob_home"] >= 0.5).astype(int)
        scored["actual_win"] = (scored["home_win_f5"] == 1.0).astype(int)
        acc = (scored["model_pick"] == scored["actual_win"]).mean() * 100

        d1, d2, d3 = st.columns(3)
        d1.metric("MAE (Runs)", f"{mae:.3f}" if pd.notna(mae) else "—")
        d2.metric("Brier Score", f"{brier:.4f}" if pd.notna(brier) else "—")
        d3.metric("Direction Acc", f"{acc:.1f}%")

    st.markdown("---")
    st.markdown("#### Forecast Archive")
    display_cols = [
        "date", "away_team", "away_sp", "home_team", "home_sp",
        "winprob_home", "close_home_prob_devig", "home_abs_devig_close_diff",
        "logloss", "close_logloss_devig", "model_vs_close_logloss_devig"
    ]
    available = [c for c in display_cols if c in scored.columns]
    st.dataframe(scored[available].sort_values("date", ascending=False), use_container_width=True)


def main():
    st.markdown(
        """<style>
        header[data-testid="stHeader"]{display:none!important}
        #MainMenu{display:none!important}
        footer{display:none!important}
        .block-container{padding-top:0.5rem!important;padding-bottom:0!important}
        h2{margin-top:0!important}
        </style>""",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h2 style='color:#1a7a3a;margin:0 0 6px 0'>⚾ ClutchSheets F5 Baseball Forecaster</h2>",
        unsafe_allow_html=True
    )

    CSV_PATH = os.environ.get("FORECAST_CSV", "daily_forecast.csv")
    DB_PATH = os.environ.get("DB_PATH", "mlb_f5_forecasts.db")

    tab1, tab2 = st.tabs(["📅 Today's Games", "📊 Performance"])

    with tab1:
        df = load_forecast(CSV_PATH)
        if df.empty:
            st.warning("No forecast data found.")
        else:
            today_str = date.today().strftime("%Y-%m-%d")
            slate_df = df.copy()

            if "date" in slate_df.columns:
                exact = slate_df[slate_df["date"] == today_str].copy()
                if exact.empty:
                    latest_date = slate_df["date"].dropna().max()
                    if pd.notna(latest_date):
                        st.warning(f"No games found for {today_str}. Showing latest available slate: {latest_date}.")
                        slate_df = slate_df[slate_df["date"] == latest_date].copy()
                    else:
                        st.warning(f"No games found for {today_str}.")
                        slate_df = pd.DataFrame()
                else:
                    slate_df = exact

            if not slate_df.empty:
                odds_lookup = fetch_odds()
                if not odds_lookup:
                    st.warning("⚠️ No odds data — check ODDS_API_KEY or API quota.")
                render_games_table(slate_df, odds_lookup)

    with tab2:
        hist = load_history(DB_PATH)
        render_performance(hist)


if __name__ == "__main__":
    main()
