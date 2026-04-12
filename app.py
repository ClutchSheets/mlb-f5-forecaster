import json
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
DAILY_ODDS_FILE = os.environ.get("DAILY_ODDS_FILE", "daily_odds.json")

BOOKS = ["betonlineag", "fanduel", "draftkings", "betmgm", "betrivers"]
BOOK_LABELS = {
    "betonlineag": "BetOnline",
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "betmgm": "BetMGM",
    "betrivers": "BetRivers",
}
KELLY_THRESHOLD = 1.5

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


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def adj_kelly(model_p: float, decimal_odds: float, scale: float = 10) -> float:
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    return ((b * model_p - (1 - model_p)) / b) * scale


def fmt_american(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


@st.cache_data(ttl=300)
def fetch_odds():
    """
    Reads pre-fetched F5 odds from daily_odds.json (written by fetch_odds.py).
    Returns lookup keyed by "AWAY_CODE@HOME_CODE" with structure:
      { "ml": {"fanduel": {"away": int, "home": int}, ...},
        "total": {...},
        "pinnacle": {"ml": {...}, "total": {...}} }
    """
    if not os.path.exists(DAILY_ODDS_FILE):
        return {}
    try:
        with open(DAILY_ODDS_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    lookup = {}
    for full_key, game in payload.get("games", {}).items():
        away_name = game.get("away", "")
        home_name = game.get("home", "")
        away_code = NAME_TO_CODE.get(away_name, away_name[:3].upper())
        home_code = NAME_TO_CODE.get(home_name, home_name[:3].upper())
        code_key = f"{away_code}@{home_code}"
        lookup[code_key] = game  # contains ml, total, pinnacle, commence_edt
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


def get_book_odds(book_key, game, away, home):
    """Reads F5 ML odds from daily_odds.json game dict."""
    entry = game.get("ml", {}).get(book_key)
    if not entry:
        return None, None
    return entry.get("away"), entry.get("home")


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
        game = odds_lookup.get(key, {})

        away_cells = ""
        home_cells = ""
        td_style = f"padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}"

        for bk_key in BOOKS:
            away_odds, home_odds = get_book_odds(bk_key, game, away, home)
            if away_odds is None:
                away_cells += f"<td style='{td_style}'><span style='color:#888'>—</span></td>"
                home_cells += f"<td style='{td_style}'><span style='color:#888'>—</span></td>"
                continue

            away_dec = american_to_decimal(away_odds)
            home_dec = american_to_decimal(home_odds)
            away_k = adj_kelly(model_away_p, away_dec)
            home_k = adj_kelly(model_home_p, home_dec)

            away_sign = fmt_american(away_odds)
            if away_k >= KELLY_THRESHOLD:
                away_cells += f"<td style='{td_style}'><span class='edge'>{away_sign}<small>K:{away_k:.1f}</small></span></td>"
            else:
                away_cells += f"<td style='{td_style}'><span style='color:#e0e0e0;font-size:0.9em'>{away_sign}</span></td>"

            home_sign = fmt_american(home_odds)
            if home_k >= KELLY_THRESHOLD:
                home_cells += f"<td style='{td_style}'><span class='edge'>{home_sign}<small>K:{home_k:.1f}</small></span></td>"
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
    # 2 rows per game × ~52px per row + 60px header, no internal scrolling
    height = len(df) * 104 + 60
    components.html(html, height=height, scrolling=False)


def prepare_scored_subset(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty or "completed5" not in hist.columns:
        return pd.DataFrame()

    scored = hist.copy()
    scored["completed5"] = pd.to_numeric(scored["completed5"], errors="coerce").fillna(0)
    scored = scored[scored["completed5"] == 1].copy()
    if scored.empty:
        return scored

    numeric_cols = [
        "winprob_home", "home_win_f5", "logloss", "brier", "mae_total",
        "close_logloss_devig", "model_vs_close_logloss_devig",
        "home_abs_devig_close_diff", "close_home_prob_devig"
    ]
    for col in numeric_cols:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")

    if "winprob_home" in scored.columns:
        scored = scored[(scored["winprob_home"] > 0.25) & (scored["winprob_home"] < 0.75)].copy()

    if "winprob_home" in scored.columns and "close_home_prob_devig" in scored.columns:
        scored["prob_diff_close"] = scored["winprob_home"] - scored["close_home_prob_devig"]
        scored["abs_prob_diff_close"] = scored["prob_diff_close"].abs()

        def fair_odds_from_prob(p):
            if pd.isna(p):
                return np.nan
            p = max(min(float(p), 0.9999), 0.0001)
            return prob_to_american(p)

        scored["model_fv_home"] = scored["winprob_home"].apply(fair_odds_from_prob)
        scored["close_fv_home"] = scored["close_home_prob_devig"].apply(fair_odds_from_prob)

    return scored.sort_values("date").copy()


def format_bucket_label(interval) -> str:
    left = int(round(interval.left * 100))
    right = int(round(interval.right * 100))
    if left < 0:
        left = 0
    return f"{left}%–{right}%"


def render_beating_the_market(df: pd.DataFrame):
    st.markdown("#### Model vs Market — Am I Getting Closer?")

    if "model_vs_close_logloss_devig" not in df.columns:
        st.info("No closing-line log loss data yet.")
        return

    plot_df = df.dropna(subset=["model_vs_close_logloss_devig"]).copy()
    if plot_df.empty:
        st.info("No closing-line log loss data yet.")
        return

    plot_df = plot_df.sort_values("date").reset_index(drop=True)
    plot_df["game_num"] = range(1, len(plot_df) + 1)
    plot_df["cumulative_median"] = plot_df["model_vs_close_logloss_devig"].expanding().median()
    plot_df["cumulative_mean"]   = plot_df["model_vs_close_logloss_devig"].expanding().mean()
    plot_df["game_label"] = plot_df["away_team"] + " @ " + plot_df["home_team"] + " (" + plot_df["date"] + ")"

    median_delta = plot_df["model_vs_close_logloss_devig"].median()
    mean_delta   = plot_df["model_vs_close_logloss_devig"].mean()
    outlier_drag = mean_delta - median_delta  # positive = outliers pulling mean up
    n            = len(plot_df)

    # Recent trend: compare last 50 games vs overall median
    window     = min(50, n)
    recent_df  = plot_df.tail(window)
    recent_avg = recent_df["model_vs_close_logloss_devig"].median()
    trend_diff = recent_avg - median_delta  # negative = recent is better than overall

    # Plain-English headline
    if median_delta < 0:
        overall_txt = f"Overall your model is <b>ahead of the market</b> by {abs(median_delta):.4f} median log loss per game across {n} games."
    else:
        overall_txt = f"Overall your model is <b>behind the market</b> by {abs(median_delta):.4f} median log loss per game across {n} games."

    if abs(trend_diff) < 0.005:
        trend_txt = f"Your last {window} games are flat — performance is holding steady."
        trend_color = "#666"
    elif trend_diff < 0:
        trend_txt = f"Your last {window} games are <b>trending better</b> (recent median {recent_avg:+.4f} vs overall {median_delta:+.4f}) — you\'re closing the gap on the market."
        trend_color = "#1a7a3a"
    else:
        trend_txt = f"Your last {window} games are <b>trending worse</b> (recent median {recent_avg:+.4f} vs overall {median_delta:+.4f}) — the gap to the market is widening."
        trend_color = "#a33333"

    st.markdown(
        f"<p style=\"margin:0 0 4px\">{overall_txt}</p>"
        f"<p style=\"margin:0;color:{trend_color}\">{trend_txt}</p>",
        unsafe_allow_html=True
    )
    st.caption("Log loss delta = model log loss minus Pinnacle closing log loss. Negative = your model priced the outcome better. Lower is better. Blue line = cumulative median (robust to outliers).")

    st.markdown("")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games Scored", n)
    c2.metric("Median Log Loss Delta", f"{median_delta:+.4f}", help="Typical game — robust to outliers")
    c3.metric("Mean Log Loss Delta",   f"{mean_delta:+.4f}",   help="Overall calibration — includes blowups")
    c4.metric("Outlier Drag",          f"{outlier_drag:+.4f}", help="Mean minus median. Positive = bad outlier games pulling mean up. Investigate these.")
    c5.metric(f"Last {window} Median", f"{recent_avg:+.4f}",
              delta=f"{trend_diff:+.4f} vs overall",
              delta_color="inverse")

    st.markdown("")

    # Shaded region below zero
    shade_df = pd.DataFrame({"x": [0, n + 1], "y1": [0, 0], "y2": [-1, -1]})
    shade = alt.Chart(shade_df).mark_rect(opacity=0.04, color="#1a7a3a").encode(
        x=alt.X("x:Q"), x2="x:Q",
        y=alt.Y("y1:Q"), y2="y2:Q"
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        strokeDash=[4, 4], color="gray", opacity=0.5
    ).encode(y="y:Q")

    # Recent window highlight band
    recent_band = alt.Chart(pd.DataFrame({
        "x1": [n - window + 1], "x2": [n + 0.5]
    })).mark_rect(opacity=0.06, color="#378AD3").encode(
        x=alt.X("x1:Q"), x2="x2:Q"
    )

    recent_label = alt.Chart(pd.DataFrame({
        "x": [n - window / 2], "y": [plot_df["model_vs_close_logloss_devig"].max() * 0.85],
        "label": [f"Last {window}"]
    })).mark_text(color="#378AD3", fontSize=11, opacity=0.7).encode(
        x="x:Q", y="y:Q", text="label:N"
    )

    trend_line_median = alt.Chart(plot_df).mark_line(color="#378AD3", strokeWidth=2.5).encode(
        x=alt.X("game_num:Q", title="Game #"),
        y=alt.Y("cumulative_median:Q",
                title="Log loss delta (negative = beating market)",
                axis=alt.Axis(format=".3f")),
        tooltip=[
            alt.Tooltip("game_label:N",        title="Game"),
            alt.Tooltip("cumulative_median:Q", title="Cumulative median", format="+.4f"),
            alt.Tooltip("cumulative_mean:Q",   title="Cumulative mean",   format="+.4f"),
            alt.Tooltip("model_vs_close_logloss_devig:Q", title="This game", format="+.4f"),
            alt.Tooltip("game_num:Q",          title="Game #"),
        ]
    )

    trend_line_mean = alt.Chart(plot_df).mark_line(
        color="#f0a500", strokeWidth=1.5, strokeDash=[4, 3], opacity=0.7
    ).encode(
        x=alt.X("game_num:Q"),
        y=alt.Y("cumulative_mean:Q"),
        tooltip=[
            alt.Tooltip("game_label:N",        title="Game"),
            alt.Tooltip("cumulative_mean:Q",   title="Cumulative mean",   format="+.4f"),
            alt.Tooltip("cumulative_median:Q", title="Cumulative median", format="+.4f"),
            alt.Tooltip("game_num:Q",          title="Game #"),
        ]
    )

    points = alt.Chart(plot_df).mark_circle(size=35, opacity=0.55).encode(
        x=alt.X("game_num:Q"),
        y=alt.Y("model_vs_close_logloss_devig:Q"),
        color=alt.condition(
            alt.datum.model_vs_close_logloss_devig < 0,
            alt.value("#1a7a3a"),
            alt.value("#a33333")
        ),
        tooltip=[
            alt.Tooltip("game_label:N",       title="Game"),
            alt.Tooltip("model_vs_close_logloss_devig:Q", title="Log loss delta", format="+.4f"),
            alt.Tooltip("game_num:Q",         title="Game #"),
        ]
    )

    st.altair_chart(
        (zero_line + recent_band + points + trend_line_mean + trend_line_median + recent_label).properties(height=400),
        use_container_width=True
    )
    st.caption(
        f"Green dots = model beat market that game. Red dots = market was better. "
        f"Blue line = cumulative median (typical game). "
        f"Orange dashed = cumulative mean (includes outliers). "
        f"Gap between lines = outlier drag. Blue shaded band = last {window} games."
    )


def render_performance(hist: pd.DataFrame):
    st.subheader("Model vs Market Performance")
    st.caption(
        "Grades the model as a pricing engine against devigged Pinnacle closing probabilities. "
        "Negative log loss delta means your model priced the outcome better than the market."
    )

    scored = prepare_scored_subset(hist)
    if scored.empty:
        st.info("No scored games yet.")
        return

    avg_abs_prob_diff = scored["abs_prob_diff_close"].mean() if "abs_prob_diff_close" in scored.columns else np.nan
    st.caption(
        f"Average absolute probability gap versus Pinnacle close: {avg_abs_prob_diff:.4f}"
        if pd.notna(avg_abs_prob_diff) else
        "Average absolute probability gap versus Pinnacle close: —"
    )

    st.markdown("---")
    render_beating_the_market(scored)

    with st.expander("Outcome-based diagnostics"):
        st.caption("Secondary metrics describing how model probabilities aligned with actual outcomes.")
        mae          = scored["mae_total"].mean()        if "mae_total"          in scored.columns else np.nan
        brier        = scored["brier"].mean()            if "brier"              in scored.columns else np.nan
        model_logloss= scored["logloss"].mean()          if "logloss"            in scored.columns else np.nan
        close_logloss= scored["close_logloss_devig"].mean() if "close_logloss_devig" in scored.columns else np.nan

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("MAE (Runs)",      f"{mae:.3f}"         if pd.notna(mae)           else "—")
        d2.metric("Brier Score",     f"{brier:.4f}"       if pd.notna(brier)         else "—")
        d3.metric("Model Log Loss",  f"{model_logloss:.4f}" if pd.notna(model_logloss) else "—")
        d4.metric("Close Log Loss",  f"{close_logloss:.4f}" if pd.notna(close_logloss) else "—")

    st.markdown("---")
    st.markdown("#### Closing-Line Comparison Archive")
    display_cols = [
        "date", "away_team", "away_sp", "home_team", "home_sp",
        "winprob_home", "close_home_prob_devig", "prob_diff_close", "abs_prob_diff_close",
        "model_fv_home", "close_fv_home",
        "home_abs_devig_close_diff", "model_vs_close_logloss_devig"
    ]
    available = [c for c in display_cols if c in scored.columns]
    show = scored[available].sort_values("date", ascending=False).copy()
    for col in ["prob_diff_close", "abs_prob_diff_close", "home_abs_devig_close_diff", "model_vs_close_logloss_devig"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(4)
    st.dataframe(show, use_container_width=True)


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
