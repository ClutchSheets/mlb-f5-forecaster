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
    height = max(980, len(df) * 78 + 90)
    components.html(html, height=height, scrolling=True)


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
        scored = scored[(scored["winprob_home"] > 0.05) & (scored["winprob_home"] < 0.95)].copy()

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


def render_close_calibration(df: pd.DataFrame):
    st.markdown("#### Calibration Against Closing Line (10% Buckets)")
    st.caption("Blue dots compare your model buckets to average devigged Pinnacle close. The dashed line is perfect agreement with close.")

    calib = df.dropna(subset=["winprob_home", "close_home_prob_devig"]).copy()
    if calib.empty:
        st.info("Not enough data for close calibration.")
        return

    bins = np.linspace(0, 1, 11)
    calib["bucket"] = pd.cut(calib["winprob_home"], bins=bins, include_lowest=True)

    calib_table = (
        calib.groupby("bucket", observed=False)
        .agg(
            avg_model=("winprob_home", "mean"),
            avg_close=("close_home_prob_devig", "mean"),
            count=("close_home_prob_devig", "count"),
        )
        .reset_index()
    )

    calib_table = calib_table[calib_table["count"] > 0].copy()
    if calib_table.empty:
        st.info("Not enough populated buckets for close calibration.")
        return

    calib_table["bucket_label"] = calib_table["bucket"].apply(format_bucket_label)

    diagonal = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    diag_line = alt.Chart(diagonal).mark_line(
        strokeDash=[4, 4], color="gray"
    ).encode(
        x=alt.X("x:Q", title="Model projected probability", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", values=[i / 20 for i in range(21)])),
        y=alt.Y("y:Q", title="Average devigged Pinnacle close", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
    )

    dots = alt.Chart(calib_table).mark_circle(
        color="#378AD3", opacity=0.85, stroke="#185FA5", strokeWidth=1.5
    ).encode(
        x=alt.X("avg_model:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", values=[i / 20 for i in range(21)])),
        y=alt.Y("avg_close:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        size=alt.Size("count:Q", scale=alt.Scale(range=[80, 450]), legend=None),
        tooltip=[
            alt.Tooltip("bucket_label:N", title="Bucket"),
            alt.Tooltip("avg_model:Q", format=".3f", title="Avg model"),
            alt.Tooltip("avg_close:Q", format=".3f", title="Avg close"),
            alt.Tooltip("count:Q", title="Games"),
        ],
    )

    st.altair_chart((diag_line + dots).properties(height=340), use_container_width=True)

    show_table = calib_table[["bucket_label", "avg_model", "avg_close", "count"]].copy()
    show_table.columns = ["bucket", "avg_model", "avg_close", "count"]
    show_table["avg_model"] = show_table["avg_model"].round(4)
    show_table["avg_close"] = show_table["avg_close"].round(4)
    st.dataframe(show_table, use_container_width=True)


def render_actual_calibration_comparison(df: pd.DataFrame):
    st.markdown("#### Model vs Market Calibration Against Actual Results")
    st.caption("Blue dots are your model buckets against actual win rate. Purple dots are the market's devigged closing buckets against actual win rate.")

    calib = df.dropna(subset=["winprob_home", "close_home_prob_devig", "home_win_f5"]).copy()
    if calib.empty:
        st.info("Not enough data for actual-result calibration.")
        return

    bins = np.linspace(0.5, 1.0, 6)  # 50–60, 60–70, 70–80, 80–90, 90–100

    # Model calibration: normalize to model favorite
    model_df = calib.copy()
    model_df["pred_prob"] = model_df["winprob_home"]
    model_df["actual"] = model_df["home_win_f5"]

    model_flip = model_df["pred_prob"] < 0.5
    model_df.loc[model_flip, "pred_prob"] = 1 - model_df.loc[model_flip, "pred_prob"]
    model_df.loc[model_flip, "actual"] = 1 - model_df.loc[model_flip, "actual"]

    model_df["bucket"] = pd.cut(model_df["pred_prob"], bins=bins, include_lowest=True)
    model_table = (
        model_df.groupby("bucket", observed=False)
        .agg(
            avg_pred=("pred_prob", "mean"),
            actual_win=("actual", "mean"),
            count=("actual", "count"),
        )
        .reset_index()
    )
    model_table = model_table[model_table["count"] > 0].copy()
    model_table["series"] = "Model"
    model_table["bucket_label"] = model_table["bucket"].apply(format_bucket_label)

    # Market calibration: normalize to market favorite
    market_df = calib.copy()
    market_df["pred_prob"] = market_df["close_home_prob_devig"]
    market_df["actual"] = market_df["home_win_f5"]

    market_flip = market_df["pred_prob"] < 0.5
    market_df.loc[market_flip, "pred_prob"] = 1 - market_df.loc[market_flip, "pred_prob"]
    market_df.loc[market_flip, "actual"] = 1 - market_df.loc[market_flip, "actual"]

    market_df["bucket"] = pd.cut(market_df["pred_prob"], bins=bins, include_lowest=True)
    market_table = (
        market_df.groupby("bucket", observed=False)
        .agg(
            avg_pred=("pred_prob", "mean"),
            actual_win=("actual", "mean"),
            count=("actual", "count"),
        )
        .reset_index()
    )
    market_table = market_table[market_table["count"] > 0].copy()
    market_table["series"] = "Market Close"
    market_table["bucket_label"] = market_table["bucket"].apply(format_bucket_label)

    combined = pd.concat([model_table, market_table], ignore_index=True)
    if combined.empty:
        st.info("Not enough populated buckets for actual-result calibration.")
        return

    diagonal = pd.DataFrame({"x": [0.5, 1.0], "y": [0.5, 1.0]})

    diag_line = alt.Chart(diagonal).mark_line(
        strokeDash=[4, 4], color="gray"
    ).encode(
        x=alt.X(
            "x:Q",
            title="Projected probability bucket average",
            scale=alt.Scale(domain=[0.5, 1.0]),
            axis=alt.Axis(format=".0%", values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ),
        y=alt.Y(
            "y:Q",
            title="Actual win rate",
            scale=alt.Scale(domain=[0.0, 1.0]),
            axis=alt.Axis(format=".0%"),
        ),
    )

    dots = alt.Chart(combined).mark_circle(opacity=0.85, strokeWidth=1.5).encode(
        x=alt.X(
            "avg_pred:Q",
            scale=alt.Scale(domain=[0.5, 1.0]),
            axis=alt.Axis(format=".0%", values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ),
        y=alt.Y(
            "actual_win:Q",
            scale=alt.Scale(domain=[0.0, 1.0]),
            axis=alt.Axis(format=".0%"),
        ),
        size=alt.Size("count:Q", scale=alt.Scale(range=[80, 450]), legend=None),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=["Model", "Market Close"], range=["#378AD3", "#7A3FB0"]),
            legend=alt.Legend(title="")
        ),
        tooltip=[
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("bucket_label:N", title="Bucket"),
            alt.Tooltip("avg_pred:Q", format=".3f", title="Avg projected"),
            alt.Tooltip("actual_win:Q", format=".3f", title="Actual win rate"),
            alt.Tooltip("count:Q", title="Games"),
        ],
    )

    st.altair_chart((diag_line + dots).properties(height=360), use_container_width=True)

    show_table = combined[["series", "bucket_label", "avg_pred", "actual_win", "count"]].copy()
    show_table.columns = ["series", "bucket", "avg_pred", "actual_win", "count"]
    show_table["avg_pred"] = show_table["avg_pred"].round(4)
    show_table["actual_win"] = show_table["actual_win"].round(4)
    st.dataframe(show_table, use_container_width=True)


def render_accuracy_over_time(df: pd.DataFrame):
    st.markdown("#### Model Accuracy Versus Closing Line Over Time")

    if df.empty:
        st.info("Not enough data for accuracy chart.")
        return

    plot_df = df.copy()
    plot_df["game_num"] = range(1, len(plot_df) + 1)

    series = []
    if "model_vs_close_logloss_devig" in plot_df.columns:
        plot_df["roll_logloss_delta"] = plot_df["model_vs_close_logloss_devig"].expanding().mean()
        series.append(
            plot_df[["game_num", "roll_logloss_delta"]].rename(columns={"roll_logloss_delta": "value"}).assign(metric="Cumulative Log Loss Delta")
        )
    if "abs_prob_diff_close" in plot_df.columns:
        plot_df["roll_abs_prob_diff"] = plot_df["abs_prob_diff_close"].expanding().mean()
        series.append(
            plot_df[["game_num", "roll_abs_prob_diff"]].rename(columns={"roll_abs_prob_diff": "value"}).assign(metric="Cumulative Abs Prob Diff")
        )

    if not series:
        st.info("Not enough data for accuracy chart.")
        return

    chart_df = pd.concat(series, ignore_index=True)

    line = alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X("game_num:Q", title="Game #"),
        y=alt.Y("value:Q", title="Value"),
        color=alt.Color("metric:N", title="Metric"),
        tooltip=["metric", alt.Tooltip("value:Q", format=".4f"), "game_num"]
    )

    rules = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    st.altair_chart((line + rules).properties(height=320), use_container_width=True)


def render_performance(hist: pd.DataFrame):
    st.subheader("Projection Accuracy Versus Pinnacle Close")
    st.caption("This page grades the model as a pricing engine. It compares your projected probabilities to devigged Pinnacle closing probabilities. It does not measure whether a bet beat the market, because no wager was assumed.")

    scored = prepare_scored_subset(hist)
    if scored.empty:
        st.info("No scored games yet.")
        return

    total = len(scored)
    close_coverage = scored["close_logloss_devig"].notna().mean() * 100 if "close_logloss_devig" in scored.columns else 0.0
    avg_abs_devig = scored["home_abs_devig_close_diff"].mean() if "home_abs_devig_close_diff" in scored.columns else np.nan
    median_abs_devig = scored["home_abs_devig_close_diff"].median() if "home_abs_devig_close_diff" in scored.columns else np.nan
    avg_logloss_delta = scored["model_vs_close_logloss_devig"].mean() if "model_vs_close_logloss_devig" in scored.columns else np.nan
    median_logloss_delta = scored["model_vs_close_logloss_devig"].median() if "model_vs_close_logloss_devig" in scored.columns else np.nan
    avg_abs_prob_diff = scored["abs_prob_diff_close"].mean() if "abs_prob_diff_close" in scored.columns else np.nan

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Games Scored", total)
    c2.metric("Close Coverage", f"{close_coverage:.1f}%")
    c3.metric("Avg Abs Devig Diff", f"{avg_abs_devig:.4f}" if pd.notna(avg_abs_devig) else "—")
    c4.metric("Median Abs Devig Diff", f"{median_abs_devig:.4f}" if pd.notna(median_abs_devig) else "—")
    c5.metric("Avg Log Loss Delta", f"{avg_logloss_delta:+.4f}" if pd.notna(avg_logloss_delta) else "—")
    c6.metric("Median Log Loss Delta", f"{median_logloss_delta:+.4f}" if pd.notna(median_logloss_delta) else "—")

    st.caption(f"Average absolute probability gap versus close: {avg_abs_prob_diff:.4f}" if pd.notna(avg_abs_prob_diff) else "Average absolute probability gap versus close: —")
    st.caption("Smaller absolute probability differences mean the model is pricing closer to Pinnacle close. Log loss delta near zero means the model is tracking close well; positive values mean it is less accurate than close.")

    st.markdown("---")
    render_accuracy_over_time(scored)

    st.markdown("---")
    render_close_calibration(scored)

    st.markdown("---")
    render_actual_calibration_comparison(scored)

    with st.expander("Outcome-based diagnostics"):
        st.caption("These remain secondary. They describe how model probabilities aligned with actual outcomes, but the primary goal above is closing-line estimation accuracy.")
        mae = scored["mae_total"].mean() if "mae_total" in scored.columns else np.nan
        brier = scored["brier"].mean() if "brier" in scored.columns else np.nan
        model_logloss = scored["logloss"].mean() if "logloss" in scored.columns else np.nan
        close_logloss = scored["close_logloss_devig"].mean() if "close_logloss_devig" in scored.columns else np.nan

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("MAE (Runs)", f"{mae:.3f}" if pd.notna(mae) else "—")
        d2.metric("Brier Score", f"{brier:.4f}" if pd.notna(brier) else "—")
        d3.metric("Model Log Loss", f"{model_logloss:.4f}" if pd.notna(model_logloss) else "—")
        d4.metric("Close Log Loss", f"{close_logloss:.4f}" if pd.notna(close_logloss) else "—")

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
