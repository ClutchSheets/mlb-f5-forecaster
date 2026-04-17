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


def highlight_color(edge: float) -> str | None:
    if edge < EDGE_THRESHOLD:
        return None
    if edge < 7.0:
        return "#2e7d32"
    if edge < 10.0:
        return "#1f6f2d"
    return "#145a22"


def format_highlight(odds: int, edge: float, color: str) -> str:
    odds_str = fmt_american(odds)
    return (
        f"<span class='edge' style='background:{color}'>"
        f"{odds_str}"
        f"<small>Edge {edge:+.1f}%</small>"
        f"</span>"
    )


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
        lookup[code_key] = game
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

def load_edge_clv_summary(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT threshold, bets, avg_clv, beat_close, updated_at
            FROM edge_clv_summary
            ORDER BY threshold
            """,
            conn,
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def load_edge_clv_bets(db_path: str, threshold: float = 0.05) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT
                date,
                away_team,
                home_team,
                pick_side,
                edge,
                odds_taken_american,
                close_prob_devig,
                clv,
                beat_close,
                result
            FROM edge_clv_bets
            WHERE threshold = ?
            ORDER BY date DESC, away_team, home_team
            """,
            conn,
            params=[threshold],
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def render_edge_clv_section(db_path: str):
    st.subheader("Beating the Market")
    st.caption(
        "Threshold report based on stored 6-hour prices. CLV is the difference between the devigged Pinnacle closing implied probability and the implied probability of the odds taken."
    )

    summary = load_edge_clv_summary(db_path)
    if summary.empty:
        st.info("No edge/CLV summary table found yet. Run grade_edge_clv.py first.")
        return

    show = summary.copy()
    show["threshold_label"] = show["threshold"].map(lambda x: f"Edge≥ {x*100:.0f}%")
    show["avg_clv_pct"] = show["avg_clv"] * 100
    show["beat_close_display"] = show["beat_close"] * 100

    row5 = show.loc[show["threshold"] == 0.05]

    c1, c2, c3 = st.columns(3)
    if not row5.empty:
        r = row5.iloc[0]
        c1.metric("Qualified Bets", int(r["bets"]))
        c2.metric("Avg CLV", f"{r['avg_clv_pct']:.2f}%")
        c3.metric("Beat Close", f"{r['beat_close_display']:.1f}%")
    else:
        c1.metric("Qualified Bets", "—")
        c2.metric("Avg CLV", "—")
        c3.metric("Beat Close", "—")

    disp = show[["threshold_label", "bets", "avg_clv_pct", "beat_close_display", "updated_at"]].rename(
        columns={
            "threshold_label": "Threshold",
            "bets": "Bets",
            "avg_clv_pct": "Avg CLV %",
            "beat_close_display": "Beat Close %",
            "updated_at": "Updated",
        }
    )
    st.dataframe(disp, use_container_width=True, hide_index=True)

    bets = load_edge_clv_bets(db_path, 0.05)
    if not bets.empty:
        st.markdown("#### Edge ≥ 5% Detail")
        bets = bets.copy()
        bets["Matchup"] = bets["away_team"] + " @ " + bets["home_team"]
        bets["Edge %"] = bets["edge"] * 100
        bets["Odds Taken"] = bets["odds_taken_american"].astype(int)
        bets["Taken Implied %"] = bets["odds_taken_american"].apply(lambda x: american_to_prob(int(x)) * 100)
        bets["Close Fair Odds"] = bets["close_prob_devig"].apply(lambda x: prob_to_american(float(x)))
        bets["Close Fair %"] = bets["close_prob_devig"] * 100
        bets["CLV %"] = bets["clv"] * 100
        bets["Beat Close"] = bets["beat_close"].map({1: "Yes", 0: "No"})
        bets["Result"] = bets["result"].map({1: "Win", 0: "Loss", 0.5: "Push"}).fillna(bets["result"])

        detail = bets[
            [
                "date",
                "Matchup",
                "pick_side",
                "Edge %",
                "Odds Taken",
                "Taken Implied %",
                "Close Fair Odds",
                "Close Fair %",
                "CLV %",
                "Beat Close",
                "Result",
            ]
        ].rename(
            columns={
                "date": "Date",
                "pick_side": "Pick",
            }
        )

        st.dataframe(detail, use_container_width=True, hide_index=True)


def get_book_odds(book_key, game, away, home):
    entry = game.get("ml", {}).get(book_key)
    if not entry:
        return None, None
    return entry.get("away"), entry.get("home")


def get_book_metrics(model_p: float, side_odds: int, opp_odds: int):
    decimal_odds = american_to_decimal(side_odds)
    adjk = adj_kelly(model_p, decimal_odds)
    if opp_odds is not None:
        market_p, _ = no_vig_prob(side_odds, opp_odds)
    else:
        market_p = american_to_prob(side_odds)
    edge = edge_pct(model_p, market_p)
    return edge, adjk


def render_metric_cell(odds: int, edge: float, td_style: str) -> str:
    color = highlight_color(edge)
    if color is None:
        return f"<td style='{td_style}'><span style='color:#e0e0e0;font-size:0.9em'>{fmt_american(odds)}</span></td>"
    return f"<td style='{td_style}'>{format_highlight(odds, edge, color)}</td>"


def render_games_table(df, odds_lookup):
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
        tr.game-odd:hover td,tr.game-even:hover td{{filter:brightness(1.08)}}
        .edge{{border-radius:4px;padding:2px 7px;white-space:nowrap;display:inline-block;font-weight:600;color:#fff;line-height:1.25}}
        .edge small{{display:block;font-size:0.75em;font-weight:500;opacity:0.96}}
        td{{padding:4px 12px;border-bottom:1px solid {BORDER}}}
      </style>
    """

    TH = "padding:10px 12px;font-size:0.72em;font-weight:700;color:#fff;letter-spacing:0.08em;text-transform:uppercase;background:#111418;border-bottom:2px solid #3a4050"
    html = css + f"""<table>
      <thead>
        <tr>
          <th style='{TH};text-align:center;width:30px'>#</th>
          <th style='{TH};text-align:left'>TEAM / PITCHER</th>
          <th style='{TH};text-align:center'>TIME</th>
          <th style='{TH};text-align:center'>WIN %</th>
          <th style='{TH};text-align:center'>ML FV</th>
          <th style='{TH};text-align:center'>EDGE</th>
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
        away_display = game.get("away", away)
        home_display = game.get("home", home)
        commence_edt = game.get("commence_edt", "") or "—"

        away_cells = ""
        home_cells = ""
        td_style = f"padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}"
        away_best_edge = None
        home_best_edge = None

        for bk_key in BOOKS:
            away_odds, home_odds = get_book_odds(bk_key, game, away, home)
            if away_odds is None or home_odds is None:
                away_cells += f"<td style='{td_style}'><span style='color:#888'>—</span></td>"
                home_cells += f"<td style='{td_style}'><span style='color:#888'>—</span></td>"
                continue

            away_edge, away_adjk = get_book_metrics(model_away_p, away_odds, home_odds)
            home_edge, home_adjk = get_book_metrics(model_home_p, home_odds, away_odds)

            away_cells += render_metric_cell(away_odds, away_edge, td_style)
            home_cells += render_metric_cell(home_odds, home_edge, td_style)

            if away_best_edge is None or away_edge > away_best_edge:
                away_best_edge = away_edge
            if home_best_edge is None or home_edge > home_best_edge:
                home_best_edge = home_edge

        away_edge_cell = f"<td style='{td_style}'><span style='color:#e0e0e0;font-size:0.9em'>{away_best_edge:+.1f}%</span></td>" if away_best_edge is not None else f"<td style='{td_style}'><span style='color:#888'>—</span></td>"
        home_edge_cell = f"<td style='{td_style}'><span style='color:#e0e0e0;font-size:0.9em'>{home_best_edge:+.1f}%</span></td>" if home_best_edge is not None else f"<td style='{td_style}'><span style='color:#888'>—</span></td>"

        game_class = "game-odd" if game_num % 2 == 1 else "game-even"
        html += f"""
        <tr class='{game_class}'>
          <td style='padding:4px 6px;text-align:center;border-bottom:1px solid {BORDER};vertical-align:bottom' rowspan='2'>
            <span style='color:#888;font-size:0.8em'>{game_num}</span>
          </td>
          <td style='padding:4px 12px;border-bottom:1px solid {BORDER}'>
            <span style='font-weight:600;color:#e0e0e0'>{away_display}</span><br>
            <span style='color:#888;font-size:0.78em'>{away_sp}</span>
          </td>
          <td style='padding:4px 12px;text-align:center;border-bottom:2px solid {BORDER};vertical-align:middle' rowspan='2'>
            <span style='color:#e0e0e0;font-size:0.9em'>{commence_edt}</span>
          </td>
          <td style='padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}'>
            <span style='color:#e0e0e0;font-size:0.9em'>{model_away_p*100:.1f}%</span>
          </td>
          <td style='padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}'>
            <span style='color:#7ec8e3;font-size:0.9em;font-weight:600'>{ml_fv_away}</span><br><span style='color:#888;font-size:0.75em'>(PIN {game.get('pinnacle', {}).get('ml', {}).get('away','—')})</span>
          </td>
          {away_edge_cell}
          {away_cells}
        </tr>
        <tr class='{game_class}'>
          <td style='padding:4px 12px;border-bottom:2px solid {BORDER}'>
            <span style='font-weight:600;color:#e0e0e0'>{home_display}</span><br>
            <span style='color:#888;font-size:0.78em'>{home_sp}</span>
          </td>
          <td style='padding:4px 12px;text-align:center;border-bottom:2px solid {BORDER}'>
            <span style='color:#e0e0e0;font-size:0.9em'>{model_home_p*100:.1f}%</span>
          </td>
          <td style='padding:4px 12px;text-align:center;border-bottom:2px solid {BORDER}'>
            <span style='color:#7ec8e3;font-size:0.9em;font-weight:600'>{ml_fv_home}</span><br><span style='color:#888;font-size:0.75em'>(PIN {game.get('pinnacle', {}).get('ml', {}).get('home','—')})</span>
          </td>
          {home_edge_cell}
          {home_cells}
        </tr>
        """

    html += "</tbody></table>"
    height = len(df) * 104 + 60
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
    plot_df["cumulative_mean"] = plot_df["model_vs_close_logloss_devig"].expanding().mean()
    plot_df["game_label"] = plot_df["away_team"] + " @ " + plot_df["home_team"] + " (" + plot_df["date"] + ")"

    median_delta = plot_df["model_vs_close_logloss_devig"].median()
    mean_delta = plot_df["model_vs_close_logloss_devig"].mean()
    outlier_drag = mean_delta - median_delta
    n = len(plot_df)

    window = min(50, n)
    recent_df = plot_df.tail(window)
    recent_avg = recent_df["model_vs_close_logloss_devig"].median()
    trend_diff = recent_avg - median_delta

    if median_delta < 0:
        overall_txt = f"Overall your model is <b>ahead of the market</b> by {abs(median_delta):.4f} median log loss per game across {n} games."
    else:
        overall_txt = f"Overall your model is <b>behind the market</b> by {abs(median_delta):.4f} median log loss per game across {n} games."

    if abs(trend_diff) < 0.005:
        trend_txt = "Your last {window} games are flat — performance is holding steady.".format(window=window)
        trend_color = "#666"
    elif trend_diff < 0:
        trend_txt = (
            f"Your last {window} games are <b>trending better</b> "
            f"(recent median {recent_avg:+.4f} vs overall {median_delta:+.4f}) — you're closing the gap on the market."
        )
        trend_color = "#1a7a3a"
    else:
        trend_txt = (
            f"Your last {window} games are <b>trending worse</b> "
            f"(recent median {recent_avg:+.4f} vs overall {median_delta:+.4f}) — the gap to the market is widening."
        )
        trend_color = "#a33333"

    st.markdown(
        f"<p style=\"margin:0 0 4px\">{overall_txt}</p>"
        f"<p style=\"margin:0;color:{trend_color}\">{trend_txt}</p>",
        unsafe_allow_html=True,
    )
    st.caption("Log loss delta = model log loss minus Pinnacle closing log loss. Negative = your model priced the outcome better. Lower is better. Blue line = cumulative median (robust to outliers).")

    st.markdown("")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games Scored", n)
    c2.metric("Median Log Loss Delta", f"{median_delta:+.4f}", help="Typical game — robust to outliers")
    c3.metric("Mean Log Loss Delta", f"{mean_delta:+.4f}", help="Overall calibration — includes blowups")
    c4.metric("Outlier Drag", f"{outlier_drag:+.4f}", help="Mean minus median. Positive = bad outlier games pulling mean up. Investigate these.")
    c5.metric(f"Last {window} Median", f"{recent_avg:+.4f}", delta=f"{trend_diff:+.4f} vs overall", delta_color="inverse")

    st.markdown("")

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        strokeDash=[4, 4], color="gray", opacity=0.5
    ).encode(y="y:Q")

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
        y=alt.Y("cumulative_median:Q", title="Log loss delta (negative = beating market)", axis=alt.Axis(format=".3f")),
        tooltip=[
            alt.Tooltip("game_label:N", title="Game"),
            alt.Tooltip("cumulative_median:Q", title="Cumulative median", format="+.4f"),
            alt.Tooltip("cumulative_mean:Q", title="Cumulative mean", format="+.4f"),
            alt.Tooltip("model_vs_close_logloss_devig:Q", title="This game", format="+.4f"),
            alt.Tooltip("game_num:Q", title="Game #"),
        ]
    )

    trend_line_mean = alt.Chart(plot_df).mark_line(
        color="#f0a500", strokeWidth=1.5, strokeDash=[4, 3], opacity=0.7
    ).encode(
        x=alt.X("game_num:Q"),
        y=alt.Y("cumulative_mean:Q"),
        tooltip=[
            alt.Tooltip("game_label:N", title="Game"),
            alt.Tooltip("cumulative_mean:Q", title="Cumulative mean", format="+.4f"),
            alt.Tooltip("cumulative_median:Q", title="Cumulative median", format="+.4f"),
            alt.Tooltip("game_num:Q", title="Game #"),
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
            alt.Tooltip("game_label:N", title="Game"),
            alt.Tooltip("model_vs_close_logloss_devig:Q", title="Log loss delta", format="+.4f"),
            alt.Tooltip("game_num:Q", title="Game #"),
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
    model_logloss = scored["logloss"].mean() if "logloss" in scored.columns else np.nan
    close_logloss = scored["close_logloss_devig"].mean() if "close_logloss_devig" in scored.columns else np.nan

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Avg Abs Prob Gap vs Close",
        f"{avg_abs_prob_diff:.4f}" if pd.notna(avg_abs_prob_diff) else "—",
        help="Average absolute difference between your model probability and devigged Pinnacle closing probability. Lower is better.",
    )
    m2.metric(
        "Model Log Loss",
        f"{model_logloss:.4f}" if pd.notna(model_logloss) else "—",
        help="Outcome-based log loss from your model probabilities.",
    )
    m3.metric(
        "Close Log Loss",
        f"{close_logloss:.4f}" if pd.notna(close_logloss) else "—",
        help="Outcome-based log loss from devigged Pinnacle closing probabilities.",
    )

    st.markdown("---")
    render_beating_the_market(scored)

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

    fetched_at = None
    odds_file_date = None
    if os.path.exists(DAILY_ODDS_FILE):
        try:
            with open(DAILY_ODDS_FILE, encoding="utf-8") as f:
                odds_payload = json.load(f)
            fetched_at = odds_payload.get("fetched_at")
            odds_file_date = odds_payload.get("date")
        except Exception:
            fetched_at = None
            odds_file_date = None

    CSV_PATH = os.environ.get("FORECAST_CSV", "daily_forecast.csv")
    DB_PATH = "mlb_f5_forecasts.db"
    st.caption(f"DB file in use: {DB_PATH}")

    tab1, tab2 = st.tabs(["📅 Today's Games", "📊 Performance"])

    with tab1:
        st.caption(
            "Edge = best available model edge vs market price\n"
            "AdjK = adjusted Kelly score (combines edge and payout quality)\n"
            "Only prices with AdjK ≥ 1.5 are highlighted\n\n"
            "F5 markets are more volatile than full game markets.\n"
            "Small edges are often noise, so an AdjK threshold is applied to isolate higher-quality opportunities."
        )

        if fetched_at:
            date_line = f"<div style='color:#8f9bad;font-size:0.82em;margin-top:6px'>Odds file date: {odds_file_date}</div>" if odds_file_date else ""
            st.markdown(
                f"""
                <div style="
                    background:#0e1117;
                    padding:12px 16px;
                    border-radius:10px;
                    display:inline-block;
                    margin: 2px 0 14px 0;
                ">
                    <div style="color:#a7b0c0;font-size:0.75em;letter-spacing:0.08em">
                        LAST UPDATED
                    </div>
                    <div style="color:#ffffff;font-size:1.12em;font-weight:700;margin-top:4px">
                        {fetched_at}
                    </div>
                    {date_line}
                </div>
                """,
                unsafe_allow_html=True
            )

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
        render_edge_clv_section(DB_PATH)
        st.markdown("---")
        hist = load_history(DB_PATH)
        render_performance(hist)


if __name__ == "__main__":
    main()
